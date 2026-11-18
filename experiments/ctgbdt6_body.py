
# CT-GBDT v6 context branch. Candidates: flow, flow + relational context (v5), and flow + the temporal groups
# recommended by the temporal lab (LAB_PARTS, chosen on train/validation/calibration only). PRE-REGISTERED RULE: among
# candidates that beat flow-only by >= 0.002 macro-MCC on BOTH validation and calibration (each with its val-chosen
# decision rule), pick the one maximising min(gain_val, gain_cal); if none qualifies, the frozen CT-GBDT v4 stays.
# The chosen set is fitted with every seed; the output format equals the CT-GBDT branch read by the post-hoc.
OUT = "/data/output/context_gbdt"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260926)
MAX_TRAIN = 3_000_000
started = time.time()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


NEED_T = [g for g in (LAB_PARTS or []) if g in GROUPS]  # temporal groups are computed only if the lab recommends them


def build(split, sample=None, ids=False):
    f = load(split, ids=ids)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    if NEED_T:
        T, t_names, t_groups = temporal(f, rows, STATE_CODES, groups=NEED_T)
    else:
        T, t_names, t_groups = np.zeros((len(rows), 0), np.float32), [], []
    X = np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), rel, T], 1)
    extra = {"stream": f["stream"].to_numpy()[rows]}
    if ids:
        extra["ids"] = f["source_id"].to_numpy()[rows].astype(str)
    del f, rel, T
    log("built", split, X.shape, round(time.time() - started), "s")
    return X, y[rows], extra, list(relational_names), t_names, t_groups


prior_counts = None


def train_sample(y):
    global prior_counts
    n = np.bincount(y, minlength=K)
    prior_counts = n
    rate = np.minimum(1.0, np.maximum(MAX_TRAIN / len(y), (MAX_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


STATE_CODES = top_state_codes(load("train")) if STATE else []
Xtr, ytr, _, REL, TN, TG = build("train", train_sample)
wtr = train_sample.weight
prior = prior_counts / prior_counts.sum()
Xva, yva, eva, *_ = build("val", ids=True)
Xca, yca, _, *_ = build("calibration")
nf, nr = len(FLOW), len(REL)
cols = {"flow": list(range(nf)), "REL": list(range(nf, nf + nr))}
for g in GROUPS:
    cols[g] = [nf + nr + i for i, gg in enumerate(TG) if gg == g]
CANDIDATES = {"flow": ["flow"], "flow+relational": ["flow", "REL"]}
if LAB_PARTS and sorted(LAB_PARTS) not in (["flow"], ["REL", "flow"]):
    CANDIDATES["flow+lab_temporal"] = list(LAB_PARTS)
rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, 1e-12)).argmax(1)}


def macro(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    return float(mcc[t > 0].mean())


def predict(booster, X, c, chunk=400_000):
    return np.concatenate([booster.predict(X[i:i + chunk][:, c]) for i in range(0, len(X), chunk)])


def fit(c, seed):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": seed}
    return lgb.train(params, lgb.Dataset(Xtr[:, c], ytr, weight=wtr), num_boost_round=300)


report = {"model": "CT-GBDT v6 (flow / flow + relational / flow + lab temporal; double held-out maximin rule)",
          "lab_parts": LAB_PARTS, "candidates": CANDIDATES, "classes": classes, "selection": {}}
fitted = {}
for name, parts in CANDIDATES.items():
    c = sorted({j for p in parts for j in cols[p]})
    booster = fit(c, SEEDS[0])
    res = {s: {r: macro(y, f(predict(booster, X, c))) for r, f in rules.items()} for s, X, y in (("val", Xva, yva), ("calibration", Xca, yca))}
    rule = max(rules, key=lambda r: res["val"][r])
    report["selection"][name] = {"validation": res["val"], "calibration": res["calibration"], "rule_on_val": rule, "features": len(c)}
    fitted[name] = (booster, c)
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    log("candidate", name, len(c), {s: {r: round(v, 4) for r, v in d.items()} for s, d in res.items()})
f_ = report["selection"]["flow"]
qualified = {}
for name, r_ in report["selection"].items():
    if name == "flow":
        continue
    gv = r_["validation"][r_["rule_on_val"]] - f_["validation"][f_["rule_on_val"]]
    gc = r_["calibration"][r_["rule_on_val"]] - f_["calibration"][f_["rule_on_val"]]
    r_["gain_validation"], r_["gain_calibration"] = gv, gc
    if gv >= 0.002 and gc >= 0.002:
        qualified[name] = min(gv, gc)
chosen = max(qualified, key=qualified.get) if qualified else "flow"
report.update(chosen_feature_set=chosen, rule="maximin of (gain_val, gain_cal) over candidates with both >= 0.002 vs flow")
json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
log("chosen", chosen, qualified)
if chosen == "flow":
    log("no candidate passed the double held-out rule; CT-GBDT v4 remains the context branch")
    raise SystemExit(0)
c = fitted[chosen][1]
boosters = [fitted[chosen][0]] + [fit(c, seed) for seed in SEEDS[1:]]
del Xtr, Xca
Xte, yte, ete, *_ = build("test", ids=True)
stream_names = np.asarray(sorted(STREAMS, key=STREAMS.get), dtype=str)
for seed, booster in zip(SEEDS, boosters):
    pv, pt = predict(booster, Xva, c), predict(booster, Xte, c)
    np.savez_compressed(os.path.join(OUT, f"ctgbdt_seed{seed}.npz"), val_ids=eva["ids"], val_y=yva.astype(np.int8),
                        val_proba=pv.astype(np.float32), test_ids=ete["ids"], test_y=yte.astype(np.int8),
                        test_proba=pt.astype(np.float32),
                        test_stream=stream_names[ete["stream"]], prior=prior)
    report[f"seed{seed}"] = {"validation_macro_mcc": {r: macro(yva, f(pv)) for r, f in rules.items()}}
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    log("seed", seed, "saved")
log("ctgbdt6 done")

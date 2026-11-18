
# CT-GBDT v8: MULTI-VIEW context branch (random-subspace ensemble; Ho, IEEE TPAMI 1998). One LightGBM per feature view
# (VIEWS: flow + relational, flow + segment-relative temporal z-scores, all features); the multi-view candidate is the
# unweighted mean of the views' class probabilities (no fitted weights). Motivation (screen kernels): single GBDTs on
# cross-capture data swing strongly with the feature set (adversarial AUC train vs holdout ~1), averaging views cancels it.
# PRE-REGISTERED RULE (as ctgbdt6/7): candidates flow, each single view, multiview; among those beating flow-only by
# >= 0.002 macro-MCC on BOTH validation and calibration (val-chosen decision rule), pick the maximiser of
# min(gain_val, gain_cal); if none qualifies, the frozen CT-GBDT v4 stays. Output format = the CT-GBDT branch files.
OUT = "/data/output/context_gbdt"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261001)
MAX_TRAIN = 1_500_000
started = time.time()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


ALL_T = list(GROUPS)
VIEW_PARTS = {"flow+REL": ["flow", "REL"], "flow+T_segz": ["flow", "T_segz"], "all": ["flow", "REL", *ALL_T]}
VIEWS = [v for v in VIEWS if v in VIEW_PARTS]
NEED_T = sorted({g for v in VIEWS for g in VIEW_PARTS[v] if g in GROUPS}, key=ALL_T.index)


def build(split, sample=None, ids=False):
    f = load(split, ids=ids)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    T, t_names, t_groups = temporal(f, rows, STATE_CODES, groups=NEED_T) if NEED_T else (np.zeros((len(rows), 0), np.float32), [], [])
    X = np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), rel, T], 1)
    extra = {"stream": f["stream"].to_numpy()[rows]}
    if ids:
        extra["ids"] = f["source_id"].to_numpy()[rows].astype(str)
    del f, rel, T
    log("built", split, X.shape, round(time.time() - started), "s")
    return X, y[rows], extra, list(relational_names), t_groups


prior_counts = None


def train_sample(y):
    global prior_counts
    n = np.bincount(y, minlength=K)
    prior_counts = n
    rate = np.minimum(1.0, np.maximum(MAX_TRAIN / len(y), (MAX_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


self_check()
STATE_CODES = top_state_codes(load("train")) if STATE else []
Xtr, ytr, _, REL, TG = build("train", train_sample)
wtr = train_sample.weight
prior = prior_counts / prior_counts.sum()
Xva, yva, eva, *_ = build("val", ids=True)
Xca, yca, _, *_ = build("calibration")
nf, nr = len(FLOW), len(REL)
cols = {"flow": list(range(nf)), "REL": list(range(nf, nf + nr))}
for g in GROUPS:
    cols[g] = [nf + nr + i for i, gg in enumerate(TG) if gg == g]
view_cols = {v: sorted({j for p in VIEW_PARTS[v] for j in cols.get(p, [])}) for v in VIEWS}
rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, 1e-12)).argmax(1)}


def macro(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    return float(mcc[t > 0].mean())


def predict(booster, X, c, chunk=400_000):
    return np.concatenate([booster.predict(X[i:i + chunk][:, c]) for i in range(0, len(X), chunk)]).astype(np.float32)


def fit(c, seed):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": seed}
    return lgb.train(params, lgb.Dataset(Xtr[:, c], ytr, weight=wtr), num_boost_round=250)


report = {"model": "CT-GBDT v8 (multi-view random-subspace context ensemble; double held-out maximin rule)",
          "views": {v: VIEW_PARTS[v] for v in VIEWS}, "classes": classes, "selection": {}}
boosters, probs = {}, {}
for name in ["flow", *VIEWS]:
    c = cols["flow"] if name == "flow" else view_cols[name]
    t0 = time.time()
    boosters[name] = fit(c, SEEDS[0])
    probs[name] = {s: predict(boosters[name], X, c) for s, X in (("val", Xva), ("calibration", Xca))}
    log("fitted", name, len(c), "features", round(time.time() - t0), "s")
probs["multiview"] = {s: np.mean([probs[v][s] for v in VIEWS], 0) for s in ("val", "calibration")}
for name, P in probs.items():
    res = {s: {r: macro(y, f_(P[s])) for r, f_ in rules.items()} for s, y in (("val", yva), ("calibration", yca))}
    rule = max(rules, key=lambda r: res["val"][r])
    report["selection"][name] = {"validation": res["val"], "calibration": res["calibration"], "rule_on_val": rule}
    log("candidate", name, {s: {r: round(v, 4) for r, v in d.items()} for s, d in res.items()})
f0 = report["selection"]["flow"]
qualified = {}
for name, r_ in report["selection"].items():
    if name == "flow":
        continue
    gv = r_["validation"][r_["rule_on_val"]] - f0["validation"][f0["rule_on_val"]]
    gc = r_["calibration"][r_["rule_on_val"]] - f0["calibration"][f0["rule_on_val"]]
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
members = VIEWS if chosen == "multiview" else [chosen]
models = {SEEDS[0]: {v: boosters[v] for v in members}}
for seed in SEEDS[1:]:
    models[seed] = {v: fit(view_cols[v], seed) for v in members}
    log("seed", seed, "fitted")
del Xtr, Xca, boosters
Xte, yte, ete, *_ = build("test", ids=True)
stream_names = np.asarray(sorted(STREAMS, key=STREAMS.get), dtype=str)
for seed, ms in models.items():
    pv = np.mean([predict(m, Xva, view_cols[v]) for v, m in ms.items()], 0)
    pt = np.mean([predict(m, Xte, view_cols[v]) for v, m in ms.items()], 0)
    np.savez_compressed(os.path.join(OUT, f"ctgbdt_seed{seed}.npz"), val_ids=eva["ids"], val_y=yva.astype(np.int8), val_proba=pv,
                        test_ids=ete["ids"], test_y=yte.astype(np.int8), test_proba=pt,
                        test_stream=stream_names[ete["stream"]], prior=prior)
    report[f"seed{seed}"] = {"validation_macro_mcc": {r: macro(yva, f_(pv)) for r, f_ in rules.items()}}
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    log("seed", seed, "saved")
log("ctgbdt8 done")

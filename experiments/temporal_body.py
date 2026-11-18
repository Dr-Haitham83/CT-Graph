
# TEMPORAL FEATURE LAB (train / validation / calibration only; the test split is never opened).
# Screens the temporal groups of temporal_lib against flow-only and flow + relational context with a fixed LightGBM.
# Screening rule (as for every earlier feature decision): a group passes iff it gains >= 0.002 macro-MCC on BOTH
# validation and calibration. The lab's recommended set maximises min(gain_val, gain_cal) over the tested sets with both
# gains >= 0.002; ctgbdt6 re-checks it at full size with the same rule before it can enter the model.
OUT = "/data/output/temporal"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260927)
LAB_TRAIN = 600_000
started = time.time()


def build(split, sample=None):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    rel_names = list(relational_names)
    T, t_names, t_groups = temporal(f, rows, STATE_CODES)
    X = np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), rel, T], 1)
    stream = f["stream"].to_numpy()[rows]
    del f, rel, T
    log("built", split, X.shape, round(time.time() - started), "s")
    return X, y[rows], stream, rel_names, t_names, t_groups


# relational() returns a DataFrame; remember its column names once
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


prior_counts = None


def train_sample(y):
    global prior_counts
    n = np.bincount(y, minlength=K)
    prior_counts = n
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(y), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep



self_check()

STATE_CODES = []
if STATE:
    STATE_CODES = top_state_codes(load("train"))
Xtr, ytr, _, REL, TN, TG = build("train", train_sample)
wtr = train_sample.weight
prior = prior_counts / prior_counts.sum()
Xva, yva, sva, *_ = build("val")
Xca, yca, sca, *_ = build("calibration")
nf, nr = len(FLOW), len(REL)
cols = {"flow": list(range(nf)), "REL": list(range(nf, nf + nr))}
for g in GROUPS:
    cols[g] = [nf + nr + i for i, gg in enumerate(TG) if gg == g]
ALLNAMES = FLOW + REL + TN
log("features", {k: len(v) for k, v in cols.items()})
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}


def scores(y, P):
    out = {}
    for rule, pred in (("bayes", P.argmax(1)), ("balanced", (P / np.maximum(prior, 1e-12)).argmax(1))):
        cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
        t, pp = cm.sum(1), cm.sum(0)
        tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(K), where=(2 * tp + fp + fn) > 0)
        present = t > 0
        out[rule] = {"macro_mcc": float(mcc[present].mean()),
                     "per_class_mcc": {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]},
                     "per_class_f1": {classes[i]: round(float(f1[i]), 4) for i in range(K) if present[i]}}
    return out


def predict(booster, X, c, chunk=400_000):
    return np.concatenate([booster.predict(X[i:i + chunk][:, c]) for i in range(0, len(X), chunk)])


report = {"prepared": PREPARED_NAME, "classes": classes, "protocol": "train/validation/calibration only; test never opened",
          "lab_train_rows": int(len(ytr)), "feature_counts": {k: len(v) for k, v in cols.items()}, "configs": {}}
cache = {}  # probabilities of the flow reference and of the current best qualified config only (memory)


def run(name, parts):
    c = sorted({j for p in parts for j in cols[p]})
    t0 = time.time()
    booster = lgb.train(PARAMS, lgb.Dataset(Xtr[:, c], ytr, weight=wtr, free_raw_data=True), num_boost_round=200)
    res = {"parts": parts, "features": len(c), "fit_seconds": round(time.time() - t0)}
    probs = {}
    for split, X, y in (("val", Xva, yva), ("calibration", Xca, yca)):
        probs[split] = predict(booster, X, c)
        res[split] = scores(y, probs[split])
    rule = max(("bayes", "balanced"), key=lambda r: res["val"][r]["macro_mcc"])
    res["rule"] = rule
    res["val_macro_mcc"], res["cal_macro_mcc"] = res["val"][rule]["macro_mcc"], res["calibration"][rule]["macro_mcc"]
    gain = booster.feature_importance("gain")
    res["top_features"] = [(ALLNAMES[c[j]], round(float(gain[j]), 1)) for j in np.argsort(-gain)[:25]]
    report["configs"][name] = res
    probs = {k: v.astype(np.float32) for k, v in probs.items()}
    if name == "flow":
        cache["flow"] = (probs, rule)
    else:
        g = min(gains(name))
        if g >= 0.002 and g > cache.get("best", (None, None, -1))[2]:
            cache["best"] = (probs, rule, g, name)
    json.dump(report, open(os.path.join(OUT, "temporal.json"), "w"), indent=1)
    log("config", name, len(c), "features | val", round(res["val_macro_mcc"], 4), "cal", round(res["cal_macro_mcc"], 4),
        rule, res["fit_seconds"], "s")
    return res


def gains(name):
    r = report["configs"][name]
    return r["val_macro_mcc"] - base["val_macro_mcc"], r["cal_macro_mcc"] - base["cal_macro_mcc"]


base = None
base = run("flow", ["flow"])
run("flow+REL", ["flow", "REL"])
for g in GROUPS:
    run(f"flow+{g}", ["flow", g])
run("flow+T_all", ["flow", *GROUPS])
run("flow+REL+T_all", ["flow", "REL", *GROUPS])



passing = [g for g in GROUPS if min(gains(f"flow+{g}")) >= 0.002]
report["passing_groups"] = passing
if passing and len(passing) < len(GROUPS):
    run("flow+T_pass", ["flow", *passing])
    run("flow+REL+T_pass", ["flow", "REL", *passing])
# Backward elimination from the best full set: drop a group if dropping it does not lower val or cal.
qualified = {n: min(gains(n)) for n in report["configs"] if n != "flow" and min(gains(n)) >= 0.002}
best = max(qualified, key=qualified.get) if qualified else "flow"
if best != "flow":
    parts = list(report["configs"][best]["parts"])
    for g in [p for p in parts if p != "flow"]:
        trial = [p for p in parts if p != g]
        name = "+".join(trial) + f" (-{g})"
        r = run(name, trial)
        b = report["configs"][best]
        if r["val_macro_mcc"] >= b["val_macro_mcc"] and r["cal_macro_mcc"] >= b["cal_macro_mcc"]:
            parts, best = trial, name
            log("dropped", g)
    qualified = {n: min(gains(n)) for n in report["configs"] if n != "flow" and min(gains(n)) >= 0.002}
    best = max(qualified, key=qualified.get) if qualified else "flow"
report["recommended"] = {"config": best, "parts": report["configs"][best]["parts"], "gain_val_cal": gains(best),
                         "rule": "max over configs of min(gain_val, gain_cal), both >= 0.002 vs flow"}
# Label-shift adaptation check (Saerens et al. 2002 EM on the unlabeled split, train prior as reference):
# adopt later only if it improves BOTH validation and calibration.
probs, rule = (cache["best"][:2] if best != "flow" else cache["flow"])
assert best == "flow" or cache["best"][3] == best


def em(P, iters=100):
    q = prior.copy()
    for _ in range(iters):
        w = P * (q / np.maximum(prior, 1e-12))
        w /= w.sum(1, keepdims=True)
        q = w.mean(0)
    return q


report["label_shift_em"] = {}
for split, y in (("val", yva), ("calibration", yca)):
    P = probs[split]
    q = em(P)
    Pa = P * (q / np.maximum(prior, 1e-12))
    Pa /= Pa.sum(1, keepdims=True)
    report["label_shift_em"][split] = {"before": scores(y, P)[rule]["macro_mcc"], "after": scores(y, Pa)[rule]["macro_mcc"]}
json.dump(report, open(os.path.join(OUT, "temporal.json"), "w"), indent=1)
b = report["configs"][best]
print(json.dumps({n: {"val": round(r["val_macro_mcc"], 4), "cal": round(r["cal_macro_mcc"], 4), "rule": r["rule"], "features": r["features"]}
                  for n, r in report["configs"].items()}, indent=1))
print("recommended", best, report["recommended"]["gain_val_cal"])
print("val per-class MCC", b["val"][b["rule"]]["per_class_mcc"])
print("cal per-class MCC", b["calibration"][b["rule"]]["per_class_mcc"])
print("top features", b["top_features"][:15])
print("label shift", report["label_shift_em"])
log("temporal lab done")

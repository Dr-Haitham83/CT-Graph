
# LAB 3: TRANSDUCTIVE SELF-TRAINING under campaign shift (train / validation / calibration only; test never opened).
# The evaluation split plays the role of the unlabelled target: the model trained on the training sample pseudo-labels
# the target's flows, the most confident ones join training, the model is refitted (2 rounds) and the target is scored.
# Target labels are used ONLY for scoring. Run with target = validation and, separately, target = calibration; a variant
# counts only if it improves BOTH (>= +0.002 macro-MCC vs the non-adapted model, decision rule fixed from the
# non-adapted validation run). Variants:
#   ST-thr  rows with max probability >= 0.95, at most 20,000 per predicted class
#   ST-bal  the 20 % most confident rows of every predicted class, at most 20,000 per class (class-balanced)
# Pseudo-labelled rows carry, in total, half the total training weight (split across classes by selected counts).
OUT = "/data/output/lab3"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261006)
LAB_TRAIN = 600_000
ROUNDS = 2
CAP = 20_000
started = time.time()
self_check()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def build(split, sample=None):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    flow = np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW])
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    del f
    log("built", split, len(rows), round(time.time() - started), "s")
    return {"y": y[rows], "flow": flow, "flow+REL": np.concatenate([flow, rel], 1)}


def train_sample(y):
    n = np.bincount(y, minlength=K)
    train_sample.counts = n
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(y), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


STATE_CODES = []
TR = build("train", train_sample)
wtr = train_sample.weight
prior = train_sample.counts / train_sample.counts.sum()
TG = {"val": build("val"), "calibration": build("calibration")}
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}
VIEWS = ["flow", "flow+REL"]


def macro(y, P, rule):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}


def predict(b, X, chunk=400_000):
    return np.concatenate([b.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def select(P, mode):
    pred, conf = P.argmax(1), P.max(1)
    chosen = []
    for k in range(K):
        idx = np.flatnonzero(pred == k)
        if not len(idx):
            continue
        if mode == "ST-thr":
            idx = idx[conf[idx] >= 0.95]
            idx = idx[np.argsort(-conf[idx])[:CAP]]
        else:
            idx = idx[np.argsort(-conf[idx])[:min(CAP, max(1, int(0.2 * len(idx))))]]
        chosen.append(idx)
    rows = np.concatenate(chosen) if chosen else np.zeros(0, np.int64)
    return rows, pred[rows]


report = {"prepared": PREPARED_NAME, "protocol": "transductive self-training; targets = validation / calibration (labels only for scoring); test never opened",
          "rounds": ROUNDS, "cap_per_class": CAP, "results": {}}
for view in VIEWS:
    Xtr = TR[view]
    base = lgb.train(PARAMS, lgb.Dataset(Xtr, TR["y"], weight=wtr), num_boost_round=200)
    P0 = {t: predict(base, TG[t][view]) for t in TG}
    rule = max(("bayes", "balanced"), key=lambda r: macro(TG["val"]["y"], P0["val"], r)[0])
    for target in TG:
        y_t = TG[target]["y"]
        m0, pcl0 = macro(y_t, P0[target], rule)
        res = {"rule": rule, "base": m0, "base_per_class": pcl0}
        for mode in ("ST-thr", "ST-bal"):
            P = P0[target]
            trace = []
            for r in range(ROUNDS):
                rows, pseudo = select(P, mode)
                counts = np.bincount(pseudo, minlength=K).astype(float)
                w_ps = (0.5 * wtr.sum() / max(len(rows), 1)) * np.ones(len(rows))
                acc = float((pseudo == y_t[rows]).mean()) if len(rows) else None
                X = np.concatenate([Xtr, TG[target][view][rows]])
                yy = np.concatenate([TR["y"], pseudo])
                ww = np.concatenate([wtr, w_ps])
                b = lgb.train(PARAMS, lgb.Dataset(X, yy, weight=ww), num_boost_round=200)
                P = predict(b, TG[target][view])
                m, pcl = macro(y_t, P, rule)
                trace.append({"round": r + 1, "pseudo_rows": int(len(rows)), "pseudo_label_accuracy": acc,
                              "pseudo_by_class": {classes[k]: int(counts[k]) for k in range(K) if counts[k]}, "macro_mcc": m})
                log(view, target, mode, "round", r + 1, "pseudo", len(rows), "acc", None if acc is None else round(acc, 4), "macro", round(m, 4))
            res[mode] = {"final": trace[-1]["macro_mcc"], "gain": trace[-1]["macro_mcc"] - m0, "trace": trace, "final_per_class": pcl}
        report["results"].setdefault(view, {})[target] = res
        json.dump(report, open(os.path.join(OUT, "lab3.json"), "w"), indent=1, default=float)
for view, d in report["results"].items():
    for mode in ("ST-thr", "ST-bal"):
        gv, gc = d["val"][mode]["gain"], d["calibration"][mode]["gain"]
        d[f"{mode}_passes"] = bool(min(gv, gc) >= 0.002)
json.dump(report, open(os.path.join(OUT, "lab3.json"), "w"), indent=1, default=float)
for view, d in report["results"].items():
    print(f"== {view}: base val {d['val']['base']:.4f} cal {d['calibration']['base']:.4f} (rule {d['val']['rule']})")
    for mode in ("ST-thr", "ST-bal"):
        print(f"   {mode}: val {d['val'][mode]['final']:.4f} ({d['val'][mode]['gain']:+.4f}) cal {d['calibration'][mode]['final']:.4f} "
              f"({d['calibration'][mode]['gain']:+.4f}) {'PASS' if d[f'{mode}_passes'] else '-'} | pseudo acc val "
              f"{d['val'][mode]['trace'][-1]['pseudo_label_accuracy']} cal {d['calibration'][mode]['trace'][-1]['pseudo_label_accuracy']}")
    print("   cal per-class base :", d["calibration"]["base_per_class"])
    print("   cal per-class ST-bal:", d["calibration"]["ST-bal"]["final_per_class"])
log("lab3 done")

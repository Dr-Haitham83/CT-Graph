
# LAB 2: remaining untested, principled levers (train / validation / calibration only; the test split is never opened).
#  CL   confident-learning label cleaning of the training sample (Northcutt, Jiang & Chuang, JAIR 2021): out-of-fold
#       probabilities (two folds of alternating availability-order blocks per segment), per-class thresholds
#       t_j = mean self-confidence, rows in the off-diagonal of the confident joint are dropped before refitting.
#  BAL  class-balanced training weights (every class carries equal total weight; Cui et al., CVPR 2019) instead of
#       weights that restore the natural training prior.
#  DRO  one exponentiated-gradient step of group DRO over training segments (Sagawa et al., ICLR 2020): segments with a
#       high out-of-fold loss are up-weighted (max / min weight ratio 10) and the model is refitted.
#  ST   short-horizon temporal features only (windows <= 60 s, intensities tau <= 10 s; no novelty memory, no running
#       statistics), computed on the segmented AND on the merged monitor timeline (boundary robustness built in).
#  HET  heterogeneous ensemble: LightGBM + ExtraTrees (Geurts, Ernst & Wehenkel, Machine Learning 2006), unweighted mean.
# Every config is scored against flow-only and flow + relational with the double held-out rule (>= +0.002 on BOTH
# validation and calibration; maximin). Nothing is adopted here.
import re
from sklearn.ensemble import ExtraTreesClassifier

OUT = "/data/output/lab2"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261005)
LAB_TRAIN = 600_000
started = time.time()
self_check()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def short_horizon(name):
    if name.startswith("z_") or "intensity_ratio" in name or "new" in name:
        return False
    m = re.search(r"burst_(\d+)_(\d+)$", name)
    if m:
        return int(m.group(2)) <= 60
    m = re.search(r"intensity_(\d+)$", name)
    if m:
        return int(m.group(1)) <= 10
    m = re.search(r"_(\d+)s$", name)
    if m:
        return int(m.group(1)) <= 60
    return name.endswith("since_last")


def fold_of(f):
    st = f["stream"].to_numpy().astype(np.int64)
    order = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), st))
    pos = np.empty(len(f), np.int64)
    pos[order] = np.arange(len(f)) - np.searchsorted(st[order], st[order])
    block = np.clip(np.minimum(2000, np.bincount(st)[st] // 10), 1, None)
    return ((pos // block) % 2).astype(np.int8)


def context(f, rows):
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    T, names_, _ = temporal(f, rows, STATE_CODES, groups=["T_count", "T_flags", "T_volume", "T_timing"])
    keep = [j for j, n in enumerate(names_) if short_horizon(n)]
    return rel, T[:, keep], [names_[j] for j in keep]


def build(split, sample=None):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    d = {"y": y[rows], "flow": np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]),
         "fold": fold_of(f)[rows], "seg": f["stream"].to_numpy()[rows]}
    d["rel"], d["st"], d["st_names"] = context(f, rows)
    hi0, st0 = f["hi"].to_numpy().copy(), f["stream"].to_numpy().copy()
    f["hi"], f["stream"], how = merged_clock(f)
    d["rel_m"], d["st_m"], _ = context(f, rows)
    f["hi"], f["stream"] = hi0, st0
    d["rel_names"] = list(relational_names)
    del f
    log("built", split, len(rows), "rows |", how, "| short-horizon temporal", d["st"].shape[1], round(time.time() - started), "s")
    return d


def train_sample(y):
    n = np.bincount(y, minlength=K)
    train_sample.counts = n
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(y), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


STATE_CODES = top_state_codes(load("train")) if STATE else []
TR = build("train", train_sample)
w_prior = train_sample.weight
prior = train_sample.counts / train_sample.counts.sum()
VA, CA = build("val"), build("calibration")
ytr = TR["y"]
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}


def X_of(d, view):
    parts = {"flow": [d["flow"]], "flow+REL": [d["flow"], d["rel"]], "flow+ST": [d["flow"], d["st"]],
             "flow+REL+ST": [d["flow"], d["rel"], d["st"]], "flow+ST[merged]": [d["flow"], d["st_m"]],
             "flow+REL+ST[merged]": [d["flow"], d["rel_m"], d["st_m"]]}[view]
    return np.concatenate(parts, 1) if len(parts) > 1 else parts[0]


def fit(X, y, w):
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=w), num_boost_round=200)


def macro(y, P, rule):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}


report = {"prepared": PREPARED_NAME, "protocol": "train/validation/calibration only; test never opened",
          "short_horizon_features": TR["st_names"], "configs": {}}
probs = {}


def evaluate(name, Pv, Pc, extra=None):
    rule = max(("bayes", "balanced"), key=lambda r: macro(VA["y"], Pv, r)[0])
    v, vpc = macro(VA["y"], Pv, rule)
    c, cpc = macro(CA["y"], Pc, rule)
    report["configs"][name] = {"rule": rule, "val": v, "calibration": c, "val_per_class": vpc, "cal_per_class": cpc, **(extra or {})}
    probs[name] = (Pv.astype(np.float32), Pc.astype(np.float32))
    json.dump(report, open(os.path.join(OUT, "lab2.json"), "w"), indent=1, default=float)
    log("config", name, "val", round(v, 4), "cal", round(c, 4), rule, round(time.time() - started), "s")


def run(name, view, weights, rows=None):
    Xtr = X_of(TR, view)
    idx = np.arange(len(ytr)) if rows is None else rows
    b = fit(Xtr[idx], ytr[idx], weights[idx])
    evaluate(name, b.predict(X_of(VA, view)), b.predict(X_of(CA, view)), {"train_rows": int(len(idx))})
    return b


# ---- references
for view in ("flow", "flow+REL"):
    run(view, view, w_prior)

# ---- confident learning (OOF on the training sample, alternating-block folds)
fold = TR["fold"]
cover = np.stack([np.bincount(ytr[fold == k], minlength=K) for k in (0, 1)])
if (cover[:, train_sample.counts > 0] < 20).any():
    fold = (rng.random(len(ytr)) < 0.5).astype(np.int8)
    report["cl_folds"] = "random (block folds missed a class)"
else:
    report["cl_folds"] = "alternating availability-order blocks per segment"
oof_loss, cl_keep = {}, {}
for view in ("flow", "flow+REL"):
    Xtr = X_of(TR, view)
    P = np.zeros((len(ytr), K))
    for k in (0, 1):
        b = fit(Xtr[fold == k], ytr[fold == k], w_prior[fold == k])
        P[fold != k] = b.predict(Xtr[fold != k])
    oof_loss[view] = -np.log(np.maximum(P[np.arange(len(ytr)), ytr], 1e-12))
    present = np.bincount(ytr, minlength=K) > 0
    t = np.array([P[ytr == j, j].mean() if present[j] else 1.0 for j in range(K)])
    above = P >= t[None, :]
    jstar = np.where(above, P, -1.0).argmax(1)
    flagged = above.any(1) & (jstar != ytr)
    keep = np.flatnonzero(~flagged)
    cl_keep[view] = keep
    report.setdefault("confident_learning", {})[view] = {
        "flagged_fraction": float(flagged.mean()),
        "flagged_by_class": {classes[j]: round(float(flagged[ytr == j].mean()), 4) for j in range(K) if present[j]},
        "flagged_to": {classes[j]: {classes[i]: int(((ytr == j) & flagged & (jstar == i)).sum()) for i in range(K)
                                    if ((ytr == j) & flagged & (jstar == i)).sum()} for j in range(K) if present[j]}}
    run(f"{view} | CL-clean", view, w_prior, keep)

# ---- class-balanced training weights
counts_s = np.bincount(ytr, minlength=K).astype(float)
w_bal = (1.0 / np.maximum(counts_s, 1))[ytr] * len(ytr) / max((counts_s > 0).sum(), 1)
for view in ("flow", "flow+REL"):
    run(f"{view} | balanced-weights", view, w_bal)
    run(f"{view} | CL-clean+balanced", view, w_bal, cl_keep[view])

# ---- group DRO, one exponentiated-gradient step over training segments (flow and flow+REL)
for view in ("flow", "flow+REL"):
    seg = TR["seg"]
    codes, inv = np.unique(seg, return_inverse=True)
    L = np.bincount(inv, weights=oof_loss[view] * w_prior) / np.maximum(np.bincount(inv, weights=w_prior), 1e-12)
    eta = np.log(10.0) / max(L.max() - L.min(), 1e-9)
    gw = np.exp(eta * (L - L.min()))
    run(f"{view} | groupDRO-1step", view, w_prior * gw[inv], None)
    report["configs"][f"{view} | groupDRO-1step"]["segments"] = int(len(codes))

# ---- short-horizon temporal features, segmented vs merged
for view in ("flow+ST", "flow+ST[merged]", "flow+REL+ST", "flow+REL+ST[merged]"):
    run(view, view, w_prior)

# ---- heterogeneous ensembles: LightGBM + ExtraTrees
for view in ("flow", "flow+REL"):
    et = ExtraTreesClassifier(n_estimators=150, max_features="sqrt", min_samples_leaf=5, max_depth=30, n_jobs=-1, random_state=7)
    Xtr = np.nan_to_num(X_of(TR, view), nan=-1e9, posinf=1e9, neginf=-1e9)
    et.fit(Xtr, ytr, sample_weight=w_prior)
    def et_proba(d):
        out = np.zeros((len(d["y"]), K))
        X = np.nan_to_num(X_of(d, view), nan=-1e9, posinf=1e9, neginf=-1e9)
        for i in range(0, len(X), 300_000):
            out[i:i + 300_000][:, et.classes_] = et.predict_proba(X[i:i + 300_000])
        return out
    Pv_et, Pc_et = et_proba(VA), et_proba(CA)
    evaluate(f"{view} | ExtraTrees", Pv_et, Pc_et)
    Pv_l, Pc_l = probs[view]
    evaluate(f"{view} | LightGBM+ExtraTrees", (Pv_l + Pv_et) / 2, (Pc_l + Pc_et) / 2)
    del et, Xtr

# ---- summary: gains vs flow and vs flow+REL, double held-out rule
f0, r0 = report["configs"]["flow"], report["configs"]["flow+REL"]
for name, r in report["configs"].items():
    r["gain_vs_flow"] = (r["val"] - f0["val"], r["calibration"] - f0["calibration"])
    r["gain_vs_flowREL"] = (r["val"] - r0["val"], r["calibration"] - r0["calibration"])
    r["passes_vs_flow"] = bool(name != "flow" and min(r["gain_vs_flow"]) >= 0.002)
eligible = {n: min(r["gain_vs_flow"]) for n, r in report["configs"].items() if r["passes_vs_flow"]}
report["best_maximin_vs_flow"] = max(eligible, key=eligible.get) if eligible else "flow"
json.dump(report, open(os.path.join(OUT, "lab2.json"), "w"), indent=1, default=float)
print(f"{'config':40s} {'val':>7s} {'cal':>7s}  gain vs flow        gain vs flow+REL    pass")
for name, r in report["configs"].items():
    print(f"{name:40s} {r['val']:.4f} {r['calibration']:.4f}  ({r['gain_vs_flow'][0]:+.4f},{r['gain_vs_flow'][1]:+.4f})  "
          f"({r['gain_vs_flowREL'][0]:+.4f},{r['gain_vs_flowREL'][1]:+.4f})  {'PASS' if r['passes_vs_flow'] else '-'}")
print("best (maximin vs flow):", report["best_maximin_vs_flow"])
print("confident learning:", json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'flagged_to'} for k, v in report["confident_learning"].items()}, indent=1))
b = report["configs"][report["best_maximin_vs_flow"]]
print("best cal per-class:", b["cal_per_class"])
print("flow cal per-class:", f0["cal_per_class"])
log("lab2 done")


# FEATURE DIAGNOSTICS + PRUNING SCREEN (train / validation / calibration only; the test split is never opened).
# Per feature (flow + relational + temporal): LightGBM gain / splits, label mutual information (exact, on train
# quantile bins), drift (univariate train-vs-holdout AUC on UNLABELLED validation+calibration features, plus an
# adversarial-validation classifier), within-train instability (class-conditional standardised mean shift between the
# early and late half of every training segment), Spearman cross-correlation clusters, and cluster-level permutation
# importance on validation and calibration. Pruning strategies derived from TRAIN (and unlabelled holdout features)
# are scored with the double held-out rule (>= +0.002 on BOTH validation and calibration vs flow-only; maximin choice).
# Permutation pruning is selected on validation, so only its calibration score is an unbiased check (flagged).
# The multi-view context ensemble averages GBDTs trained on different feature views (random-subspace ensemble).
import scipy.stats as ss
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

OUT = "/data/output/screen"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260930)
LAB_TRAIN = 600_000
started = time.time()
self_check()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def segment_position(f):
    st = f["stream"].to_numpy().astype(np.int64)
    order = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), st))
    pos = np.empty(len(f), np.int64)
    pos[order] = np.arange(len(f)) - np.searchsorted(st[order], st[order])
    size = np.bincount(st)[st]
    return (pos >= size // 2).astype(np.int8)  # 0 = early half, 1 = late half of its segment


def build(split, sample=None):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    T, t_names, t_groups = temporal(f, rows, STATE_CODES)
    X = np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), rel, T], 1)
    half = segment_position(f)[rows]
    stream = f["stream"].to_numpy()[rows]
    del f, rel, T
    log("built", split, X.shape, round(time.time() - started), "s")
    return X, y[rows], half, stream, list(relational_names), t_names, t_groups


def train_sample(y):
    n = np.bincount(y, minlength=K)
    train_sample.counts = n
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(y), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


STATE_CODES = top_state_codes(load("train")) if STATE else []
Xtr, ytr, half_tr, _, REL, TN, TG = build("train", train_sample)
wtr = train_sample.weight
prior = train_sample.counts / train_sample.counts.sum()
Xva, yva, *_ = build("val")
Xca, yca, *_ = build("calibration")
NAMES = FLOW + REL + TN
GROUP = ["flow"] * len(FLOW) + ["REL"] * len(REL) + list(TG)
F = len(NAMES)
idx = {g: [j for j in range(F) if GROUP[j] == g] for g in dict.fromkeys(GROUP)}
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}


def macro_scores(y, P):
    out = {}
    for rule, pred in (("bayes", P.argmax(1)), ("balanced", (P / np.maximum(prior, 1e-12)).argmax(1))):
        cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
        t, pp = cm.sum(1), cm.sum(0)
        tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
        present = t > 0
        out[rule] = {"macro_mcc": float(mcc[present].mean()),
                     "per_class_mcc": {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}}
    return out


def predict(booster, X, c, chunk=400_000):
    return np.concatenate([booster.predict(X[i:i + chunk][:, c]) for i in range(0, len(X), chunk)]).astype(np.float32)


report = {"prepared": PREPARED_NAME, "protocol": "train/validation/calibration only; test never opened",
          "features": F, "groups": {g: len(v) for g, v in idx.items()}, "configs": {}}
json_path = os.path.join(OUT, "screen.json")

# ---------------- 1. full model, gain / split importance
c_all = list(range(F))
full = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, weight=wtr), num_boost_round=200)
gain = full.feature_importance("gain")
splits = full.feature_importance("split")
log("full model fitted", round(time.time() - started), "s")

# ---------------- 2. label mutual information on train quantile bins (nan = own bin), normalised by H(Y)
def bins_of(col_train, cols_other, nb=32):
    finite = col_train[np.isfinite(col_train)]
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, nb + 1)[1:-1])) if len(finite) else np.zeros(0)
    def code(x):
        c = np.searchsorted(edges, x, side="right")
        return np.where(np.isfinite(x), c, len(edges) + 1)
    return code(col_train), [code(x) for x in cols_other], len(edges) + 2


def mutual_info(codes, labels, nb, nl, weight=None):
    joint = np.bincount(codes * nl + labels, weights=weight, minlength=nb * nl).reshape(nb, nl)
    joint = joint / joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


py_ = np.bincount(ytr, weights=wtr, minlength=K) / wtr.sum()
HY = float(-(py_[py_ > 0] * np.log(py_[py_ > 0])).sum())
n_drift = min(200_000, len(ytr), len(yva) + len(yca))
tr_d = rng.choice(len(ytr), n_drift, replace=False)
ho_rows = rng.choice(len(yva) + len(yca), n_drift, replace=False)
ho_X = np.concatenate([Xva[ho_rows[ho_rows < len(yva)]], Xca[ho_rows[ho_rows >= len(yva)] - len(yva)]])
label_mi, drift_auc, drift_mi = np.zeros(F), np.zeros(F), np.zeros(F)
for j in range(F):
    ct, (cd_tr, cd_ho), nb = bins_of(Xtr[:, j], [Xtr[tr_d, j], ho_X[:, j]])
    label_mi[j] = mutual_info(ct, ytr, nb, K, wtr) / HY
    split_codes = np.concatenate([cd_tr, cd_ho])
    split_lab = np.r_[np.zeros(n_drift, np.int64), np.ones(n_drift, np.int64)]
    drift_mi[j] = mutual_info(split_codes, split_lab, nb, 2) / np.log(2)
    v = np.nan_to_num(np.concatenate([Xtr[tr_d, j], ho_X[:, j]]).astype(np.float64), nan=-1e30)
    r = ss.rankdata(v)
    auc = (r[n_drift:].sum() - n_drift * (n_drift + 1) / 2) / (n_drift * n_drift)
    drift_auc[j] = abs(auc - 0.5) * 2
log("mutual information + univariate drift done", round(time.time() - started), "s")

# ---------------- 3. adversarial validation (train vs unlabelled validation+calibration features)
Xadv = np.concatenate([Xtr[tr_d], ho_X])
yadv = np.r_[np.zeros(n_drift), np.ones(n_drift)]
perm = rng.permutation(len(yadv))
cut = int(0.7 * len(perm))
adv = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "feature_fraction": 0.5, "verbose": -1,
                 "num_threads": os.cpu_count() or 4, "seed": 7}, lgb.Dataset(Xadv[perm[:cut]], yadv[perm[:cut]]), num_boost_round=150)
padv = adv.predict(Xadv[perm[cut:]])
yt = yadv[perm[cut:]]
rk = ss.rankdata(padv)
report["adversarial_auc"] = float((rk[yt == 1].sum() - (yt == 1).sum() * ((yt == 1).sum() + 1) / 2) / ((yt == 1).sum() * (yt == 0).sum()))
adv_gain = adv.feature_importance("gain")
adv_gain = adv_gain / max(adv_gain.sum(), 1e-12)
del Xadv, ho_X
log("adversarial AUC", round(report["adversarial_auc"], 4))

# ---------------- 4. within-train instability (early vs late half of each segment, per class)
instab = np.zeros(F)
halves = [(np.flatnonzero((ytr == k) & (half_tr == 0)), np.flatnonzero((ytr == k) & (half_tr == 1))) for k in range(K)]
for j in range(F):
    x = Xtr[:, j].astype(np.float64)
    tot, wsum = 0.0, 0.0
    for ia, ib in halves:
        a, b = x[ia], x[ib]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 50 or len(b) < 50:
            continue
        sd = np.sqrt((a.var() + b.var()) / 2) + 1e-6
        tot += abs(a.mean() - b.mean()) / sd
        wsum += 1
    instab[j] = tot / wsum if wsum else np.nan
log("instability done", round(time.time() - started), "s")

# ---------------- 5. Spearman cross-correlation clusters (average linkage, |rho| >= 0.95)
sub = rng.choice(len(ytr), min(100_000, len(ytr)), replace=False)
R = np.column_stack([ss.rankdata(np.nan_to_num(Xtr[sub, j], nan=np.nanmedian(Xtr[sub, j]) if np.isfinite(Xtr[sub, j]).any() else 0.0))
                     for j in range(F)])
with np.errstate(invalid="ignore", divide="ignore"):
    rho = np.nan_to_num(np.corrcoef(R, rowvar=False), nan=0.0)
np.fill_diagonal(rho, 1.0)
dist = np.clip(1 - np.abs(rho), 0, None)
np.fill_diagonal(dist, 0)
cluster = fcluster(linkage(squareform(dist, checks=False), "average"), t=0.05, criterion="distance")
del R
report["correlation_clusters"] = int(cluster.max())
log("clusters", cluster.max(), "for", F, "features")

# ---------------- 6. cluster-level permutation importance of the full model (validation / calibration subsamples)
def subsample(y, cap=80_000):
    return np.sort(rng.choice(len(y), min(cap, len(y)), replace=False))


perm_imp = {}
for split, X, y in (("val", Xva, yva), ("calibration", Xca, yca)):
    rows = subsample(y)
    Xs, ys = X[rows].copy(), y[rows]
    base_s = macro_scores(ys, full.predict(Xs))["bayes"]["macro_mcc"]
    imp = np.zeros(cluster.max() + 1)
    for cl in range(1, cluster.max() + 1):
        cols = np.flatnonzero(cluster == cl)
        saved = Xs[:, cols].copy()
        Xs[:, cols] = saved[rng.permutation(len(ys))]
        imp[cl] = base_s - macro_scores(ys, full.predict(Xs))["bayes"]["macro_mcc"]
        Xs[:, cols] = saved
    perm_imp[split] = imp
    log("permutation importance", split, "base", round(base_s, 4), round(time.time() - started), "s")

# ---------------- per-feature table
table = []
for j in range(F):
    table.append({"feature": NAMES[j], "group": GROUP[j], "gain": float(gain[j]), "splits": int(splits[j]),
                  "label_mi": float(label_mi[j]), "drift_auc": float(drift_auc[j]), "drift_mi": float(drift_mi[j]),
                  "adversarial_gain": float(adv_gain[j]), "instability": float(instab[j]), "cluster": int(cluster[j]),
                  "perm_val": float(perm_imp["val"][cluster[j]]), "perm_cal": float(perm_imp["calibration"][cluster[j]])})
pd.DataFrame(table).to_csv(os.path.join(OUT, "features.csv"), index=False)
T_ = pd.DataFrame(table)

# ---------------- 7. pruning strategies (all derived from train / unlabelled holdout features, except P9)
gain_rank = np.argsort(np.argsort(gain)) / max(F - 1, 1)
drift_cut = np.quantile(drift_auc, 0.85)
instab_cut = np.nanquantile(instab, 0.85)
keep_cluster_rep = np.zeros(F, bool)
for cl in np.unique(cluster):
    members = np.flatnonzero(cluster == cl)
    keep_cluster_rep[members[np.argmax(label_mi[members])]] = True
flow_set = set(idx["flow"])
scale_free = [j for j in range(F) if j in flow_set or any(k in NAMES[j] for k in ("_frac_", "gap_cv", "lgap_cv", "burst", "intensity_ratio"))
              or NAMES[j].startswith("z_")]
low_gain = set(np.flatnonzero((gain_rank < 0.30) & ~np.isin(np.arange(F), list(flow_set))))
drifty = set(np.flatnonzero((drift_auc > drift_cut) | (adv_gain > np.quantile(adv_gain, 0.95)))) - flow_set
unstable = set(np.flatnonzero(np.nan_to_num(instab, nan=0) > instab_cut)) - flow_set
redundant = set(np.flatnonzero(~keep_cluster_rep)) - flow_set
neg_val = set(j for j in range(F) if perm_imp["val"][cluster[j]] < 0) - flow_set
configs = {
    "P0_flow": sorted(flow_set),
    "P1_flow+REL": sorted(flow_set | set(idx["REL"])),
    "P2_all": c_all,
    "P3_all-lowgain": sorted(set(c_all) - low_gain),
    "P4_all-redundant": sorted(set(c_all) - redundant),
    "P5_all-drift": sorted(set(c_all) - drifty),
    "P6_all-unstable": sorted(set(c_all) - unstable),
    "P7_scale_free": sorted(set(scale_free)),
    "P8_all-(redundant|drift|unstable)": sorted(set(c_all) - redundant - drifty - unstable),
    "P9_all-negative_val_permutation (val-selected; cal is the unbiased check)": sorted(set(c_all) - neg_val),
    "P10_flow+T_segz": sorted(flow_set | set(idx.get("T_segz", []))),
    "P11_flow+REL+T_segz": sorted(flow_set | set(idx["REL"]) | set(idx.get("T_segz", []))),
}
report["pruning_sets"] = {k: len(v) for k, v in configs.items()}
report["removed"] = {"low_gain": len(low_gain), "drift": len(drifty), "unstable": len(unstable), "redundant": len(redundant),
                     "negative_val_permutation": len(neg_val)}
probs = {}
for name, c in configs.items():
    t0 = time.time()
    booster = full if name == "P2_all" else lgb.train(PARAMS, lgb.Dataset(Xtr[:, c], ytr, weight=wtr), num_boost_round=200)
    res = {}
    keep_probs = name in ("P0_flow", "P1_flow+REL", "P2_all", "P8_all-(redundant|drift|unstable)", "P10_flow+T_segz")
    if keep_probs:
        probs[name] = {}
    for split, X, y in (("val", Xva, yva), ("calibration", Xca, yca)):
        P = predict(booster, X, c)
        if keep_probs:
            probs[name][split] = P
        res[split] = macro_scores(y, P)
    rule = max(("bayes", "balanced"), key=lambda r: res["val"][r]["macro_mcc"])
    report["configs"][name] = {"features": len(c), "rule": rule, "val": res["val"][rule], "calibration": res["calibration"][rule],
                               "fit_seconds": round(time.time() - t0)}
    json.dump(report, open(json_path, "w"), indent=1)
    log("config", name, len(c), "val", round(res["val"][rule]["macro_mcc"], 4), "cal", round(res["calibration"][rule]["macro_mcc"], 4), rule)

# ---------------- 8. multi-view context ensemble (average of views; no weights fitted)
for name, views in (("E1_views(REL,T_segz,all)", ["P1_flow+REL", "P10_flow+T_segz", "P2_all"]),
                    ("E2_views(REL,T_segz,all,pruned)", ["P1_flow+REL", "P10_flow+T_segz", "P2_all", "P8_all-(redundant|drift|unstable)"]),
                    ("E3_views(flow,REL,T_segz)", ["P0_flow", "P1_flow+REL", "P10_flow+T_segz"])):
    res = {s: macro_scores(y, np.mean([probs[v][s] for v in views], 0)) for s, y in (("val", yva), ("calibration", yca))}
    rule = max(("bayes", "balanced"), key=lambda r: res["val"][r]["macro_mcc"])
    report["configs"][name] = {"views": views, "rule": rule, "val": res["val"][rule], "calibration": res["calibration"][rule]}
    log("config", name, "val", round(res["val"][rule]["macro_mcc"], 4), "cal", round(res["calibration"][rule]["macro_mcc"], 4), rule)
b0 = report["configs"]["P0_flow"]
for name, r in report["configs"].items():
    r["gain_vs_flow"] = (r["val"]["macro_mcc"] - b0["val"]["macro_mcc"], r["calibration"]["macro_mcc"] - b0["calibration"]["macro_mcc"])
eligible = {n: min(r["gain_vs_flow"]) for n, r in report["configs"].items() if not n.startswith(("P0", "P9")) and min(r["gain_vs_flow"]) >= 0.002}
report["recommended"] = max(eligible, key=eligible.get) if eligible else "P0_flow"
json.dump(report, open(json_path, "w"), indent=1, default=float)

# ---------------- printout
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 20)
cols_show = ["feature", "group", "gain", "label_mi", "drift_auc", "adversarial_gain", "instability", "cluster", "perm_val", "perm_cal"]
print("adversarial AUC train vs holdout:", round(report["adversarial_auc"], 4), "| clusters:", report["correlation_clusters"], "for", F)
print("\nLOWEST-RANKED non-flow features (gain):\n", T_[T_.group != "flow"].nsmallest(25, "gain")[cols_show].to_string(index=False))
print("\nzero-gain features:", int((T_.gain == 0).sum()), "| by group:", T_[T_.gain == 0].group.value_counts().to_dict())
print("\nHIGHEST DRIFT (train vs holdout):\n", T_.nlargest(20, "drift_auc")[cols_show].to_string(index=False))
print("\nTOP ADVERSARIAL-VALIDATION features:\n", T_.nlargest(15, "adversarial_gain")[cols_show].to_string(index=False))
print("\nMOST UNSTABLE within train:\n", T_.nlargest(15, "instability")[cols_show].to_string(index=False))
print("\nHIGH-GAIN but DRIFTING (potential shortcuts):\n",
      T_[(T_.gain > T_.gain.quantile(0.8)) & (T_.drift_auc > T_.drift_auc.quantile(0.8))][cols_show].to_string(index=False))
harm = T_[(T_.perm_val < 0) & (T_.perm_cal < 0)]
print("\nclusters HARMFUL on both val and cal (permuting improves macro-MCC):", harm.cluster.nunique(), "features", len(harm))
print(harm.sort_values("perm_cal")[cols_show].head(25).to_string(index=False))
big = T_.groupby("cluster").feature.apply(list)
print("\nlargest redundancy clusters:", [(len(v), v[:6]) for v in sorted(big, key=len, reverse=True)[:12]])
print("\nGROUP summary (mean label MI, mean drift AUC, mean instability, total gain share):")
g = T_.groupby("group").agg(label_mi=("label_mi", "mean"), drift_auc=("drift_auc", "mean"), instability=("instability", "mean"),
                            gain=("gain", "sum"))
g["gain"] = g["gain"] / g["gain"].sum()
print(g.round(4).to_string())
print("\nCONFIGS:")
for name, r in report["configs"].items():
    print(f"  {name:70s} val {r['val']['macro_mcc']:.4f} cal {r['calibration']['macro_mcc']:.4f} gain {tuple(round(x, 4) for x in r['gain_vs_flow'])}")
print("recommended (maximin, train-derived configs only):", report["recommended"])
best = report["configs"][report["recommended"]]
print("val per-class", best["val"]["per_class_mcc"])
print("cal per-class", best["calibration"]["per_class_mcc"])
log("screen done")

# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# DEEP ANALYSIS of one frozen split (train / validation / calibration only; the test split is never opened).
#   A. per-flow Bayes-error lower bound from identical feature vectors with different labels (exact), per class,
#      and the class pairs that share vectors;
#   B. mutual information I(feature; class) (quantile-binned) on train and validation, and its stability;
#   C. train->validation shift: label-prior KL, per-feature PSI and KS, adversarial validation AUC overall and
#      per class (class-conditional covariate/concept drift), drift feature ranking;
#   D. label noise by confident learning (Northcutt et al., JAIR 2021) on stream-grouped out-of-fold probabilities;
#   E. context information gain: cross-entropy reduction (bits/flow) from causal cardinality context, overall,
#      per class and on label-conflicting vectors;
#   F. remedy tournament with fixed LightGBM hyper-parameters (no early stopping on validation): selection on
#      validation macro-MCC, confirmation on the calibration split; greedy combination of the winning remedies.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = "/data/output/deep"
os.makedirs(OUT, exist_ok=True)
SEED = 20260925
MAX_TRAIN = 3_000_000
WINDOWS = (10.0, 60.0, 600.0)
HORIZON = 3600.0
rng = np.random.default_rng(SEED)
import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
classes = prep["classes"]
K = len(classes)
width = len(prep["means"])
FLOW = prep["feature_names"] + prep["schema"]["categorical"]
label_set = pa.array(classes)
report = {"prepared": PREPARED_NAME, "classes": classes, "sizes": manifest["sizes"],
          "protocol": "train/validation/calibration only; test never opened"}


def save():
    json.dump(report, open(os.path.join(OUT, "deep.json"), "w"), indent=1, default=float)


def shard_path(split, shard):
    return shard["path"] if os.path.exists(shard["path"]) else os.path.join(root, "prepared", split, os.path.basename(shard["path"]))


def load(split):
    frames = []
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=["numeric", "categories", "label", "hi", "source_order",
                                                             "s0", "s1", "d0", "d1", "dport"])
        x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
        cats = []
        for j, vocab in enumerate(prep["vocabs"]):
            values = pc.list_element(t["categories"], j)
            index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
            cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
        f = pd.DataFrame(np.concatenate([x, np.column_stack(cats)], 1) if cats else x, columns=FLOW)
        f["y"] = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        for c in ("hi", "source_order", "s0", "s1", "d0", "d1", "dport"):
            f[c] = t[c].to_numpy(zero_copy_only=False)
        f["stream"] = shard["stream"]
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    return f[f["y"] >= 0].reset_index(drop=True)


def pair_code(a, b, c):
    h = (a.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (b.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
    return (h ^ (c.astype(np.uint64) * np.uint64(0x165667B19E3779F9))).view(np.int64)


def cardinality(f):
    """Causal per-stream fan-out/fan-in novelty context (as in the hypothesis kernel), float32."""
    order0 = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), pd.factorize(f["stream"])[0]))
    stream = pd.factorize(f["stream"])[0].astype(np.int64)[order0]
    src = pair_code(f["s0"].to_numpy()[order0], f["s1"].to_numpy()[order0], stream)
    dst = pair_code(f["d0"].to_numpy()[order0], f["d1"].to_numpy()[order0], stream)
    t = f["hi"].to_numpy(np.float64)[order0]
    rel = t - pd.Series(t).groupby(stream).transform("min").to_numpy()
    port = f["dport"].to_numpy().astype(np.int64)[order0]
    flags = {}
    for name, a, b in (("sp", src, port), ("sd", src, dst), ("dp", dst, port), ("ds", dst, src)):
        prev = pd.Series(t).groupby(pair_code(a, b, np.zeros_like(a))).shift(1).to_numpy()
        flags[name] = (np.isnan(prev) | (t - prev > HORIZON)).astype(np.float64)
    out = {}
    for role, ep, names in (("src", src, ("sp", "sd")), ("dst", dst, ("dp", "ds"))):
        order = np.lexsort((np.arange(len(t)), rel, ep))
        composite = pd.factorize(ep[order])[0].astype(np.float64) * 1e7 + rel[order]
        for w in WINDOWS:
            start = np.searchsorted(composite, composite - w, side="right")
            count = np.arange(len(order)) - start + 1
            cols = {f"{role}_flows_{int(w)}s": count.astype(np.float64)}
            for n in names:
                c = np.cumsum(flags[n][order])
                s = c - np.where(start > 0, c[np.maximum(start - 1, 0)], 0.0)
                cols[f"{role}_new_{n}_{int(w)}s"] = s
                cols[f"{role}_new_{n}_frac_{int(w)}s"] = s / count
            for k, v in cols.items():
                col = np.empty(len(t), np.float32)
                col[order0[order]] = v
                out[k] = col
    return pd.DataFrame(out)


def hashes(x):
    return pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(x, nan=-7.77e30)), index=False).to_numpy()


def cm_summary(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    t, p = cm.sum(1).astype(float), cm.sum(0).astype(float)
    s, c = cm.sum(), np.trace(cm)
    tp = np.diag(cm).astype(float)
    fn, fp = t - tp, p - tp
    tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_k = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    f1_k = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    present = t > 0
    den_all = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return {"macro_mcc": float(mcc_k[present].mean()), "mcc": float((c * s - (p * t).sum()) / den_all) if den_all > 0 else 0.0,
            "accuracy": float(c / s), "per_class_f1": {classes[i]: round(float(f1_k[i]), 4) for i in range(K) if present[i]}}


# ------------------------------------------------------------------------------------------------ load
started = time.time()
train, val, cal = load("train"), load("val"), load("calibration")
log("rows train/val/cal", len(train), len(val), len(cal), round(time.time() - started), "s")
ctx = {name: cardinality(f) for name, f in (("train", train), ("val", val), ("cal", cal))}
CTX = list(ctx["train"].columns)
log("cardinality context", len(CTX), "features", round(time.time() - started), "s")
y_tr, y_va, y_ca = train["y"].to_numpy(), val["y"].to_numpy(), cal["y"].to_numpy()
n_tr = np.bincount(y_tr, minlength=K)
prior = n_tr / n_tr.sum()
report["class_counts"] = {"train": dict(zip(classes, n_tr.tolist())), "val": dict(zip(classes, np.bincount(y_va, minlength=K).tolist())),
                          "calibration": dict(zip(classes, np.bincount(y_ca, minlength=K).tolist()))}

# ------------------------------------------------------------------------------------------------ A. Bayes bound
usable = ~np.all(np.isnan(train[FLOW].to_numpy(np.float32)), 0)
cols = [c for c, u in zip(FLOW, usable) if u]
h_tr = hashes(train[cols].to_numpy(np.float32))
table = pd.DataFrame({"h": h_tr, "y": y_tr}).value_counts().unstack(fill_value=0).reindex(columns=range(K), fill_value=0)
counts = table.to_numpy(np.float64)
total = counts.sum(1)
best = counts.max(1)
labels_per_vector = (counts > 0).sum(1)
bayes_err = float((total - best).sum() / total.sum())
row_conflict = labels_per_vector > 1
per_class_conflict = {classes[c]: float(counts[row_conflict, c].sum() / max(counts[:, c].sum(), 1)) for c in range(K)}
per_class_floor = {}  # errors forced on class c by the Bayes decision of shared vectors
winner = counts.argmax(1)
for c in range(K):
    lost = counts[:, c] * (winner != c)
    per_class_floor[classes[c]] = float(lost.sum() / max(counts[:, c].sum(), 1))
pairs = []
for a in range(K):
    for b in range(a + 1, K):
        w = np.minimum(counts[:, a], counts[:, b]).sum()
        if w > 0:
            pairs.append((float(w), classes[a], classes[b]))
pairs.sort(reverse=True)
h_va = hashes(val[cols].to_numpy(np.float32))
index = table.index.to_numpy()
order = np.argsort(index)
pos = np.minimum(np.searchsorted(index[order], h_va), len(index) - 1)
seen = index[order][pos] == h_va
lookup_pred = winner[order][pos]
report["A_bayes_bound"] = {
    "distinct_train_vectors": int(len(index)), "train_rows": int(total.sum()),
    "rows_on_label_conflicting_vectors": float(counts[row_conflict].sum() / total.sum()),
    "per_flow_bayes_error_lower_bound_train": bayes_err,
    "max_per_flow_accuracy_train": 1 - bayes_err,
    "per_class_fraction_on_conflicting_vectors": per_class_conflict,
    "per_class_minimum_recall_loss": per_class_floor,
    "top_vector_sharing_pairs": pairs[:12],
    "validation_rows_with_train_vector": float(seen.mean()),
    "validation_lookup_accuracy_on_seen_rows": float((lookup_pred[seen] == y_va[seen]).mean()) if seen.any() else None,
    "validation_lookup_macro_mcc_on_seen_rows": cm_summary(y_va[seen], lookup_pred[seen])["macro_mcc"] if seen.any() else None}
val_conflicting = np.zeros(len(y_va), bool)
val_conflicting[seen] = row_conflict[order][pos][seen]
del table, counts
log("A done", {k: v for k, v in report["A_bayes_bound"].items() if not isinstance(v, (dict, list))})
save()

# ------------------------------------------------------------------------------------------------ stratified training sample
rate = np.minimum(1.0, np.maximum(MAX_TRAIN / len(y_tr), (MAX_TRAIN / (2 * K)) / np.maximum(n_tr, 1)))
keep = rng.random(len(y_tr)) < rate[y_tr]
sample_w = (1.0 / rate[y_tr[keep]]).astype(np.float64)  # restores the full-train class distribution
Xtr = pd.concat([train.loc[keep, FLOW].reset_index(drop=True), ctx["train"].loc[keep].reset_index(drop=True)], axis=1)
ytr = y_tr[keep]
stream_tr = train.loc[keep, "stream"].to_numpy()
Xva = pd.concat([val[FLOW], ctx["val"]], axis=1)
Xca = pd.concat([cal[FLOW], ctx["cal"]], axis=1)
report["training_sample"] = {"rows": int(keep.sum()), "rate_per_class": dict(zip(classes, np.round(rate, 5).tolist()))}
del train, ctx
log("training sample", int(keep.sum()))

# ------------------------------------------------------------------------------------------------ B. mutual information
def binned(col, edges):
    v = np.nan_to_num(col, nan=-1e30)
    return np.searchsorted(edges, v, side="right")


def mutual_information(bins, y, weight=None):
    w = np.ones(len(y)) if weight is None else weight
    joint = np.zeros((bins.max() + 1, K))
    np.add.at(joint, (bins, y), w)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


H_y = float(-(prior[prior > 0] * np.log2(prior[prior > 0])).sum())
mi = {}
for c in FLOW + CTX:
    col = Xtr[c].to_numpy(np.float64)
    finite = col[np.isfinite(col)]
    if len(finite) == 0 or np.nanmin(finite) == np.nanmax(finite):
        continue
    edges = np.unique(np.quantile(finite, np.linspace(0, 1, 33)[1:-1]))
    mi[c] = {"train_bits": mutual_information(binned(col, edges), ytr, sample_w),
             "val_bits": mutual_information(binned(Xva[c].to_numpy(np.float64), edges), y_va)}
ranked = sorted(mi.items(), key=lambda kv: -kv[1]["train_bits"])
tr_rank = np.argsort(np.argsort([-v["train_bits"] for _, v in ranked]))
va_rank = np.argsort(np.argsort([-v["val_bits"] for _, v in ranked]))
report["B_mutual_information"] = {"label_entropy_bits": H_y, "top": [(k, round(v["train_bits"], 4), round(v["val_bits"], 4)) for k, v in ranked[:25]],
                                  "spearman_train_val_rank": float(np.corrcoef(tr_rank, va_rank)[0, 1]),
                                  "context_features_in_top25": [k for k, _ in ranked[:25] if k in CTX]}
log("B done; top MI", report["B_mutual_information"]["top"][:6])
save()

# ------------------------------------------------------------------------------------------------ C. shift
p_va = np.bincount(y_va, minlength=K) / len(y_va)
kl = float(np.sum(np.where(p_va > 0, p_va * np.log2(np.maximum(p_va, 1e-12) / np.maximum(prior, 1e-12)), 0)))
psi, ks = {}, {}
for c in FLOW + CTX:
    a, b = Xtr[c].to_numpy(np.float64), Xva[c].to_numpy(np.float64)
    fa = a[np.isfinite(a)]
    if len(fa) == 0 or fa.min() == fa.max():
        continue
    edges = np.unique(np.quantile(fa, np.linspace(0, 1, 11)[1:-1]))
    pa_ = np.bincount(binned(a, edges), weights=sample_w, minlength=len(edges) + 1)
    pb_ = np.bincount(binned(b, edges), minlength=len(edges) + 1).astype(float)
    pa_, pb_ = pa_ / pa_.sum() + 1e-6, pb_ / pb_.sum() + 1e-6
    psi[c] = float(np.sum((pb_ - pa_) * np.log(pb_ / pa_)))
    qs = np.unique(np.quantile(fa, np.linspace(0, 1, 101)))
    ca = np.cumsum(np.bincount(binned(a, qs), weights=sample_w, minlength=len(qs) + 1)); ca /= ca[-1]
    cb = np.cumsum(np.bincount(binned(b, qs), minlength=len(qs) + 1).astype(float)); cb /= cb[-1]
    ks[c] = float(np.max(np.abs(ca - cb)))


def adversarial(columns, rows_tr, rows_va, folds=3, rounds=150):
    """Cross-fitted train-vs-validation classifier: AUC, gain ranking and out-of-fold P(val | x) for train rows."""
    Xa = pd.concat([Xtr.loc[rows_tr, columns], Xva.loc[rows_va, columns]], ignore_index=True)
    lab = np.r_[np.zeros(len(rows_tr)), np.ones(len(rows_va))]
    fold = rng.integers(0, folds, len(lab))
    oof = np.zeros(len(lab))
    gain = np.zeros(len(columns))
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": os.cpu_count() or 4, "seed": SEED}
    for f in range(folds):
        m = lgb.train(params, lgb.Dataset(Xa[fold != f], lab[fold != f]), num_boost_round=rounds)
        oof[fold == f] = m.predict(Xa[fold == f])
        gain += m.feature_importance("gain")
    return float(roc_auc_score(lab, oof)), dict(zip(columns, (gain / folds).tolist())), oof[:len(rows_tr)]


n_adv = min(400_000, len(ytr), len(y_va))
rows_tr = rng.choice(len(ytr), n_adv, replace=False, p=sample_w / sample_w.sum())
rows_va = rng.choice(len(y_va), n_adv, replace=False)
auc_all, gain_all, _ = adversarial(FLOW, rows_tr, rows_va)
drift_rank = [k for k, _ in sorted(gain_all.items(), key=lambda kv: -kv[1])]
per_class_auc = {}
for c in range(K):
    a_rows = np.flatnonzero(ytr == c)
    b_rows = np.flatnonzero(y_va == c)
    if len(a_rows) >= 2000 and len(b_rows) >= 2000:
        m = min(len(a_rows), len(b_rows), 150_000)
        per_class_auc[classes[c]] = adversarial(FLOW, rng.choice(a_rows, m, replace=False), rng.choice(b_rows, m, replace=False), rounds=100)[0]
report["C_shift"] = {"label_prior_kl_bits_val_vs_train": kl,
                     "psi_top": sorted(psi.items(), key=lambda kv: -kv[1])[:15],
                     "ks_top": sorted(ks.items(), key=lambda kv: -kv[1])[:15],
                     "features_psi_over_0.25": int(sum(v > 0.25 for v in psi.values())),
                     "adversarial_auc_all_flow_features": auc_all,
                     "adversarial_top_drift_features": drift_rank[:15],
                     "adversarial_auc_per_class": per_class_auc,
                     "context_psi": {k: psi[k] for k in CTX if k in psi}}
log("C done: adversarial AUC", round(auc_all, 4), "per class", {k: round(v, 3) for k, v in per_class_auc.items()})
save()

# ------------------------------------------------------------------------------------------------ D. confident learning
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": SEED, "max_bin": 255}
streams = np.unique(stream_tr)
group_fold = {s: i % 3 for i, s in enumerate(rng.permutation(streams))}
fold = np.asarray([group_fold[s] for s in stream_tr])
if len(streams) < 3:  # too few streams: fall back to random folds
    fold = rng.integers(0, 3, len(ytr))
oof = np.zeros((len(ytr), K))
for f in range(3):
    m = lgb.train(PARAMS, lgb.Dataset(Xtr.loc[fold != f, FLOW], ytr[fold != f], weight=sample_w[fold != f]), num_boost_round=150)
    oof[fold == f] = m.predict(Xtr.loc[fold == f, FLOW])
thresholds = np.asarray([oof[ytr == j, j].mean() if (ytr == j).any() else 1.0 for j in range(K)])
above = oof >= thresholds[None, :]
masked = np.where(above, oof, -1)
confident = np.where(above.any(1), masked.argmax(1), -1)
cj = np.zeros((K, K))
ok = confident >= 0
np.add.at(cj, (ytr[ok], confident[ok]), sample_w[ok])
noise_rate = {classes[i]: float(1 - cj[i, i] / cj[i].sum()) if cj[i].sum() else None for i in range(K)}
issues = ok & (confident != ytr)
noisy_pairs = sorted([(float(cj[i, j]), classes[i], classes[j]) for i in range(K) for j in range(K) if i != j and cj[i, j] > 0], reverse=True)
report["D_confident_learning"] = {"folds": "stream-grouped 3-fold" if len(streams) >= 3 else "random 3-fold",
                                  "estimated_noise_rate_per_class": noise_rate,
                                  "suspected_label_issues_weighted_fraction": float(sample_w[issues].sum() / sample_w.sum()),
                                  "top_confident_off_diagonal": noisy_pairs[:10],
                                  "oof_macro_mcc": cm_summary(ytr, oof.argmax(1))["macro_mcc"]}
log("D done", {k: v for k, v in report["D_confident_learning"].items() if k != "top_confident_off_diagonal"})
save()

# ------------------------------------------------------------------------------------------------ E/F. remedy tournament
def fit_eval(name, columns, weight, rows=None, rounds=300):
    started = time.time()
    rows = np.arange(len(ytr)) if rows is None else rows
    m = lgb.train(PARAMS, lgb.Dataset(Xtr.loc[rows, columns], ytr[rows], weight=weight[rows]), num_boost_round=rounds)
    out = {}
    for split, X, y in (("val", Xva, y_va), ("cal", Xca, y_ca)):
        p = m.predict(X[columns])
        ce = -np.log2(np.maximum(p[np.arange(len(y)), y], 1e-12))
        res = {"cross_entropy_bits": float(ce.mean()),
               "per_class_ce_bits": {classes[c]: round(float(ce[y == c].mean()), 4) for c in range(K) if (y == c).any()}}
        for rule, f in (("bayes", p.argmax(1)), ("balanced", (p / np.maximum(prior, 1e-12)).argmax(1))):
            res[rule] = cm_summary(y, f)
        if split == "val":
            res["conflicting_subset_ce_bits"] = float(ce[val_conflicting].mean()) if val_conflicting.any() else None
        out[split] = res
    rule = max(("bayes", "balanced"), key=lambda r: out["val"][r]["macro_mcc"])
    summary = {"rule_on_val": rule, "val_macro_mcc": out["val"][rule]["macro_mcc"], "cal_macro_mcc": out["cal"][rule]["macro_mcc"],
               "val_mcc": out["val"][rule]["mcc"], "cal_mcc": out["cal"][rule]["mcc"], "val_ce_bits": out["val"]["cross_entropy_bits"],
               "seconds": time.time() - started, "detail": out}
    report.setdefault("F_remedies", {})[name] = summary
    log("remedy", name, {k: round(v, 4) for k, v in summary.items() if isinstance(v, float)})
    save()
    return summary


base_w = sample_w.copy()
r0 = fit_eval("R0_flow", FLOW, base_w)
top_drift = [c for c in drift_rank if c in FLOW][:10]
fit_eval("R1_drop_top10_drift", [c for c in FLOW if c not in top_drift], base_w)
# Covariate-shift importance weights (density ratio p_val/p_train) from a cross-fitted classifier on ALL train rows.
adv_rows_tr = np.arange(len(ytr))
adv_rows_va = rng.choice(len(y_va), min(len(y_va), 1_000_000), replace=False)
_, _, p_val_given_x = adversarial(FLOW, adv_rows_tr, adv_rows_va, folds=2, rounds=100)
ratio = np.clip(p_val_given_x / np.maximum(1 - p_val_given_x, 1e-6) * (len(adv_rows_tr) / len(adv_rows_va)), 0.1, 10.0)
iw = base_w * ratio
for c in range(K):  # keep class priors: reweight only within each class (covariate shift inside classes)
    m_ = ytr == c
    if m_.any():
        iw[m_] *= base_w[m_].sum() / iw[m_].sum()
fit_eval("R2_importance_weights", FLOW, iw)
cb = base_w * (1.0 / np.maximum(prior[ytr], 1e-12)) ** 0.5
cb *= base_w.sum() / cb.sum()
fit_eval("R3_class_balanced_sqrt", FLOW, cb)
fit_eval("R4_confident_learning_clean", FLOW, base_w, rows=np.flatnonzero(~issues))
seg = base_w.copy()
for s in streams:
    m_ = stream_tr == s
    seg[m_] *= (base_w.sum() / len(streams)) / base_w[m_].sum()
fit_eval("R5_segment_balanced", FLOW, seg)
fit_eval("R6_flow_plus_context", FLOW + CTX, base_w)
# Greedy combination of the remedies that beat R0 on validation (confirmation reported on calibration).
winners = [k for k, v in report["F_remedies"].items() if k != "R0_flow" and v["val_macro_mcc"] > r0["val_macro_mcc"] + 0.002]
report["F_winners_on_val"] = winners
if len(winners) >= 2:
    columns = list(FLOW)
    w = base_w.copy()
    rows = None
    if "R1_drop_top10_drift" in winners:
        columns = [c for c in columns if c not in top_drift]
    if "R6_flow_plus_context" in winners:
        columns = columns + CTX
    if "R2_importance_weights" in winners:
        w = w * (iw / base_w)
    if "R3_class_balanced_sqrt" in winners:
        w = w * (cb / base_w)
    if "R5_segment_balanced" in winners:
        w = w * (seg / base_w)
    if "R4_confident_learning_clean" in winners:
        rows = np.flatnonzero(~issues)
    fit_eval("R7_greedy_combination", columns, w, rows=rows)
ctx_gain = {c: round(report["F_remedies"]["R0_flow"]["detail"]["val"]["per_class_ce_bits"].get(c, 0)
                     - report["F_remedies"]["R6_flow_plus_context"]["detail"]["val"]["per_class_ce_bits"].get(c, 0), 4)
            for c in classes}
report["E_context_information_gain_bits_per_flow"] = {
    "overall": r0["val_ce_bits"] - report["F_remedies"]["R6_flow_plus_context"]["val_ce_bits"],
    "per_class": ctx_gain,
    "conflicting_vectors": (report["F_remedies"]["R0_flow"]["detail"]["val"]["conflicting_subset_ce_bits"] or 0)
                           - (report["F_remedies"]["R6_flow_plus_context"]["detail"]["val"]["conflicting_subset_ce_bits"] or 0)}
save()
print(json.dumps({k: report[k] for k in ("A_bayes_bound", "B_mutual_information", "C_shift", "D_confident_learning",
                                         "E_context_information_gain_bits_per_flow", "F_winners_on_val")}, indent=1, default=float)[:20000])
for name, v in report["F_remedies"].items():
    print(f"{name:32s} val mMCC {v['val_macro_mcc']:.4f} cal mMCC {v['cal_macro_mcc']:.4f} val MCC {v['val_mcc']:.4f} "
          f"cal MCC {v['cal_mcc']:.4f} CE {v['val_ce_bits']:.4f} rule {v['rule_on_val']}")
log("deep analysis done")

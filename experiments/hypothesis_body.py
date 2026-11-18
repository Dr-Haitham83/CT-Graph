# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# VALIDATION-ONLY hypothesis test (no test file is opened): do causal CARDINALITY features (fan-out / fan-in
# novelty of peers and service ports per endpoint, in short windows) resolve the classes that per-flow features
# cannot separate (label-conflicting feature vectors, e.g. ToN scanning)?
#
# For every flow (availability time hi, per segment-scoped stream, ties by source_order):
#   novelty flags  n_sp: (src, dport) not seen in the last H s     n_sd: (src, dst) not seen in the last H s
#                  n_dp: (dst, dport) not seen in the last H s     n_ds: (dst, src) not seen in the last H s
#   window sums over the endpoint's own past flows (t - W, t], W in WINDOWS: flows, sum of the novelty flags.
# LightGBM (fixed params, early stopping on validation) on train: flow features vs flow + cardinality context.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = "/data/output/hypothesis"
os.makedirs(OUT, exist_ok=True)
WINDOWS = (10.0, 60.0, 600.0)
HORIZON = 3600.0
MAX_TRAIN = 6_000_000

matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
classes = prep["classes"]
K = len(classes)
width = len(prep["means"])
label_set = pa.array(classes)


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
        f = pd.DataFrame(np.concatenate([x, np.column_stack(cats)], 1) if cats else x,
                         columns=prep["feature_names"] + prep["schema"]["categorical"])
        f["y"] = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        for c in ("hi", "source_order", "s0", "s1", "d0", "d1", "dport"):
            f[c] = t[c].to_numpy(zero_copy_only=False)
        f["stream"] = shard["stream"]
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def pair_code(a, b, c):
    """64-bit hash of an (address, address/port) pair; collisions are negligible at these sizes."""
    h = (a.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (b.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
    return (h ^ (c.astype(np.uint64) * np.uint64(0x165667B19E3779F9))).view(np.int64)


def cardinality(f):
    """Causal per-stream cardinality context (uses only flows at or before each flow in availability order)."""
    f = f.sort_values(["stream", "hi", "source_order"], kind="stable").reset_index(drop=True)
    stream = pd.factorize(f["stream"])[0].astype(np.int64)
    src = pair_code(f["s0"].to_numpy(), f["s1"].to_numpy(), stream)
    dst = pair_code(f["d0"].to_numpy(), f["d1"].to_numpy(), stream)
    t = f["hi"].to_numpy(np.float64)
    rel = t - pd.Series(t).groupby(stream).transform("min").to_numpy()
    port = f["dport"].to_numpy().astype(np.int64)
    flags = {}
    for name, a, b in (("sp", src, port), ("sd", src, dst), ("dp", dst, port), ("ds", dst, src)):
        key = pair_code(a, b, np.zeros_like(a))
        prev = pd.Series(t).groupby(key).shift(1).to_numpy()  # previous occurrence in availability order
        flags[name] = (np.isnan(prev) | (t - prev > HORIZON)).astype(np.float64)
    out = {}
    for role, ep, names in (("src", src, ("sp", "sd")), ("dst", dst, ("dp", "ds"))):
        order = np.lexsort((np.arange(len(t)), rel, ep))
        group = pd.factorize(ep[order])[0].astype(np.float64)
        composite = group * 1e7 + rel[order]
        for w in WINDOWS:
            start = np.searchsorted(composite, composite - w, side="right")
            idx = np.arange(len(order))
            count = idx - start + 1
            res = {f"{role}_flows_{int(w)}s": count.astype(np.float64)}
            for n in names:
                c = np.cumsum(flags[n][order])
                s = c - np.where(start > 0, c[np.maximum(start - 1, 0)], 0.0)
                res[f"{role}_new_{n}_{int(w)}s"] = s
                res[f"{role}_new_{n}_frac_{int(w)}s"] = s / count
            for k, v in res.items():
                col = np.empty(len(t))
                col[order] = v
                out[k] = col
    ctx = pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})
    return f, ctx


def cm_summary(cm):
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
            "accuracy": float(c / s), "per_class_f1": dict(zip(classes, np.round(f1_k, 4).tolist()))}


log("loading train/val")
train, val = load("train"), load("val")
log("rows", len(train), len(val))
train, ctx_tr = cardinality(train)
val, ctx_va = cardinality(val)
log("cardinality features", list(ctx_tr.columns))
flow_cols = prep["feature_names"] + prep["schema"]["categorical"]
ytr, yva = train["y"].to_numpy(), val["y"].to_numpy()
keep_tr = ytr >= 0
rng = np.random.default_rng(7)
if keep_tr.sum() > MAX_TRAIN:  # stratified cap: keep every row of classes with < MAX_TRAIN/K rows
    counts = np.bincount(ytr[keep_tr], minlength=K)
    rate = np.minimum(1.0, (MAX_TRAIN / K) / np.maximum(counts, 1))
    rate = np.maximum(rate, MAX_TRAIN / keep_tr.sum() * 0.5)
    keep_tr &= rng.random(len(ytr)) < rate[np.maximum(ytr, 0)]
weights = None
keep_va = yva >= 0
report = {"classes": classes, "train_rows_used": int(keep_tr.sum()), "val_rows": int(keep_va.sum()),
          "class_prior_train_full": np.bincount(ytr[ytr >= 0], minlength=K).tolist()}
prior = np.bincount(ytr[ytr >= 0], minlength=K) / (ytr >= 0).sum()
sample_prior = np.bincount(ytr[keep_tr], minlength=K) / keep_tr.sum()
import lightgbm as lgb  # noqa: E402

# Class-separation of the new features for label-conflicting validation vectors (descriptive).
report["feature_means_by_class_val"] = {c: {k: float(ctx_va[k][keep_va & (yva == i)].mean()) for k in ctx_va.columns if k.endswith("_60s")}
                                        for i, c in enumerate(classes) if (yva == i).any()}
xtr_flow, xva_flow = train.loc[keep_tr, flow_cols].reset_index(drop=True), val.loc[keep_va, flow_cols].reset_index(drop=True)
ctx_tr, ctx_va = ctx_tr.loc[keep_tr].reset_index(drop=True), ctx_va.loc[keep_va].reset_index(drop=True)
ytr, yva = ytr[keep_tr], yva[keep_va]
del train, val
sets = {"flow": (xtr_flow, xva_flow),
        "flow+cardinality": (pd.concat([xtr_flow, ctx_tr], axis=1), pd.concat([xva_flow, ctx_va], axis=1))}
for name, (xtr, xva) in sets.items():
    started = time.time()
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 42, "max_bin": 255}
    booster = lgb.train(params, lgb.Dataset(xtr, ytr), num_boost_round=600,
                        valid_sets=[lgb.Dataset(xva, yva)],
                        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)])
    p = booster.predict(xva, num_iteration=booster.best_iteration)
    p = p / sample_prior * prior  # undo the stratified cap (prior shift back to the training distribution)
    res = {}
    for rule, pred in (("bayes", p.argmax(1)), ("balanced", (p / prior).argmax(1))):
        res[rule] = cm_summary(np.bincount(yva * K + pred, minlength=K * K).reshape(K, K))
    gain = dict(zip(xtr.columns, booster.feature_importance("gain").tolist()))
    report[name] = {"validation": res, "trees": booster.best_iteration, "seconds": time.time() - started,
                    "top_gain": sorted(gain.items(), key=lambda kv: -kv[1])[:25]}
    log(name, {r: (round(v["macro_mcc"], 4), round(v["mcc"], 4)) for r, v in res.items()})
    log(name, "per-class F1 (bayes)", res["bayes"]["per_class_f1"])
    json.dump(report, open(os.path.join(OUT, "hypothesis.json"), "w"), indent=1)
print(json.dumps({k: report[k] for k in ("train_rows_used", "val_rows")}, indent=1))
for name in sets:
    print(name, json.dumps({r: {m: v[m] for m in ("macro_mcc", "mcc", "accuracy", "per_class_f1")}
                            for r, v in report[name]["validation"].items()}, indent=1))
    print(" top gain", report[name]["top_gain"][:15])
log("hypothesis kernel done")

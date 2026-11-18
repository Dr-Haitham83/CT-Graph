# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# LEAVE-ONE-CAPTURE-OUT domain-generalisation analysis inside the SOURCE site (train + validation + calibration rows of
# one cross-site run; the other site, i.e. the test split, is never opened). Each capture is a domain: models are fit
# on the other captures and scored on the held-out one, which proxies the cross-site shift per class. Configurations:
#   C0 flow | C1 flow + raw causal cardinality context | C2 flow + segment-standardised context (v4b context_norm
#   analogue) | C3 C2 minus the flow features that drift most across captures (chosen inside each fold) | C4 C2 with
#   class-balanced weights.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

OUT = "/data/output/loco"
os.makedirs(OUT, exist_ok=True)
WINDOWS = (10.0, 60.0, 600.0)
HORIZON = 3600.0
rng = np.random.default_rng(20260926)
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
classes = prep["classes"]
K = len(classes)
width = len(prep["means"])
FLOW = prep["feature_names"] + prep["schema"]["categorical"]
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
        f = pd.DataFrame(np.concatenate([x, np.column_stack(cats)], 1) if cats else x, columns=FLOW)
        f["y"] = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        for c in ("hi", "source_order", "s0", "s1", "d0", "d1", "dport"):
            f[c] = t[c].to_numpy(zero_copy_only=False)
        f["stream"] = shard["stream"]
        f["split"] = split
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def pair_code(a, b, c):
    h = (a.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ (b.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F))
    return (h ^ (c.astype(np.uint64) * np.uint64(0x165667B19E3779F9))).view(np.int64)


def cardinality(f):
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
            for k, v in cols.items():
                col = np.empty(len(t))
                col[order0[order]] = v
                out[k] = col
    raw = pd.DataFrame(out)
    # Segment-standardised analogue of context_norm: log activity z-scored by the causal running mean/std of the stream.
    logs = np.log1p(raw)
    s_ord = pd.Series(pd.factorize(f["stream"])[0])
    norm = {}
    order_all = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), s_ord.to_numpy()))
    for k in raw.columns:
        v = logs[k].to_numpy()[order_all]
        g = s_ord.to_numpy()[order_all]
        csum = pd.Series(v).groupby(g).cumsum().to_numpy()
        csq = pd.Series(v * v).groupby(g).cumsum().to_numpy()
        n = pd.Series(np.ones_like(v)).groupby(g).cumsum().to_numpy()
        mean = csum / n
        var = np.maximum(csq / n - mean * mean, 0)
        z = np.empty(len(v))
        z[order_all] = (v - mean) / np.sqrt(var + 1e-2)
        norm["z_" + k] = z
    return raw.astype(np.float32), pd.DataFrame(norm).astype(np.float32)


frames = [load(s) for s in ("train", "val", "calibration")]
data = pd.concat(frames, ignore_index=True)
data = data[data["y"] >= 0].reset_index(drop=True)
raw_ctx, norm_ctx = cardinality(data)
capture = data["stream"].str.replace(r"/segment-\d+$", "", regex=True).to_numpy()
captures = np.unique(capture)
if len(captures) < 3:  # fall back to segments when the source site has fewer than three captures
    capture = data["stream"].to_numpy()
    captures = np.unique(capture)
y = data["y"].to_numpy().astype(np.int64)
report = {"prepared": PREPARED_NAME, "classes": classes, "rows": int(len(y)), "domains": int(len(captures)),
          "domain_unit": "capture" if "/segment-" in str(data["stream"].iloc[0]) else "stream",
          "class_counts": dict(zip(classes, np.bincount(y, minlength=K).tolist())),
          "protocol": "source-site train+val+calibration only; leave-one-capture-out; the other site (test) is never opened"}
log("rows", len(y), "domains", len(captures))
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
          "num_threads": os.cpu_count() or 4, "seed": 7}


def drift_features(tr):
    """Flow features that best separate the training captures (multi-domain drift), by LightGBM gain."""
    doms = pd.factorize(capture[tr])[0]
    sub = rng.choice(np.flatnonzero(tr), min(200_000, tr.sum()), replace=False)
    m = lgb.train({**PARAMS, "num_class": int(doms.max()) + 1, "objective": "multiclass"},
                  lgb.Dataset(data.loc[sub, FLOW], pd.factorize(capture[sub])[0]), num_boost_round=60)
    gain = dict(zip(FLOW, m.feature_importance("gain")))
    return [k for k, _ in sorted(gain.items(), key=lambda kv: -kv[1])[:8]]


def evaluate(name, columns_fn, weight_fn=None):
    preds = np.full(len(y), -1)
    dropped = []
    for d in captures:
        te = capture == d
        tr = ~te
        if len(np.unique(y[tr])) < 2 or not te.any():
            continue
        cols = columns_fn(tr)
        dropped.append([c for c in FLOW if c not in cols])
        X = pd.concat([data[[c for c in cols if c in data.columns]], raw_ctx[[c for c in cols if c in raw_ctx.columns]],
                       norm_ctx[[c for c in cols if c in norm_ctx.columns]]], axis=1)
        w = weight_fn(tr) if weight_fn else None
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], weight=w), num_boost_round=200)
        preds[te] = m.predict(X[te]).argmax(1)
    ok = preds >= 0
    cm = np.bincount(y[ok] * K + preds[ok], minlength=K * K).reshape(K, K).astype(float)
    t, p = cm.sum(1), cm.sum(0)
    tp = np.diag(cm)
    s = cm.sum()
    fn, fp = t - tp, p - tp
    tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(K), where=(2 * tp + fp + fn) > 0)
    prec = np.divide(tp, p, out=np.zeros(K), where=p > 0)
    present = t > 0
    res = {"loco_macro_mcc": float(mcc[present].mean()),
           "per_class_f1": {classes[i]: round(float(f1[i]), 4) for i in range(K) if present[i]},
           "per_class_precision": {classes[i]: round(float(prec[i]), 4) for i in range(K) if present[i]},
           "top_confusions": sorted([(int(cm[i, j]), classes[i], classes[j]) for i in range(K) for j in range(K) if i != j and cm[i, j] > 0], reverse=True)[:6]}
    if dropped and dropped[0]:
        res["dropped_example"] = dropped[0]
    report.setdefault("configs", {})[name] = res
    log(name, round(res["loco_macro_mcc"], 4), res["per_class_f1"])
    json.dump(report, open(os.path.join(OUT, "loco.json"), "w"), indent=1, default=float)


RAW, NORM = list(raw_ctx.columns), list(norm_ctx.columns)
evaluate("C0_flow", lambda tr: FLOW)
evaluate("C1_flow+raw_context", lambda tr: FLOW + RAW)
evaluate("C2_flow+segment_standardised_context", lambda tr: FLOW + NORM)
evaluate("C3_C2_minus_capture_drift_features", lambda tr: [c for c in FLOW if c not in drift_features(tr)] + NORM)
counts = np.bincount(y, minlength=K).astype(float)
evaluate("C4_C2_class_balanced", lambda tr: FLOW + NORM, lambda tr: (1.0 / np.maximum(counts[y[tr]], 1)) ** 0.5)
best = max(report["configs"], key=lambda k: report["configs"][k]["loco_macro_mcc"])
report["best_config"] = best
json.dump(report, open(os.path.join(OUT, "loco.json"), "w"), indent=1, default=float)
for name, res in report["configs"].items():
    print(f"{name:40s} LOCO macro-MCC {res['loco_macro_mcc']:.4f}  F1 {res['per_class_f1']}")
    print("   precision", res["per_class_precision"], "\n   confusions", res["top_confusions"][:4])
log("loco done; best", best)


# CAUSAL TEMPORAL SMOOTHING OF THE GRAPH BRANCH (merged monitor timeline; KEY = src or pair; tau = 10 s; mix = 0.5).
# Lab 4 / lab 5 found this smoothing helps the context branch; here the same post-processing is applied to the frozen
# v4b graph seeds. PRE-REGISTERED ADOPTION (validation only, like every graph-branch decision; the graph pipeline keeps
# calibration for its own temperature / conformal steps): (i) smoothed graph ensemble val macro-MCC >= unsmoothed + 0.002
# AND (ii) equal-weight hybrid with the deployed context branch: smoothed-graph hybrid >= unsmoothed-graph hybrid + 0.002.
# Only if adopted are the test predictions smoothed (no test metric is computed here) and written in the graph-run
# layout (evaluations/<id>/predictions-rank0.parquet + validation/) for the post-hoc.
import shutil

OUT = "/data/output/gsmooth"
os.makedirs(OUT, exist_ok=True)
TAU, MIX = 10.0, 0.5
KEY = globals().get("KEY", "src")
started = time.time()


@njit(cache=False)
def ema_same_group(g, t, P, tau):
    n, k = P.shape
    out = np.zeros((n, k))
    has = np.zeros(n)
    S = np.zeros(k)
    W = 0.0
    for i in range(n):
        if i > 0 and g[i] == g[i - 1]:
            d = np.exp(-(t[i] - t[i - 1]) / tau)
            for c in range(k):
                S[c] = d * (S[c] + P[i - 1, c])
            W = d * (W + 1.0)
        else:
            for c in range(k):
                S[c] = 0.0
            W = 0.0
        if W > 1e-9:
            for c in range(k):
                out[i, c] = S[c] / W
            has[i] = 1.0
    return out, has


def split_meta(split):
    f = load(split, ids=True)
    tm, _, how = merged_clock(f)
    ids = f["source_id"].to_numpy().astype(str)
    o = np.argsort(ids)
    m = {"ids": ids[o], "s0": f["s0"].to_numpy()[o], "s1": f["s1"].to_numpy()[o], "d0": f["d0"].to_numpy()[o],
         "d1": f["d1"].to_numpy()[o], "t": tm[o], "so": f["source_order"].to_numpy()[o]}
    del f
    log("meta", split, len(ids), how)
    return m


def smooth(meta, ids, P):
    pos = np.searchsorted(meta["ids"], ids)
    assert np.all(pos < len(meta["ids"])) and np.array_equal(meta["ids"][pos], ids)
    zero = np.zeros(len(ids), np.int64)
    g = key(meta["s0"][pos], meta["s1"][pos], zero)
    if KEY == "pair":
        g = key(g, key(meta["d0"][pos], meta["d1"][pos], zero))
    t, so = meta["t"][pos], meta["so"][pos]
    order = np.lexsort((so, t, g))
    ema, has = ema_same_group(g[order], t[order], P[order].astype(np.float64), TAU)
    out = np.empty_like(P, dtype=np.float64)
    out[order] = np.where(has[:, None] > 0, (1 - MIX) * P[order] + MIX * ema, P[order])
    return out


def read_preds(directory):
    files = sorted(glob.glob(os.path.join(directory, "predictions-rank*.parquet")))
    t = pa.concat_tables([pq.read_table(p) for p in files])
    return t


runs = [p for p in sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True))
        if glob.glob(os.path.join(os.path.dirname(p), "validation", "predictions-rank*.parquet"))]
log("graph runs", runs)
res0 = json.load(open(runs[0]))
cls = res0["classes"]
KC = len(cls)
cnt = np.asarray([res0["split_class_counts"]["train"].get(c, 0) for c in cls], float)
prior_g = cnt / cnt.sum()


def macro(y, P, rule):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior_g, 1e-12)).argmax(1)
    cm = np.bincount(y * KC + pred, minlength=KC * KC).reshape(KC, KC).astype(float)
    tt, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = tt - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(KC), where=den > 0)
    return float(mcc[tt > 0].mean())


def best(y, P):
    return max(macro(y, P, r) for r in ("bayes", "balanced"))


meta_val = split_meta("val")
vals = []
for p in runs:
    t = read_preds(os.path.join(os.path.dirname(p), "validation"))
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object).astype(str)
    o = np.argsort(ids)
    P = t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1)[o].astype(np.float64)
    vals.append((ids[o], t["label"].to_numpy()[o], P))
ids_v, y_v = vals[0][0], vals[0][1]
G = np.mean([v[2] for v in vals], 0)
Gs = np.mean([smooth(meta_val, ids_v, v[2]) for v in vals], 0)
keep = y_v >= 0
ct = [np.load(p, allow_pickle=False) for p in sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))]
C = []
for c in ct:
    cid = c["val_ids"].astype(str)
    o = np.argsort(cid)
    pos = np.searchsorted(cid[o], ids_v[keep])
    assert np.array_equal(cid[o][pos], ids_v[keep])
    C.append(c["val_proba"].astype(np.float64)[o][pos])
C = np.mean(C, 0)
y = y_v[keep].astype(np.int64)
r = {"graph": best(y, G[keep]), "graph_smoothed": best(y, Gs[keep]),
     "hybrid": best(y, 0.5 * G[keep] + 0.5 * C), "hybrid_smoothed_graph": best(y, 0.5 * Gs[keep] + 0.5 * C)}
r["graph_gain"], r["hybrid_gain"] = r["graph_smoothed"] - r["graph"], r["hybrid_smoothed_graph"] - r["hybrid"]
r["adopt"] = bool(r["graph_gain"] >= 0.002 and r["hybrid_gain"] >= 0.002)
r.update(key=KEY, tau_s=TAU, mix=MIX, runs=runs, context_files=len(ct))
json.dump(r, open(os.path.join(OUT, "gsmooth.json"), "w"), indent=1, default=float)
log("validation:", {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, "ADOPT" if r["adopt"] else "not adopted")
if not r["adopt"]:
    raise SystemExit(0)
meta_test = split_meta("test")
for i, p in enumerate(runs):
    src_dir = os.path.dirname(p)
    dst_dir = os.path.join("/data/output", f"gsmooth-run{i}", "evaluations", os.path.basename(src_dir))
    os.makedirs(os.path.join(dst_dir, "validation"), exist_ok=True)
    shutil.copy2(p, os.path.join(dst_dir, "results.json"))
    for sub, meta in (("", meta_test), ("validation", meta_val)):
        t = read_preds(os.path.join(src_dir, sub))
        ids = np.asarray(t["source_id"].to_pylist(), dtype=object).astype(str)
        P = t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1).astype(np.float64)
        Ps = smooth(meta, ids, P).astype(np.float32)
        arr = pa.ListArray.from_arrays(pa.array(np.arange(0, Ps.size + 1, Ps.shape[1], dtype=np.int32)), pa.array(Ps.ravel(), pa.float32()))
        t = t.set_column(t.schema.get_field_index("probabilities"), pa.field("probabilities", t.schema.field("probabilities").type), arr.cast(t.schema.field("probabilities").type))
        t = t.drop([c for c in t.column_names if c.endswith("_prediction")])  # stale (unsmoothed) decisions
        pq.write_table(t, os.path.join(dst_dir, sub, "predictions-rank0.parquet"))
    log("smoothed graph run", i, "->", dst_dir)
log("graph smoothing done")

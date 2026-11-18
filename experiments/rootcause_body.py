# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# ROOT-CAUSE ANALYSIS of the confusable class pairs (train / validation / calibration only; test never opened).
# 1. Causal relational feature library per segment-scoped stream (availability order; only earlier flows of the same
#    stream contribute): per source / destination / pair in 10 s and 60 s windows - flows, distinct destination ports,
#    distinct peers, distinct source ports, sequential-port rate, SYN-only / RST / empty-response fractions, inter-arrival
#    regularity (CV of log gaps) and burstiness (10 s vs 60 s rate).
# 2. Invariance test (two environments = train and validation, i.e. different campaigns): for every confusable pair and
#    feature, AUC and mutual information in EACH environment with the orientation fixed on train, and the
#    class-conditional shift (KS). A feature is invariant-discriminative if min(AUC_train, AUC_val) >= 0.75.
# 3. Formal bound: refining identical-vector groups by the invariant features gives the context-conditioned per-flow
#    Bayes-error lower bound, i.e. how much of the "irreducible" error context can remove (train and validation).
# 4. Confirmation: LightGBM flow vs flow+all relational vs flow+invariant subset (validation for analysis, calibration
#    as an independent confirmation set).
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

OUT = "/data/output/rootcause"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260926)
WINDOWS = (10.0, 60.0)
PAIRS = {"nids-ton": [("scanning", "Benign"), ("mitm", "Benign"), ("ransomware", "Benign"), ("injection", "Benign"),
                      ("injection", "xss"), ("dos", "xss"), ("password", "Benign")],
         "nids-cic": [("dos_attacks-slowhttptest", "ftp-bruteforce"), ("infilteration", "Benign"), ("brute_force_-web", "Benign"),
                      ("brute_force_-xss", "Benign"), ("sql_injection", "Benign"), ("brute_force_-xss", "sql_injection"),
                      ("dos_attacks-slowloris", "Benign")],
         "nids-5gs-bs1": [("udpscan", "udpflood"), ("udpscan", "Benign"), ("httpflood", "slowratedos"), ("Benign", "udpflood")],
         "nids-5gs-bs2": [("udpscan", "udpflood"), ("udpscan", "Benign"), ("httpflood", "slowratedos"), ("Benign", "udpflood")]}
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
classes = prep["classes"]
K = len(classes)
width = len(prep["means"])
FLOW = prep["feature_names"] + prep["schema"]["categorical"]
label_set = pa.array(classes)
names = prep["feature_names"]


def pick(*candidates):
    return next((c for c in candidates if c in names), None)


FLAGS = pick("TCP_FLAGS", "CLIENT_TCP_FLAGS")
OUT_BYTES = pick("OUT_BYTES", "DstBytes")
DUR = pick("FLOW_DURATION_MILLISECONDS", "Dur", "duration_s")


def shard_path(split, shard):
    return shard["path"] if os.path.exists(shard["path"]) else os.path.join(root, "prepared", split, os.path.basename(shard["path"]))


def load(split):
    frames = []
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=["numeric", "categories", "label", "hi", "source_order",
                                                             "s0", "s1", "d0", "d1", "sport", "dport"])
        x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
        cats = []
        for j, vocab in enumerate(prep["vocabs"]):
            values = pc.list_element(t["categories"], j)
            index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
            cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
        f = pd.DataFrame(np.concatenate([x, np.column_stack(cats)], 1) if cats else x, columns=FLOW)
        f["y"] = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        for c in ("hi", "source_order", "s0", "s1", "d0", "d1", "sport", "dport"):
            f[c] = t[c].to_numpy(zero_copy_only=False)
        f["stream"] = shard["stream"]
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    return f[f["y"] >= 0].reset_index(drop=True)


def key(*cols):
    h = np.zeros(len(cols[0]), np.uint64)
    for i, c in enumerate(cols):
        h ^= (np.asarray(c).astype(np.int64).astype(np.uint64) + np.uint64(i + 1)) * np.uint64([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                                                                                            0x27D4EB2F165667C5][i % 4])
        h = (h ^ (h >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9)
    return h.view(np.int64)


def relational(f):
    """Causal window statistics per group (only rows at or before each row in availability order within its stream)."""
    stream = pd.factorize(f["stream"])[0].astype(np.int64)
    order0 = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), stream))
    t = f["hi"].to_numpy(np.float64)[order0]
    st = stream[order0]
    rel = t - pd.Series(t).groupby(st).transform("min").to_numpy()
    src = key(f["s0"].to_numpy()[order0], f["s1"].to_numpy()[order0], st)
    dst = key(f["d0"].to_numpy()[order0], f["d1"].to_numpy()[order0], st)
    dport = f["dport"].to_numpy()[order0].astype(np.int64)
    sport = f["sport"].to_numpy()[order0].astype(np.int64)
    flags = np.nan_to_num(f[FLAGS].to_numpy()[order0], nan=0).astype(np.int64) if FLAGS else np.zeros(len(t), np.int64)
    syn_only = (((flags & 0x02) > 0) & ((flags & 0x10) == 0)).astype(np.float64)
    rst = ((flags & 0x04) > 0).astype(np.float64)
    empty = (np.nan_to_num(f[OUT_BYTES].to_numpy()[order0], nan=0) <= 0).astype(np.float64) if OUT_BYTES else np.zeros(len(t))
    out = {}

    def first_seen(group, value, horizon):
        prev = pd.Series(rel).groupby(key(group, value)).shift(1).to_numpy()
        return (np.isnan(prev) | (rel - prev > horizon)).astype(np.float64)

    for role, g in (("src", src), ("dst", dst), ("pair", key(src, dst))):
        order = np.lexsort((np.arange(len(t)), rel, g))
        gid = pd.factorize(g[order])[0].astype(np.float64)
        composite = gid * 1e7 + rel[order]
        signals = {"newdport": first_seen(g, dport, 60.0)[order], "syn": syn_only[order], "rst": rst[order], "empty": empty[order]}
        if role != "pair":
            peer = dst if role == "src" else src
            signals["newpeer"] = first_seen(g, peer, 60.0)[order]
            signals["newsport"] = first_seen(g, sport, 60.0)[order]
        if role == "src":
            dp = dport[order]
            same = np.r_[False, gid[1:] == gid[:-1]]
            signals["seqport"] = (same & (np.abs(dp - np.r_[dp[0], dp[:-1]]) == 1)).astype(np.float64)
            gap = np.where(same, np.diff(rel[order], prepend=0.0), np.nan)
            lg = np.log1p(np.nan_to_num(gap, nan=0.0))
            signals["lgap"], signals["lgap2"] = lg, lg * lg
        counts = {}
        for w in WINDOWS:
            start = np.searchsorted(composite, composite - w, side="right")
            n = np.arange(len(order)) - start + 1
            counts[w] = n
            res = {f"{role}_flows_{int(w)}s": np.log1p(n)}
            for name, sig in signals.items():
                c = np.cumsum(sig)
                s = c - np.where(start > 0, c[np.maximum(start - 1, 0)], 0.0)
                if name in ("lgap", "lgap2"):
                    continue
                res[f"{role}_{name}_frac_{int(w)}s"] = s / n
                if name.startswith("new"):
                    res[f"{role}_{name}_{int(w)}s"] = np.log1p(s)
            if role == "src":
                c1, c2 = np.cumsum(signals["lgap"]), np.cumsum(signals["lgap2"])
                s1 = c1 - np.where(start > 0, c1[np.maximum(start - 1, 0)], 0.0)
                s2 = c2 - np.where(start > 0, c2[np.maximum(start - 1, 0)], 0.0)
                mean = s1 / n
                sd = np.sqrt(np.maximum(s2 / n - mean * mean, 0))
                res[f"src_gap_cv_{int(w)}s"] = sd / np.maximum(mean, 1e-3)
            for k, v in res.items():
                col = np.empty(len(t), np.float32)
                col[order0[order]] = v
                out[k] = col
        burst = np.empty(len(t), np.float32)
        burst[order0[order]] = np.log1p(counts[10.0]) - np.log1p(counts[60.0])
        out[f"{role}_burstiness"] = burst
    return pd.DataFrame(out)


def hashes(x):
    return pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(x, nan=-7.77e30)), index=False).to_numpy()


started = time.time()
data = {s: load(s) for s in ("train", "val", "calibration")}
REL = None
rel = {}
for s, f in data.items():
    rel[s] = relational(f)
    REL = list(rel[s].columns)
log("loaded + relational features", {s: len(f) for s, f in data.items()}, len(REL), "features", round(time.time() - started), "s")
report = {"prepared": PREPARED_NAME, "classes": classes, "relational_features": REL, "pairs": {},
          "protocol": "train/validation/calibration only; test never opened"}
cidx = {c: i for i, c in enumerate(classes)}
ytr, yva, yca = (data[s]["y"].to_numpy() for s in ("train", "val", "calibration"))
# identical-vector groups (train) for the formal bound
cols = [c for c in FLOW if not np.all(np.isnan(data["train"][c].to_numpy(np.float32)))]
h_tr = hashes(data["train"][cols].to_numpy(np.float32))
h_va = hashes(data["val"][cols].to_numpy(np.float32))
h_ca = hashes(data["calibration"][cols].to_numpy(np.float32))


def auc_mi(xa, xb):
    lab = np.r_[np.ones(len(xa)), np.zeros(len(xb))]
    v = np.nan_to_num(np.r_[xa, xb], nan=0.0)
    if len(np.unique(v)) < 2:
        return 0.5, 0.0
    auc = roc_auc_score(lab, v)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, 17)[1:-1]))
    b = np.searchsorted(edges, v)
    joint = np.zeros((len(edges) + 1, 2))
    np.add.at(joint, (b, lab.astype(int)), 1)
    joint /= joint.sum()
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(auc), float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())


def ks(a, b):
    a, b = np.sort(np.nan_to_num(a, nan=0)), np.sort(np.nan_to_num(b, nan=0))
    grid = np.unique(np.r_[a, b])
    if len(grid) > 2000:
        grid = np.quantile(grid, np.linspace(0, 1, 2000))
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b))))


invariant_union = set()
for a_name, b_name in PAIRS.get(PREPARED_NAME, []):
    if a_name not in cidx or b_name not in cidx:
        continue
    a, b = cidx[a_name], cidx[b_name]
    sets = {}
    for env, y_, X in (("train", ytr, rel["train"]), ("val", yva, rel["val"])):
        ia, ib = np.flatnonzero(y_ == a), np.flatnonzero(y_ == b)
        if len(ia) < 30 or len(ib) < 30:
            sets[env] = None
            continue
        ia = rng.choice(ia, min(len(ia), 100_000), replace=False)
        ib = rng.choice(ib, min(len(ib), 100_000), replace=False)
        sets[env] = (ia, ib, X)
    if sets.get("train") is None or sets.get("val") is None:
        report["pairs"][f"{a_name} vs {b_name}"] = {"status": "too few rows in train or validation"}
        continue
    feats = {}
    for fname in REL:
        tr_auc, tr_mi = auc_mi(sets["train"][2][fname].to_numpy()[sets["train"][0]], sets["train"][2][fname].to_numpy()[sets["train"][1]])
        va_auc, va_mi = auc_mi(sets["val"][2][fname].to_numpy()[sets["val"][0]], sets["val"][2][fname].to_numpy()[sets["val"][1]])
        sign = 1 if tr_auc >= 0.5 else -1  # orientation fixed on TRAIN
        tr_o, va_o = (tr_auc, va_auc) if sign > 0 else (1 - tr_auc, 1 - va_auc)
        shift_a = ks(sets["train"][2][fname].to_numpy()[sets["train"][0]], sets["val"][2][fname].to_numpy()[sets["val"][0]])
        feats[fname] = {"auc_train": round(tr_o, 4), "auc_val": round(va_o, 4), "mi_train_bits": round(tr_mi, 4), "mi_val_bits": round(va_mi, 4),
                        "ks_shift_class_a": round(shift_a, 4), "invariant_score": round(min(tr_o, va_o), 4)}
    ranked = sorted(feats.items(), key=lambda kv: -kv[1]["invariant_score"])
    invariant = [k for k, v in ranked if v["invariant_score"] >= 0.75]
    invariant_union.update(invariant[:6])
    # Formal bound, pair rows only. In-sample: plug-in Bayes error of identical-vector groups on TRAIN (and refined by
    # the top-3 train-ranked features on train-quantile bins; in-sample refinement is optimistic by construction).
    # Out-of-sample: a lookup classifier fitted on TRAIN groups, scored on VALIDATION rows whose group exists in
    # train - per-flow (vector groups) vs context-refined (vector x context-bin groups). No validation label is used
    # to fit anything.
    bound = {}
    top = [k for k, _ in ranked[:3]]
    mt, mv, mc = (ytr == a) | (ytr == b), (yva == a) | (yva == b), (yca == a) | (yca == b)
    zt, zv, zc = np.zeros((mt.sum(), 0), np.int64), np.zeros((mv.sum(), 0), np.int64), np.zeros((mc.sum(), 0), np.int64)
    if top:
        cols_t, cols_v, cols_c = [], [], []
        for t_ in top:
            xt_, xv_ = rel["train"][t_].to_numpy()[mt], rel["val"][t_].to_numpy()[mv]
            edges = np.unique(np.quantile(np.nan_to_num(xt_), np.linspace(0, 1, 9)[1:-1]))
            cols_t.append(np.searchsorted(edges, np.nan_to_num(xt_))); cols_v.append(np.searchsorted(edges, np.nan_to_num(xv_)))
            cols_c.append(np.searchsorted(edges, np.nan_to_num(rel["calibration"][t_].to_numpy()[mc])))
        zt, zv, zc = np.column_stack(cols_t), np.column_stack(cols_v), np.column_stack(cols_c)
    for label_, groups_t, groups_v, groups_c in (("per_flow", h_tr[mt], h_va[mv], h_ca[mc]),
                                        ("flow_plus_top3_context", key(h_tr[mt], *[zt[:, i] for i in range(zt.shape[1])]),
                                         key(h_va[mv], *[zv[:, i] for i in range(zv.shape[1])]),
                                         key(h_ca[mc], *[zc[:, i] for i in range(zc.shape[1])]))):
        yt_, yv_ = (ytr[mt] == a).astype(int), (yva[mv] == a).astype(int)
        per = pd.DataFrame({"g": groups_t, "y": yt_}).groupby("g")["y"].agg(["sum", "count"])
        insample = float(np.minimum(per["sum"], per["count"] - per["sum"]).sum() / len(yt_))
        majority = (per["sum"] * 2 >= per["count"]).astype(int)
        order = np.argsort(per.index.to_numpy())
        idx = per.index.to_numpy()[order]
        pos = np.minimum(np.searchsorted(idx, groups_v), len(idx) - 1)
        hit = idx[pos] == groups_v
        pred = majority.to_numpy()[order][pos]
        bound[label_] = {"train_in_sample_bayes_error": insample,
                         "val_rows_covered": float(hit.mean()),
                         "val_out_of_sample_lookup_error": float((pred[hit] != yv_[hit]).mean()) if hit.any() else None,
                         "val_out_of_sample_class_a_recall": float(((pred == 1) & (yv_ == 1) & hit).sum() / max(((yv_ == 1) & hit).sum(), 1)),
                         "train_rows": int(len(yt_)), "val_rows": int(len(yv_)), "val_rows_covered_n": int(hit.sum()),
                         "val_errors_n": int((pred[hit] != yv_[hit]).sum())}
        # HELD-OUT: the calibration split took no part in ranking the features (train + validation) or in fitting the lookup.
        yc_ = (yca[mc] == a).astype(int)
        posc = np.minimum(np.searchsorted(idx, groups_c), len(idx) - 1)
        hitc = idx[posc] == groups_c
        predc = majority.to_numpy()[order][posc]
        bound[label_].update({"cal_rows": int(len(yc_)), "cal_rows_a": int(yc_.sum()), "cal_rows_covered_n": int(hitc.sum()),
                              "cal_rows_covered": float(hitc.mean()) if len(yc_) else None,
                              "cal_lookup_errors_n": int((predc[hitc] != yc_[hitc]).sum()),
                              "cal_out_of_sample_lookup_error": float((predc[hitc] != yc_[hitc]).mean()) if hitc.any() else None})
    report["pairs"][f"{a_name} vs {b_name}"] = {"top_features": ranked[:10], "invariant_features": invariant,
                                                "n_invariant": len(invariant), "formal_bound": bound}
    log(a_name, "vs", b_name, "invariant", invariant[:5], "bound", json.dumps(bound))
    json.dump(report, open(os.path.join(OUT, "rootcause.json"), "w"), indent=1, default=float)

# Confirmation with LightGBM (fixed hyper-parameters, no early stopping on validation).
if globals().get("SKIP_CONFIRM"):
    log("root-cause bound done (confirmation skipped)")
    raise SystemExit(0)
MAX_TRAIN = 3_000_000
n = np.bincount(ytr, minlength=K)
rate = np.minimum(1.0, np.maximum(MAX_TRAIN / len(ytr), (MAX_TRAIN / (2 * K)) / np.maximum(n, 1)))
keep = rng.random(len(ytr)) < rate[ytr]
w = 1.0 / rate[ytr[keep]]
prior = n / n.sum()
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}


def macro(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(K), where=(2 * tp + fp + fn) > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(f1[i]), 4) for i in range(K) if present[i]}


# Scale-free context: fractions, coefficients of variation and burstiness ratios are invariant to a global change of
# attack/traffic intensity; absolute counts are not. Segment-standardised counts z-score each log count by the causal
# running mean/std of its own segment (the v4b context_norm analogue).
SCALE_FREE = [c for c in REL if "_frac_" in c or "gap_cv" in c or "burstiness" in c]
COUNTS = [c for c in REL if c not in SCALE_FREE]
for split in rel:
    f = data[split]
    order = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), pd.factorize(f["stream"])[0]))
    g = pd.factorize(f["stream"])[0][order]
    for c in COUNTS:
        v = rel[split][c].to_numpy(np.float64)[order]
        n_ = pd.Series(np.ones(len(v))).groupby(g).cumsum().to_numpy()
        m1 = pd.Series(v).groupby(g).cumsum().to_numpy() / n_
        m2 = pd.Series(v * v).groupby(g).cumsum().to_numpy() / n_
        z = np.empty(len(v), np.float32)
        z[order] = (v - m1) / np.sqrt(np.maximum(m2 - m1 * m1, 0) + 1e-2)
        rel[split]["z_" + c] = z
SEGZ = ["z_" + c for c in COUNTS]
configs = {"flow": [], "flow+all_relational": REL, "flow+invariant": sorted(invariant_union),
           "flow+scale_free": SCALE_FREE, "flow+scale_free+segment_z_counts": SCALE_FREE + SEGZ,
           "flow+scale_free_invariant": [c for c in sorted(invariant_union) if c in SCALE_FREE]}
report["confirmation"] = {}
for name, extra in configs.items():
    Xtr = pd.concat([data["train"].loc[keep, FLOW].reset_index(drop=True), rel["train"].loc[keep, extra].reset_index(drop=True)], axis=1)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr[keep], weight=w), num_boost_round=300)
    res = {}
    for split, ys in (("val", yva), ("calibration", yca)):
        X = pd.concat([data[split][FLOW], rel[split][extra]], axis=1)
        p = m.predict(X)
        rule = "bayes"
        mm_b, f1_b = macro(ys, p.argmax(1))
        mm_l, f1_l = macro(ys, (p / np.maximum(prior, 1e-12)).argmax(1))
        res[split] = {"bayes": mm_b, "balanced": mm_l, "per_class_f1_bayes": f1_b}
    report["confirmation"][name] = {"extra_features": len(extra), **res}
    log("confirm", name, {s: (round(v["bayes"], 4), round(v["balanced"], 4)) for s, v in res.items()})
    json.dump(report, open(os.path.join(OUT, "rootcause.json"), "w"), indent=1, default=float)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "top_features"} for k, v in report["pairs"].items()}, indent=1, default=float)[:15000])
print(json.dumps(report["confirmation"], indent=1, default=float))
log("root-cause analysis done")

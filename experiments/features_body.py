# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# ENGINEERED-FEATURE DISCOVERY for generalisation under campaign shift (train / validation / calibration only; the test
# split is never opened). Candidate per-flow families are derived from the raw flow fields by physical meaning and scale
# invariance; raw fields that act as host/campaign fingerprints are candidates for removal. A family is KEPT only if it
# improves validation AND calibration macro-MCC (both different campaigns from training) without hurting the worst
# validation segments (CVaR_0.25); kept families then face backward elimination; removal groups are tested the same way.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import lightgbm as lgb

OUT = "/data/output/features"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260926)
EPS = 1e-6
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
classes = prep["classes"]
K = len(classes)
width = len(prep["means"])
NUM = prep["feature_names"]
CAT = prep["schema"]["categorical"]
FLOW = NUM + CAT
label_set = pa.array(classes)


def shard_path(split, shard):
    return shard["path"] if os.path.exists(shard["path"]) else os.path.join(root, "prepared", split, os.path.basename(shard["path"]))


def load(split):
    frames = []
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=["numeric", "categories", "label"])
        x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
        cats = []
        for j, vocab in enumerate(prep["vocabs"]):
            values = pc.list_element(t["categories"], j)
            index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
            cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
        f = pd.DataFrame(np.concatenate([x, np.column_stack(cats)], 1) if cats else x, columns=FLOW)
        f["y"] = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        f["stream"] = shard["stream"]
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    return f[f["y"] >= 0].reset_index(drop=True)


def col(f, *names):
    for n in names:
        if n in f.columns:
            return np.nan_to_num(f[n].to_numpy(np.float64), nan=0.0)
    return None


def safe_div(a, b):
    return a / (np.abs(b) + EPS)


def derive(f):
    """Per-flow engineered families (only from fields the dataset has); every value is a ratio, a fraction, an
    indicator or a log rate, i.e. invariant to a global change of traffic intensity where possible."""
    fam = {}
    ib, ob = col(f, "IN_BYTES", "SrcBytes"), col(f, "OUT_BYTES", "DstBytes")
    ip_, op_ = col(f, "IN_PKTS", "SrcPkts"), col(f, "OUT_PKTS", "DstPkts")
    dur = col(f, "FLOW_DURATION_MILLISECONDS")
    dur = dur / 1000.0 if dur is not None else col(f, "Dur", "duration_s")
    F1 = {}
    if ib is not None and ip_ is not None:
        F1["mean_size_in"] = safe_div(ib, ip_)
        if ob is not None and op_ is not None:
            F1["mean_size_out"] = safe_div(ob, op_)
            F1["log_size_ratio"] = np.log1p(F1["mean_size_out"]) - np.log1p(F1["mean_size_in"])
    hist = [c for c in ("NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
                        "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES") if c in f.columns]
    if hist:
        H = np.column_stack([col(f, c) for c in hist])
        P = H / (H.sum(1, keepdims=True) + EPS)
        for i, c in enumerate(hist):
            F1[f"frac_{c.lower()}"] = P[:, i]
        F1["size_hist_entropy"] = -(np.where(P > 0, P * np.log(np.maximum(P, EPS)), 0)).sum(1)
    lo, hi = col(f, "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN"), col(f, "LONGEST_FLOW_PKT", "MAX_IP_PKT_LEN")
    if lo is not None and hi is not None:
        F1["pkt_len_range_ratio"] = np.log1p(hi) - np.log1p(lo)
    fam["F1_size_shape"] = F1
    F2 = {}
    if ib is not None and ob is not None:
        F2["log_byte_ratio"] = np.log1p(ob) - np.log1p(ib)
        F2["response_present"] = (ob > 0).astype(float)
    if ip_ is not None and op_ is not None:
        F2["log_pkt_ratio"] = np.log1p(op_) - np.log1p(ip_)
    fam["F2_direction_balance"] = F2
    F3 = {}
    for d, avg, sd, mx in (("fwd", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV", "SRC_TO_DST_IAT_MAX"),
                           ("bwd", "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV", "DST_TO_SRC_IAT_MAX"),
                           ("src", "SIntPkt", "SrcJitter", None), ("dst", "DIntPkt", "DstJitter", None)):
        a, s = col(f, avg), col(f, sd)
        if a is not None and s is not None:
            F3[f"iat_cv_{d}"] = safe_div(s, a)
            m = col(f, mx) if mx else None
            if m is not None:
                F3[f"iat_peak_{d}"] = safe_div(m, a)
    if dur is not None:
        F3["zero_duration"] = (dur <= 0).astype(float)
        if ib is not None and ob is not None:
            F3["log_byte_rate"] = np.log1p(safe_div(ib + ob, dur))
        if ip_ is not None and op_ is not None:
            F3["log_pkt_rate"] = np.log1p(safe_div(ip_ + op_, dur))
    if ip_ is not None and op_ is not None:
        F3["single_packet"] = ((ip_ + op_) <= 1).astype(float)
    fam["F3_temporal_shape"] = F3
    F4 = {}
    for side, name in (("all", "TCP_FLAGS"), ("cli", "CLIENT_TCP_FLAGS"), ("srv", "SERVER_TCP_FLAGS")):
        v = col(f, name)
        if v is None:
            continue
        v = v.astype(np.int64)
        for bit, flag in ((0x02, "syn"), (0x10, "ack"), (0x01, "fin"), (0x04, "rst"), (0x08, "psh"), (0x20, "urg")):
            F4[f"{side}_{flag}"] = ((v & bit) > 0).astype(float)
    if "cli_syn" in F4 and "srv_syn" in F4:
        F4["handshake_complete"] = F4["cli_syn"] * F4["srv_syn"] * F4["srv_ack"]
        F4["half_open"] = F4["cli_syn"] * (1 - F4["srv_ack"])
        F4["server_silent"] = (col(f, "SERVER_TCP_FLAGS") == 0).astype(float)
        F4["reset_by_server"] = F4["srv_rst"]
        F4["bilateral_fin"] = F4["cli_fin"] * F4["srv_fin"]
    fam["F4_tcp_semantics"] = F4
    F5 = {}
    for d, r, p in (("in", ("RETRANSMITTED_IN_PKTS", "SrcRetra"), ip_), ("out", ("RETRANSMITTED_OUT_PKTS", "DstRetra"), op_)):
        rv = col(f, *r)
        if rv is not None and p is not None:
            F5[f"retrans_rate_{d}"] = safe_div(rv, p)
    for d, l_, p in (("src", "SrcLoss", ip_), ("dst", "DstLoss", op_)):
        lv = col(f, l_)
        if lv is not None and p is not None:
            F5[f"loss_rate_{d}"] = safe_div(lv, p)
    fam["F5_reliability"] = F5
    F6 = {}
    for d, app, b, p in (("src", "SAppBytes", ib, ip_), ("dst", "DAppBytes", ob, op_)):
        av = col(f, app)
        if av is not None and b is not None and p is not None:
            F6[f"app_bytes_per_pkt_{d}"] = safe_div(av, p)
            F6[f"header_overhead_{d}"] = 1 - safe_div(av, b)
    fam["F6_payload"] = F6
    F7 = {}
    wi, wo = col(f, "TCP_WIN_MAX_IN", "SrcWin"), col(f, "TCP_WIN_MAX_OUT", "DstWin")
    if wi is not None and wo is not None:
        F7["log_win_ratio"] = np.log1p(wo) - np.log1p(wi)
        F7["zero_win_in"], F7["zero_win_out"] = (wi == 0).astype(float), (wo == 0).astype(float)
    fam["F7_window"] = F7
    return {k: pd.DataFrame({n: v.astype(np.float32) for n, v in d.items()}) for k, d in fam.items() if d}


REMOVAL = {"R_ttl": ["MIN_TTL", "MAX_TTL", "sTtl", "dTtl", "sHops", "dHops"],
           "R_window_raw": ["TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "SrcWin", "DstWin"],
           "R_absolute_throughput": ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "SRC_TO_DST_AVG_THROUGHPUT",
                                     "DST_TO_SRC_AVG_THROUGHPUT", "SrcLoad", "DstLoad", "Load", "Rate", "SrcRate", "DstRate"],
           "R_dns_ttl": ["DNS_TTL_ANSWER"]}
REMOVAL = {k: [c for c in v if c in FLOW] for k, v in REMOVAL.items()}
REMOVAL = {k: v for k, v in REMOVAL.items() if v}

started = time.time()
data = {s: load(s) for s in ("train", "val", "calibration")}
derived = {s: derive(data[s]) for s in data}
FAM = {k: list(v.columns) for k, v in derived["train"].items()}
log("rows", {s: len(d) for s, d in data.items()}, "families", {k: len(v) for k, v in FAM.items()}, "removal", REMOVAL,
    round(time.time() - started), "s")
ytr, yva, yca = (data[s]["y"].to_numpy() for s in ("train", "val", "calibration"))
seg_val = pd.factorize(data["val"]["stream"])[0]
n = np.bincount(ytr, minlength=K)
MAX_TRAIN = 3_000_000
rate = np.minimum(1.0, np.maximum(MAX_TRAIN / len(ytr), (MAX_TRAIN / (2 * K)) / np.maximum(n, 1)))
keep = rng.random(len(ytr)) < rate[ytr]
weight = 1.0 / rate[ytr[keep]]
prior = n / n.sum()
# Subset the training rows once (stratified cap; weights restore the full-train class distribution).
data["train"] = data["train"].loc[keep].reset_index(drop=True)
derived["train"] = {k: v.loc[keep].reset_index(drop=True) for k, v in derived["train"].items()}
ytr = ytr[keep]
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


def cvar(y, p, seg, alpha=0.25):
    scores = []
    for s_ in np.unique(seg):
        m = seg == s_
        if m.sum() >= 100 and len(np.unique(y[m])) >= 2:
            scores.append(macro(y[m], p[m])[0])
    if not scores:
        return None
    return float(np.sort(scores)[:max(1, int(np.ceil(alpha * len(scores))))].mean())


def frame(split, families, drop):
    base = [c for c in FLOW if c not in drop]
    parts = [data[split][base]] + [derived[split][f_] for f_ in families]
    return pd.concat(parts, axis=1)


results = {}


def evaluate(name, families, drop=()):
    t0 = time.time()
    Xtr = frame("train", families, drop)
    m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr, weight=weight), num_boost_round=300)
    out = {"families": list(families), "dropped": list(drop), "features": int(Xtr.shape[1])}
    for split, ys in (("val", yva), ("calibration", yca)):
        P = m.predict(frame(split, families, drop))
        rules = {"bayes": P.argmax(1), "balanced": (P / np.maximum(prior, 1e-12)).argmax(1)}
        best = max(rules, key=lambda r: macro(ys, rules[r])[0]) if split == "val" else out["rule"]
        out["rule"] = best
        mm, f1 = macro(ys, rules[best])
        out[split] = {"macro_mcc": mm, "per_class_f1": f1}
        if split == "val":
            out["val_cvar"] = cvar(ys, rules[best], seg_val)
    gain = dict(zip(Xtr.columns, m.feature_importance("gain")))
    out["top_gain"] = sorted(((k, round(float(v), 1)) for k, v in gain.items()), key=lambda kv: -kv[1])[:15]
    out["seconds"] = time.time() - t0
    results[name] = out
    log(name, "val", round(out["val"]["macro_mcc"], 4), "cal", round(out["calibration"]["macro_mcc"], 4), "cvar", out["val_cvar"])
    json.dump({"results": results}, open(os.path.join(OUT, "features.json"), "w"), indent=1, default=float)
    return out


def improves(new, ref, margin=0.002, cvar_tol=0.005):
    ok = (new["val"]["macro_mcc"] >= ref["val"]["macro_mcc"] + margin and
          new["calibration"]["macro_mcc"] >= ref["calibration"]["macro_mcc"] + margin)
    if ok and new["val_cvar"] is not None and ref["val_cvar"] is not None:
        ok = new["val_cvar"] >= ref["val_cvar"] - cvar_tol
    return ok


base = evaluate("BASE_current_features", [])
kept = [f_ for f_ in FAM if improves(evaluate(f"ADD_{f_}", [f_]), base)]
log("families kept after forward screening", kept)
removed = [r for r, cols_ in REMOVAL.items() if improves(evaluate(f"DROP_{r}", [], cols_), base)]
log("raw groups removed after screening", removed)
drop_cols = [c for r in removed for c in REMOVAL[r]]
combo = evaluate("COMBINED", kept, drop_cols) if (kept or drop_cols) else base
final_families = list(kept)
for f_ in list(kept):  # backward elimination: a kept family that adds nothing to the combination is removed
    rest = [g for g in final_families if g != f_]
    without = evaluate(f"COMBINED_minus_{f_}", rest, drop_cols)
    if without["val"]["macro_mcc"] >= combo["val"]["macro_mcc"] - 0.001 and without["calibration"]["macro_mcc"] >= combo["calibration"]["macro_mcc"] - 0.001:
        final_families = rest
        combo = without
# Invariance diagnostics of every derived feature (class-conditional train->validation KS and MI in each split).
diag = {}
for f_, cols_ in FAM.items():
    for c in cols_:
        a, b = derived["train"][f_][c].to_numpy(), derived["val"][f_][c].to_numpy()
        kss = []
        for k in range(K):
            ta, vb = a[ytr == k], b[yva == k]
            if len(ta) > 50 and len(vb) > 50:
                qa = np.quantile(ta, np.linspace(0, 1, 101)); qb = np.quantile(vb, np.linspace(0, 1, 101))
                grid = np.unique(np.r_[qa, qb])
                kss.append(float(np.max(np.abs(np.searchsorted(np.sort(ta), grid, side="right") / len(ta) -
                                                   np.searchsorted(np.sort(vb), grid, side="right") / len(vb)))))
        diag[c] = {"family": f_, "mean_classwise_ks_train_val": float(np.mean(kss)) if kss else None}
report = {"prepared": PREPARED_NAME, "classes": classes, "families": FAM, "removal_groups": REMOVAL, "results": results,
          "kept_after_screening": kept, "removed_raw_groups": removed, "final_families": final_families,
          "final": {k: combo[k] for k in ("val", "calibration", "val_cvar", "features", "families", "dropped")},
          "base": {k: base[k] for k in ("val", "calibration", "val_cvar", "features")}, "invariance": diag,
          "protocol": "train/validation/calibration only; keep iff better on validation AND calibration (+0.002) without CVaR loss"}
json.dump(report, open(os.path.join(OUT, "features.json"), "w"), indent=1, default=float)
for name, r in results.items():
    print(f"{name:40s} val {r['val']['macro_mcc']:.4f} cal {r['calibration']['macro_mcc']:.4f} cvar {r['val_cvar']} feats {r['features']}")
print("FINAL families", final_families, "| removed raw groups", removed)
print("base per-class val", base["val"]["per_class_f1"]); print("final per-class val", combo["val"]["per_class_f1"])
print("base per-class cal", base["calibration"]["per_class_f1"]); print("final per-class cal", combo["calibration"]["per_class_f1"])
log("feature discovery done")

# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME, SEEDS)
# CT-GBDT v5 context branch. Candidate feature sets: flow, flow + causal relational context (the root-cause kernel's
# library: per source / destination / pair windows of source-port novelty, destination-port novelty, peer novelty,
# SYN / RST / empty-response fractions, inter-arrival regularity, burstiness; label-free and computed the same way for
# every split, using only earlier flows of the same segment). PRE-REGISTERED RULE: flow + relational is adopted only if
# it beats flow-only by >= 0.002 macro-MCC on BOTH validation and calibration (no test data); otherwise flow-only.
# The chosen set is then fitted with every seed; the output format equals the CT-GBDT branch used by the post-hoc.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import lightgbm as lgb

OUT = "/data/output/context_gbdt"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260926)
WINDOWS = (10.0, 60.0)
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


def shard_path(split, shard):
    return shard["path"] if os.path.exists(shard["path"]) else os.path.join(root, "prepared", split, os.path.basename(shard["path"]))


def load(split):
    frames = []
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=["numeric", "categories", "label", "hi", "source_order", "source_id",
                                                             "s0", "s1", "d0", "d1", "sport", "dport", "stream"])
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
        f["source_id"] = np.asarray(t["source_id"].to_pylist(), dtype=object)
        f["stream"] = np.asarray(t["stream"].to_pylist(), dtype=object)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


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



started = time.time()
data, rel = {}, {}
for split in ("train", "val", "calibration", "test"):
    f = load(split)
    rel[split] = relational(f)  # every flow of the segment forms the causal context (no label is used to filter it)
    keep_rows = (f["y"] >= 0).to_numpy()
    data[split] = f.loc[keep_rows].reset_index(drop=True)
    rel[split] = rel[split].loc[keep_rows].reset_index(drop=True)
    log("loaded", split, len(data[split]), round(time.time() - started), "s")
REL = list(rel["train"].columns)
ytr = data["train"]["y"].to_numpy()
n = np.bincount(ytr, minlength=K)
prior = n / n.sum()
MAX_TRAIN = 3_000_000  # identical to the root-cause confirmation that motivated this branch
rate = np.minimum(1.0, np.maximum(MAX_TRAIN / len(ytr), (MAX_TRAIN / (2 * K)) / np.maximum(n, 1)))
keep = rng.random(len(ytr)) < rate[ytr]
weight = 1.0 / rate[ytr[keep]]


def matrix(split, sets, rows=None):
    X = pd.concat([data[split][FLOW]] + ([rel[split]] if "relational" in sets else []), axis=1)
    return X if rows is None else X.loc[rows].reset_index(drop=True)


def macro(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    return float(mcc[t > 0].mean())


rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, 1e-12)).argmax(1)}


def fit(sets, seed):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": seed}
    # Fixed 300 rounds (no early stopping): validation is reserved for the feature-set / rule / fusion decisions.
    booster = lgb.train(params, lgb.Dataset(matrix("train", sets, np.flatnonzero(keep)), ytr[keep], weight=weight), num_boost_round=300)
    booster.best_iteration = 300
    return booster


report = {"model": "CT-GBDT v5 (flow vs flow + causal relational context; double held-out rule)", "relational_features": REL,
          "classes": classes, "selection": {}}
fitted = {}
for name, sets in (("flow", ()), ("flow+relational", ("relational",))):
    booster = fit(sets, SEEDS[0])
    res = {}
    for split in ("val", "calibration"):
        P = booster.predict(matrix(split, sets), num_iteration=booster.best_iteration)
        y = data[split]["y"].to_numpy()
        res[split] = {r: macro(y, f(P)) for r, f in rules.items()}
    rule = max(rules, key=lambda r: res["val"][r])
    report["selection"][name] = {"validation": res["val"], "calibration": res["calibration"], "rule_on_val": rule,
                                 "trees": booster.best_iteration}
    fitted[name] = (booster, sets, rule)
    log("candidate", name, {s: {r: round(v, 4) for r, v in d.items()} for s, d in res.items()})
f_, r_ = report["selection"]["flow"], report["selection"]["flow+relational"]
gain_val = r_["validation"][r_["rule_on_val"]] - f_["validation"][f_["rule_on_val"]]
gain_cal = r_["calibration"][r_["rule_on_val"]] - f_["calibration"][f_["rule_on_val"]]
chosen = "flow+relational" if (gain_val >= 0.002 and gain_cal >= 0.002) else "flow"
report.update(chosen_feature_set=chosen, gain_validation=gain_val, gain_calibration=gain_cal,
              rule="adopt flow+relational iff +0.002 on BOTH validation and calibration")
json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
log("chosen", chosen, "gains val/cal", round(gain_val, 4), round(gain_cal, 4))
if chosen == "flow":
    # Rejected: the deployed context branch stays the frozen CT-GBDT v4 (which also chose flow only); nothing to export.
    log("relational context rejected by the double held-out rule; CT-GBDT v4 remains the context branch")
    raise SystemExit(0)
sets = fitted[chosen][1]
Xva, Xte = matrix("val", sets), matrix("test", sets)
yva, yte = data["val"]["y"].to_numpy(), data["test"]["y"].to_numpy()
for seed in SEEDS:
    booster = fitted[chosen][0] if seed == SEEDS[0] else fit(sets, seed)
    pv = booster.predict(Xva, num_iteration=booster.best_iteration)
    pt = booster.predict(Xte, num_iteration=booster.best_iteration)
    np.savez_compressed(os.path.join(OUT, f"ctgbdt_seed{seed}.npz"), val_ids=data["val"]["source_id"].to_numpy().astype(str),
                        val_y=yva.astype(np.int8), val_proba=pv.astype(np.float32), test_ids=data["test"]["source_id"].to_numpy().astype(str),
                        test_y=yte.astype(np.int8), test_proba=pt.astype(np.float32), test_stream=data["test"]["stream"].to_numpy().astype(str),
                        prior=prior)
    report[f"seed{seed}"] = {"trees": booster.best_iteration, "validation_macro_mcc": {r: macro(yva, f(pv)) for r, f in rules.items()}}
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    log("seed", seed, "saved")
log("ctgbdt5 done")

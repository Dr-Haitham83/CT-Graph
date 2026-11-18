# flake8: noqa -- executed after the COMMON prelude and temporal_lib.py (load, relational, key, merged_clock, FLOW, K, classes).
# STRICT GRAPH-FREE ABLATION (major revision, reviewer-style audit of the matched baseline). Everything is fixed to the
# context-branch protocol so that only the input features differ:
#   * LightGBM with the context-branch parameters, all training rows (no class-balanced sampling, no weights), seeds 42-44;
#   * causal smoothing fixed to what the adoption rule decided for the context branch (SMOOTH_KEY: None, "src" or "pair");
#   * decision rule reported under BOTH selections: validation macro-MCC and CVaR_0.25 over validation segments;
#   * REL_MODE: "none" (= context branch without relational features), "seg" (relational features per segment, as in the
#     matched baseline), "merged" (relational features recomputed on ONE merged monitor timeline per split, no resets).
# Validation, calibration and test are all reported, so the source-side decision of the adoption rule can be read next to
# the target-site outcome. Nothing is tuned on test.
import zlib

import lightgbm as lgb

OUT = "/data/output/strict"
os.makedirs(OUT, exist_ok=True)
SEEDS = tuple(globals().get("SEEDS_OVERRIDE", (42, 43, 44)))
TAU, MIX = 10.0, 0.5
REL_MODE = globals().get("REL_MODE", "seg")
EVASION = globals().get("EVASION")  # "rotate:R" / "thin:F" on the TEST attack records, same hash as run_pipeline.apply_evasion
SMOOTH_KEY = globals().get("SMOOTH_KEY")
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


def smooth(f, P):
    """Causal smoothing of the context branch (Eq. smoothing): per segment, keyed on source or on (source, destination)."""
    if SMOOTH_KEY is None:
        return P
    seg = f["stream"].to_numpy().astype(np.int64)
    g = key(f["s0"].to_numpy(), f["s1"].to_numpy(), seg)
    if SMOOTH_KEY == "pair":
        g = key(g, key(f["d0"].to_numpy(), f["d1"].to_numpy(), seg))
    t = f["hi"].to_numpy(np.float64)
    order = np.lexsort((f["source_order"].to_numpy(), t, g))
    ema, has = ema_same_group(g[order], t[order], P[order].astype(np.float64), TAU)
    out = np.empty_like(P, dtype=np.float64)
    out[order] = np.where(has[:, None] > 0, (1 - MIX) * P[order] + MIX * ema, P[order])
    return out


def cm_macro(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, p, s = cm.sum(1), cm.sum(0), cm.sum()
    tp = np.diag(cm); fn, fp = t - tp, p - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    return float(mcc[t > 0].mean())


def cvar(y, pred, seg, alpha=0.25, min_rows=100):
    """CVaR_0.25 over validation segments, with the segment filter of the post-hoc (>= 100 rows, >= 2 classes)."""
    scores = [cm_macro(y[seg == s], pred[seg == s]) for s in np.unique(seg)
              if (seg == s).sum() >= min_rows and len(np.unique(y[seg == s])) >= 2]
    return float(np.sort(scores)[:max(1, int(np.ceil(alpha * len(scores))))].mean()) if scores else -1.0


def evade(f):
    kind, value = EVASION.split(":")
    value = int(value)
    attack = (f["y"].to_numpy() >= 0) & (f["y"].to_numpy() != [c.lower() for c in classes].index("benign"))
    code = np.array([zlib.crc32(str(v).encode()) for v in f["source_id"].astype(str)], dtype=np.int64)
    if kind == "thin":
        keep = ~attack | (code % value == 0)
        return f[keep].reset_index(drop=True)
    f = f.copy()
    f.loc[attack, "s0"] = 0x5EED
    f.loc[attack, "s1"] = 1 + code[attack] % value
    return f


def features(split):
    f = load(split, ids=True)
    if EVASION and split == "test":
        f = evade(f)
        log("evasion", EVASION, "test rows", len(f))
    X = f[FLOW].to_numpy(np.float32)
    names = list(FLOW)
    if REL_MODE != "none":
        g = f
        if REL_MODE == "merged":
            g = f.copy()
            g["hi"], g["stream"], how = merged_clock(f)
            log(split, "merged timeline:", how)
        rel = relational(g)  # every record of the split is causal context
        X = np.concatenate([X, rel.to_numpy(np.float32)], 1)
        names += list(rel.columns)
    return f, X, names


fs, Xs = {}, {}
for split in ("train", "val", "calibration", "test"):
    fs[split], Xs[split], names = features(split)
log("features", {s: Xs[s].shape for s in Xs}, "mode", REL_MODE, "smoothing", SMOOTH_KEY)
y = {s: fs[s]["y"].to_numpy() for s in fs}
lab = {s: y[s] >= 0 for s in fs}
ytr, Xtr = y["train"][lab["train"]], Xs["train"][lab["train"]]
prior = np.bincount(ytr, minlength=K) / len(ytr)
rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, 1e-12)).argmax(1)}
params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255,
          "num_threads": os.cpu_count() or 4, "verbose": -1}
P = {s: [] for s in ("val", "calibration", "test")}
iters = []
for seed in SEEDS:
    b = lgb.train({**params, "seed": seed}, lgb.Dataset(Xtr, ytr), 1000,
                  valid_sets=[lgb.Dataset(Xs["val"][lab["val"]], y["val"][lab["val"]])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    iters.append(int(b.best_iteration))
    for s in P:
        P[s].append(b.predict(Xs[s], num_iteration=b.best_iteration))
    log("seed", seed, "iterations", b.best_iteration)
    if globals().get("SAVE_CONTEXT"):
        # per-seed predictions in the context-branch format read by the post-hoc (smoothing is linear, so the mean of the
        # smoothed seeds equals the smoothed mean)
        os.makedirs(os.path.join(OUT, "context_gbdt"), exist_ok=True)
        np.savez_compressed(os.path.join(OUT, "context_gbdt", f"ctgbdt_seed{seed}.npz"),
                            val_ids=np.asarray(fs["val"]["source_id"].astype(str), dtype=str),
                            val_proba=smooth(fs["val"], P["val"][-1]).astype(np.float32),
                            test_ids=np.asarray(fs["test"]["source_id"].astype(str), dtype=str),
                            test_proba=smooth(fs["test"], P["test"][-1]).astype(np.float32),
                            prior=prior.astype(np.float64))
P = {s: smooth(fs[s], np.mean(v, 0)) for s, v in P.items()}
report = {"evasion": EVASION, "rel_mode": REL_MODE, "smoothing_key": SMOOTH_KEY, "features": len(names), "relational": len(names) - len(FLOW),
          "train_rows": int(lab["train"].sum()), "seeds": list(SEEDS), "iterations": iters,
          "protocol": "context-branch parameters and training rows; smoothing fixed; rule chosen on validation only", "scores": {}}
for r, f in rules.items():
    report["scores"][r] = {s: cm_macro(y[s][lab[s]], f(P[s])[lab[s]]) for s in P}
    report["scores"][r]["val_cvar"] = cvar(y["val"][lab["val"]], f(P["val"])[lab["val"]],
                                           fs["val"]["stream"].to_numpy()[lab["val"]])
report["selected"] = {"by_val_macro_mcc": max(rules, key=lambda r: report["scores"][r]["val"]),
                      "by_val_cvar": max(rules, key=lambda r: report["scores"][r]["val_cvar"])}
for how, r in report["selected"].items():
    report[f"test_{how}"] = report["scores"][r]["test"]
log(json.dumps(report["scores"]), report["selected"])
report["elapsed_s"] = time.time() - started
json.dump(report, open(os.path.join(OUT, "strict.json"), "w"), indent=1, default=float)
log("done")

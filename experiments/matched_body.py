# flake8: noqa -- executed after the COMMON prelude and temporal_lib.py (load, relational, key, merged_clock, FLOW, K, classes).
# MATCHED GRAPH-FREE BASELINE (major revision). The GBDT baselines of the paper see per-flow features only; this kernel gives
# them the same 59 causal relational features as the context branch, and the same causal smoothing
# (p' = 0.5 p + 0.5 EMA_10s of the probabilities of the same source's earlier flows on one merged monitor timeline).
# Models: LightGBM with the context-branch settings and XGBoost with the baseline settings, three seeds each.
# Decision rule (Bayes / prior-balanced) and smoothing on/off are chosen on VALIDATION macro-MCC; test is scored once.
# Output: baselines/<name>_test_predictions.npz in the format of the baseline kernels, so that the post-hoc can pair them
# with CT-Graph over the same (stream, block) and segment units.
import lightgbm as lgb
import xgboost as xgb

OUT = "/data/output/matched"
os.makedirs(os.path.join(OUT, "baselines"), exist_ok=True)
rng = np.random.default_rng(20261117)
MAX_TRAIN = int(globals().get("MAX_TRAIN", 3_000_000))
SEEDS = tuple(globals().get("SEEDS_OVERRIDE", (42, 43, 44)))
TAU, MIX = 10.0, 0.5
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
    tm, _, _ = merged_clock(f)
    g = key(f["s0"].to_numpy(), f["s1"].to_numpy(), np.zeros(len(f), np.int64))
    order = np.lexsort((f["source_order"].to_numpy(), tm, g))
    ema, has = ema_same_group(g[order], tm[order], P[order].astype(np.float64), TAU)
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


def features(split, sample=False):
    f = load(split, ids=True)
    rel = relational(f)  # every record of the split is causal context, labelled or not
    y_ = f["y"].to_numpy()
    rows = np.arange(len(f))
    w_ = None
    if sample:
        lab = np.flatnonzero(y_ >= 0)
        n_ = np.bincount(y_[lab], minlength=K)
        rate = np.minimum(1.0, np.maximum(MAX_TRAIN / len(lab), (MAX_TRAIN / (2 * K)) / np.maximum(n_, 1)))
        rows = lab[rng.random(len(lab)) < rate[y_[lab]]]
        w_ = 1.0 / rate[y_[rows]]
    X = np.concatenate([f[FLOW].to_numpy(np.float32)[rows], rel.to_numpy(np.float32)[rows]], 1)
    return f.iloc[rows].reset_index(drop=True), X, list(FLOW) + list(rel.columns), w_


ftr, Xtr, names_, wtr = features("train", sample=True)
fva, Xva, _, _ = features("val")
fte, Xte, _, _ = features("test")
log("features", Xtr.shape, Xva.shape, Xte.shape, "relational", len(names_) - len(FLOW))
ytr, yva, yte = ftr["y"].to_numpy(), fva["y"].to_numpy(), fte["y"].to_numpy()
prior = np.bincount(ytr, minlength=K, weights=wtr) / wtr.sum()
pick = np.arange(len(ytr))
vmask = yva >= 0
rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, 1e-12)).argmax(1)}
report = {"features": len(names_), "relational": len(names_) - len(FLOW), "train_rows_used": int(len(pick)), "seeds": list(SEEDS),
          "selection": "decision rule and smoothing on/off by validation macro-MCC; test scored once", "models": {}}


def fit_lgbm(seed):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
              "num_threads": os.cpu_count() or 4, "seed": seed}
    b = lgb.train(params, lgb.Dataset(Xtr[pick], ytr[pick], weight=wtr), 1000, valid_sets=[lgb.Dataset(Xva[vmask], yva[vmask])],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
    return b.predict(Xva, num_iteration=b.best_iteration), b.predict(Xte, num_iteration=b.best_iteration), int(b.best_iteration)


def fit_xgb(seed):
    m = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.9, tree_method="hist",
                          objective="multi:softprob", num_class=K, n_jobs=os.cpu_count() or 4, random_state=seed,
                          early_stopping_rounds=40, eval_metric="mlogloss")
    m.fit(Xtr[pick], ytr[pick], sample_weight=wtr, eval_set=[(Xva[vmask], yva[vmask])], verbose=False)
    return m.predict_proba(Xva), m.predict_proba(Xte), int(m.best_iteration)


ref = sorted(glob.glob("/data/input/**/baselines/LightGBM_test_predictions.npz", recursive=True))
novel_ids = None
if ref:
    r_ = np.load(ref[0], allow_pickle=False)
    novel_ids = (r_["source_id"].astype(str), r_["novel"])
for name, fit in (("LightGBM_rel", fit_lgbm), ("XGBoost_rel", fit_xgb)):
    pv, pt, its = [], [], []
    for seed in SEEDS:
        a, b, it = fit(seed)
        pv.append(a); pt.append(b); its.append(it)
        log(name, "seed", seed, "iterations", it, "val", round(cm_macro(yva[vmask], a[vmask].argmax(1)), 4))
    Pv, Pt = np.mean(pv, 0), np.mean(pt, 0)
    variants = {"raw": (Pv, Pt), "smoothed": (smooth(fva, Pv), smooth(fte, Pt))}
    grid = {f"{v}|{r}": cm_macro(yva[vmask], f(P[0])[vmask]) for v, P in variants.items() for r, f in rules.items()}
    best = max(grid, key=grid.get)
    v_, r_ = best.split("|")
    Ptest = variants[v_][1]
    pred = rules[r_](Ptest)
    tk = yte >= 0
    seed_test = [cm_macro(yte[tk], rules[r_](p)[tk]) for p in pt]
    report["models"][name] = {"validation_grid": grid, "chosen": best, "iterations": its, "test_macro_mcc": cm_macro(yte[tk], pred[tk]),
                              "test_macro_mcc_per_seed_unsmoothed": seed_test}
    ids = fte["source_id"].astype(str).to_numpy()
    novel = np.zeros(len(ids), bool)
    if novel_ids is not None:
        o = np.argsort(novel_ids[0]); pos = np.searchsorted(novel_ids[0][o], ids).clip(0, len(o) - 1)
        hit = novel_ids[0][o][pos] == ids
        novel[hit] = novel_ids[1][o][pos][hit]
    ben = [c.lower() for c in classes].index("benign")
    np.savez_compressed(os.path.join(OUT, "baselines", f"{name}_test_predictions.npz"), source_id=ids, label=yte.astype(np.int64),
                        prediction=pred.astype(np.int64), novel=novel, benign_probability=Ptest[:, ben].astype(np.float32))
    log(name, "chosen", best, "test macro-MCC", round(report["models"][name]["test_macro_mcc"], 4))
    json.dump(report, open(os.path.join(OUT, "matched.json"), "w"), indent=1, default=float)
report["elapsed_s"] = time.time() - started
json.dump(report, open(os.path.join(OUT, "matched.json"), "w"), indent=1, default=float)
log("done")

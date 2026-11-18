
# INFERENCE LATENCY of the GBDT baselines and of the CPU part of the CT-Graph hybrid (CPU; the GPU graph branch
# is timed inside its own frozen test pass: results.json test.microbatch_latency_s). Boosters were not persisted, so each
# is re-fitted with its FROZEN hyper-parameters and FROZEN number of trees (no early stopping) on a training sample;
# prediction cost depends on the number and size of the trees, not on the split values. Timed:
#   batch 1         single-flow predict calls (median / p95 over random test flows)
#   micro-batch     MICRO consecutive test flows = the graph branch's causal micro-batch (median / p95)
#   full test       one call over the whole test split (amortised microseconds per flow)
# plus the per-flow cost of the causal smoothing (ToN source key, BS1 pair key) and of the causal relational features
# (CIC context / graph inputs), both amortised over the test split. Labels are only used to fit the boosters.
import xgboost as xgb

OUT = "/data/output/latency"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261013)
started = time.time()
THREADS = os.cpu_count() or 4
SAMPLE = globals().get("SAMPLE", 2_000_000)
ONE_CALLS, MICRO_CALLS = 1000, 300


def matrix(f, rows, rel=None):
    X = np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW])
    return np.concatenate([X, rel[rows]], 1) if rel is not None else X


tr = load("train")
ytr_all = tr["y"].to_numpy()
lab = np.flatnonzero(ytr_all >= 0)
rows_tr = np.sort(rng.choice(lab, min(SAMPLE, len(lab)), replace=False))
te = load("test")
rows_te = np.flatnonzero(te["y"].to_numpy() >= 0)
report = {"cpu_threads": THREADS, "micro_batch": MICRO, "train_rows_used": int(len(rows_tr)), "test_rows": int(len(rows_te)),
          "cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(" :\t") if os.path.exists("/proc/cpuinfo") else None}
rel_tr = rel_te = None
if CT_FEATURES == "flow+REL":
    rel_tr = relational(tr).to_numpy(np.float32)
    t0 = time.perf_counter()
    rel_te = relational(te).to_numpy(np.float32)
    report["relational_features_us_per_flow"] = 1e6 * (time.perf_counter() - t0) / len(te)
Xtr_flow, ytr = matrix(tr, rows_tr), ytr_all[rows_tr]
Xte_flow = matrix(te, rows_te)
Xtr_ct = matrix(tr, rows_tr, rel_tr) if rel_tr is not None else Xtr_flow
Xte_ct = matrix(te, rows_te, rel_te) if rel_te is not None else Xte_flow
log("data", Xtr_flow.shape, Xte_flow.shape, Xte_ct.shape, round(time.time() - started), "s")

base = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "num_threads": THREADS,
        "verbose": -1, "max_bin": 255}
ct_params = dict(base, learning_rate=0.08) if CT_FEATURES == "flow+REL" else dict(base)
models = {}
models["LightGBM"] = [lgb.train(dict(base, seed=42), lgb.Dataset(Xtr_flow, ytr), num_boost_round=LGBM_TREES)]
xp = {"objective": "multi:softprob", "num_class": K, "tree_method": "hist", "max_depth": 8, "eta": 0.1, "subsample": 0.8,
      "colsample_bytree": 0.9, "min_child_weight": 5, "nthread": THREADS, "seed": 42}
models["XGBoost"] = [xgb.train(xp, xgb.QuantileDMatrix(Xtr_flow, ytr, max_bin=256), num_boost_round=XGB_ROUNDS)]
models["CT-GBDT"] = [lgb.train(dict(ct_params, seed=s), lgb.Dataset(Xtr_ct, ytr), num_boost_round=n) for s, n in zip((42, 43, 44), CT_TREES)]
log("fitted", {k: len(v) for k, v in models.items()}, round(time.time() - started), "s")
report["trees"] = {"LightGBM": LGBM_TREES, "XGBoost": XGB_ROUNDS, "CT-GBDT": CT_TREES, "CT-GBDT features": CT_FEATURES}


def predictor(name):
    if name == "XGBoost":
        b = models[name][0]
        return lambda X: b.inplace_predict(X)
    bs = models[name]
    return lambda X: [m.predict(X) for m in bs]


def timing(fn, X):
    fn(X[:MICRO])  # warm-up
    one = []
    for i in rng.choice(len(X), ONE_CALLS, replace=False):
        t0 = time.perf_counter(); fn(X[i:i + 1]); one.append(time.perf_counter() - t0)
    micro = []
    starts = rng.choice(len(X) - MICRO, MICRO_CALLS, replace=False)
    for s in starts:
        t0 = time.perf_counter(); fn(X[s:s + MICRO]); micro.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); fn(X); full = time.perf_counter() - t0
    return {"batch1_ms": {"p50": 1e3 * float(np.median(one)), "p95": 1e3 * float(np.percentile(one, 95))},
            "micro_ms": {"p50": 1e3 * float(np.median(micro)), "p95": 1e3 * float(np.percentile(micro, 95))},
            "micro_us_per_flow": 1e6 * float(np.median(micro)) / MICRO,
            "full_test_s": full, "full_us_per_flow": 1e6 * full / len(X)}


report["timing"] = {}
for name, X in (("LightGBM", Xte_flow), ("XGBoost", Xte_flow), ("CT-GBDT", Xte_ct)):
    report["timing"][name] = timing(predictor(name), X)
    log(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in report["timing"][name].items()}))

# OPTIONAL per-seed cost of the context branch (single-seed ablation of the CT-Graph ensembles).
if globals().get("PER_SEED"):
    for i_, m_ in enumerate(models["CT-GBDT"]):
        report["timing"][f"CT-GBDT seed {i_}"] = timing(lambda X, m=m_: m.predict(X), Xte_ct)
        log(f"CT-GBDT seed {i_}", json.dumps({k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()})
                                               for k, v in report["timing"][f"CT-GBDT seed {i_}"].items()}))
# causal smoothing (frozen: tau 10 s, mix 0.5, merged monitor timeline) of the context probabilities
if SMOOTH_KEY:
    from numba import njit

    @njit(cache=False)
    def ema_same_group(g, t, P, tau):
        n, k = P.shape
        out = np.zeros((n, k)); has = np.zeros(n); S = np.zeros(k); W = 0.0
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

    P = np.mean(predictor("CT-GBDT")(Xte_ct), 0)
    tm, _, _ = merged_clock(te)
    tm = tm[rows_te]
    s0, s1 = te["s0"].to_numpy()[rows_te], te["s1"].to_numpy()[rows_te]
    zero = np.zeros(len(rows_te), np.int64)
    ema_same_group(np.zeros(10, np.int64), np.arange(10.0), np.ones((10, K)), 10.0)  # JIT warm-up
    t0 = time.perf_counter()
    g = key(s0, s1, zero)
    if SMOOTH_KEY == "pair":
        g = key(g, key(te["d0"].to_numpy()[rows_te], te["d1"].to_numpy()[rows_te], zero))
    order = np.lexsort((te["source_order"].to_numpy()[rows_te], tm, g))
    ema, has = ema_same_group(g[order], tm[order], P[order].astype(np.float64), 10.0)
    _ = np.where(has[:, None] > 0, 0.5 * P[order] + 0.5 * ema, P[order])
    report["smoothing_us_per_flow"] = 1e6 * (time.perf_counter() - t0) / len(rows_te)
# equal-weight fusion + decision of one micro-batch
G = rng.random((MICRO, K)); C = rng.random((MICRO, K))
t0 = time.perf_counter()
for _ in range(1000):
    (0.5 * G + 0.5 * C).argmax(1)
report["fusion_us_per_flow"] = 1e6 * (time.perf_counter() - t0) / (1000 * MICRO)
json.dump(report, open(os.path.join(OUT, "latency.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
log("latency benchmark done")

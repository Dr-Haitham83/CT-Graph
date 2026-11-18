
# COLLECTIVE-CLASSIFICATION LAB (train / validation / calibration only; the test split is never opened).
# Stage 1 = LightGBM on flow + relational context. Stage 2 adds, for every flow, the causal window means (60 s, 300 s)
# of the stage-1 class probabilities of the EARLIER flows of the same source, destination and pair (iterative
# classification, Neville & Jensen 2000; Sen et al., AI Magazine 2008). Training-split stage-1 probabilities are
# out-of-fold: two folds of alternating blocks of consecutive flows per segment (availability order; block = min(2000,
# segment size / 10) flows, so short captures and sub-minute floods split too; every class must occur in both folds), each predicted by the model fitted on the other
# fold. Validation / calibration use the stage-1 model fitted on the whole training sample (deployment setting).
# Adoption later requires >= +0.002 macro-MCC over flow + relational on BOTH validation and calibration.
# MERGED = True recomputes everything on one continuous monitor timeline (merged_clock): the boundary-robustness check.
OUT = "/data/output/collective"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260928)
LAB_TRAIN = 600_000
BLOCK = 2000
AGG_WIN = (60.0, 300.0)
started = time.time()
MERGED = globals().get("MERGED", False)


def load_split(split):
    f = load(split)
    if MERGED:
        f["hi"], f["stream"], how = merged_clock(f)
        log("merged timeline:", how)
    return f


relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}


def sample(candidates, y, cap):
    n = np.bincount(y[candidates], minlength=K)
    rate = np.minimum(1.0, np.maximum(cap / len(candidates), (cap / (2 * K)) / np.maximum(n, 1)))
    keep = candidates[rng.random(len(candidates)) < rate[y[candidates]]]
    return keep, 1.0 / rate[y[keep]]


def stage_matrix(f, rel, rows):
    return np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), rel[rows]], 1)


def fit(X, y, w):
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=w), num_boost_round=200)


def predict_rows(booster, f, rel, rows, chunk=400_000):
    out = np.empty((len(rows), K), np.float32)
    for i in range(0, len(rows), chunk):
        out[i:i + chunk] = booster.predict(stage_matrix(f, rel, rows[i:i + chunk]))
    return out


def neighbour(f, P, rows):
    """Causal window means of the OTHER earlier flows' stage-1 probabilities per source / destination / pair."""
    N = len(f)
    st = f["stream"].to_numpy().astype(np.int64)
    order0 = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), st))
    inv0 = np.empty(N, np.int64)
    inv0[order0] = np.arange(N)
    pos0 = inv0[rows]
    st0 = st[order0]
    t0 = f["hi"].to_numpy(np.float64)[order0]
    t0 = t0 - pd.Series(t0).groupby(st0).transform("min").to_numpy()
    src = key(f["s0"].to_numpy()[order0], f["s1"].to_numpy()[order0], st0)
    dst = key(f["d0"].to_numpy()[order0], f["d1"].to_numpy()[order0], st0)
    P0 = P[order0]
    idx = np.arange(N)
    feats, names_ = [], []
    for role, g in (("src", src), ("dst", dst), ("pair", key(src, dst))):
        order = np.lexsort((idx, t0, g))
        gs, ts = g[order], t0[order]
        rank = np.empty(N, np.int64)
        rank[order] = idx
        take = rank[pos0]
        starts = {w: window_starts(gs, ts, w)[take] for w in AGG_WIN}
        for w in AGG_WIN:
            n = take - starts[w] + 1
            feats.append(np.log1p(n - 1).astype(np.float32))
            names_.append(f"nb_{role}_count_{int(w)}s")
        for k in range(K):
            pk = P0[order, k]
            c = np.cumsum(pk, dtype=np.float64)
            own = pk[take]
            for w in AGG_WIN:
                s = starts[w]
                n = take - s + 1
                tot = c[take] - np.where(s > 0, c[np.maximum(s - 1, 0)], 0.0)
                feats.append(np.where(n > 1, (tot - own) / np.maximum(n - 1, 1), np.nan).astype(np.float32))
                names_.append(f"nb_{role}_p_{classes[k]}_{int(w)}s")
    return np.column_stack(feats), names_


def scores(y, P):
    out = {}
    for rule, pred in (("bayes", P.argmax(1)), ("balanced", (P / np.maximum(prior, 1e-12)).argmax(1))):
        cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
        t, pp = cm.sum(1), cm.sum(0)
        tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
        present = t > 0
        out[rule] = {"macro_mcc": float(mcc[present].mean()),
                     "per_class_mcc": {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}}
    return out


# ---- training split: out-of-fold stage-1 probabilities for every flow (labelled or not)
f = load_split("train")
y_all = f["y"].to_numpy()
rel = relational(f).to_numpy(np.float32)
REL = list(relational_names)
lab = np.flatnonzero(y_all >= 0)
counts = np.bincount(y_all[lab], minlength=K)
prior = counts / counts.sum()
stream_ = f["stream"].to_numpy().astype(np.int64)
order_ = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), stream_))
position = np.empty(len(f), np.int64)
position[order_] = np.arange(len(f)) - np.searchsorted(stream_[order_], stream_[order_])  # rank within its segment
seg_size = np.bincount(stream_)[stream_]
block_size = np.clip(np.minimum(BLOCK, seg_size // 10), 1, None)
fold = ((position // block_size) % 2).astype(np.int8)
coverage = np.stack([np.bincount(y_all[lab][fold[lab] == k_], minlength=K) for k_ in (0, 1)])
present = counts > 0
log("fold class coverage", {classes[i]: coverage[:, i].tolist() for i in range(K) if present[i]})
if (coverage[:, present] < 20).any():
    raise SystemExit(f"a class is (almost) absent from one fold: {coverage.tolist()}")
P_tr = np.empty((len(f), K), np.float32)
for k_ in (0, 1):
    rows_fit, w_fit = sample(lab[fold[lab] == k_], y_all, LAB_TRAIN // 2)
    booster = fit(stage_matrix(f, rel, rows_fit), y_all[rows_fit], w_fit)
    other = np.flatnonzero(fold != k_)
    P_tr[other] = predict_rows(booster, f, rel, other)
    log("fold", k_, "fitted on", len(rows_fit), "predicted", len(other), round(time.time() - started), "s")
s_full, w_full = sample(lab, y_all, LAB_TRAIN)
X1_tr = stage_matrix(f, rel, s_full)
y_tr = y_all[s_full]
stage1 = fit(X1_tr, y_tr, w_full)
flow_only = fit(X1_tr[:, :len(FLOW)], y_tr, w_full)
A_tr, NB = neighbour(f, P_tr, s_full)
own_tr = P_tr[s_full]
report = {"prepared": PREPARED_NAME, "classes": classes, "protocol": "train/validation/calibration only; test never opened",
          "oof": "two folds of alternating blocks of min(2000, segment/10) flows per segment", "fold_class_coverage": coverage.tolist(), "neighbour_features": NB,
          "oof_train_macro_mcc": scores(y_all[lab], P_tr[lab])["bayes"]["macro_mcc"], "configs": {}}
del f, rel, P_tr
log("train stage done", X1_tr.shape, A_tr.shape, round(time.time() - started), "s")
data = {}
for split in ("val", "calibration"):
    f = load_split(split)
    y = f["y"].to_numpy()
    rel = relational(f).to_numpy(np.float32)
    rows = np.flatnonzero(y >= 0)
    P = predict_rows(stage1, f, rel, np.arange(len(f)))
    A, _ = neighbour(f, P, rows)
    data[split] = {"X1": stage_matrix(f, rel, rows), "own": P[rows], "A": A, "y": y[rows]}
    del f, rel, P
    log("built", split, len(rows), round(time.time() - started), "s")
stage2 = {"stage2_neighbours": lambda d: np.concatenate([d["X1"], d["A"]], 1),
          "stage2_own+neighbours": lambda d: np.concatenate([d["X1"], d["own"], d["A"]], 1)}
train_inputs = {"X1": X1_tr, "own": own_tr, "A": A_tr}
models = {"flow": (flow_only, lambda d: d["X1"][:, :len(FLOW)]), "flow+REL (stage 1)": (stage1, lambda d: d["X1"])}
for name, make in stage2.items():
    models[name] = (fit(make(train_inputs), y_tr, w_full), make)
    log("fitted", name)
for name, (booster, make) in models.items():
    res = {s: scores(data[s]["y"], booster.predict(make(data[s]))) for s in ("val", "calibration")}
    rule = max(("bayes", "balanced"), key=lambda r: res["val"][r]["macro_mcc"])
    report["configs"][name] = {"rule": rule, "val": res["val"][rule], "calibration": res["calibration"][rule]}
    log("config", name, "val", round(res["val"][rule]["macro_mcc"], 4), "cal", round(res["calibration"][rule]["macro_mcc"], 4), rule)
    json.dump(report, open(os.path.join(OUT, "collective.json"), "w"), indent=1)
ref = report["configs"]["flow+REL (stage 1)"]
for name in stage2:
    r = report["configs"][name]
    r["gain_vs_stage1"] = (r["val"]["macro_mcc"] - ref["val"]["macro_mcc"], r["calibration"]["macro_mcc"] - ref["calibration"]["macro_mcc"])
json.dump(report, open(os.path.join(OUT, "collective.json"), "w"), indent=1)
for name, r in report["configs"].items():
    print(name, round(r["val"]["macro_mcc"], 4), round(r["calibration"]["macro_mcc"], 4), r.get("gain_vs_stage1"))
    print("   val", r["val"]["per_class_mcc"])
    print("   cal", r["calibration"]["per_class_mcc"])
log("collective lab done")

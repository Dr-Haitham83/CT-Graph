
# CT-GBDT v9 = CT-GBDT v7 + HETEROGENEOUS candidates (LightGBM + ExtraTrees, unweighted mean; Geurts et al., Machine
# Learning 2006). Lab 2 found LightGBM + ExtraTrees on flow + relational context above flow + relational on BOTH CIC
# validation and calibration. ExtraTrees: 100 trees, sqrt features, min 20 rows per leaf, depth <= 30, fitted on a
# 1M-row subsample of the same weighted training sample.
# CT-GBDT v7 context branch with a COLLECTIVE (two-stage) candidate. Stage 1 = LightGBM on flow + relational context.
# Stage 2 adds the flow's own stage-1 probabilities and the causal 60 s / 300 s window means of the stage-1
# probabilities of the earlier flows of the same source, destination and pair (iterative classification; Sen et al.,
# AI Magazine 2008). Training-split stage-1 probabilities are out-of-fold (two folds of alternating blocks of
# min(2000, segment/10) consecutive flows per segment; every class must occur in both folds).
# PRE-REGISTERED RULE (as ctgbdt6; amended 2026-09-27 before any ToN / CIC ctgbdt7 run to carry both stage-2 variants of
# the collective lab): candidates flow, flow + relational, collective (own + neighbours), collective_neighbours (neighbours
# only); among those beating flow-only by
# >= 0.002 macro-MCC on BOTH validation and calibration (val-chosen decision rule), pick the maximiser of
# min(gain_val, gain_cal); if none qualifies, the frozen CT-GBDT v4 stays. Output format = the CT-GBDT branch files.
OUT = "/data/output/context_gbdt"
os.makedirs(OUT, exist_ok=True)
from sklearn.ensemble import ExtraTreesClassifier
rng = np.random.default_rng(20260929)
ET_ROWS = 1_000_000
MAX_TRAIN = 3_000_000
BLOCK = 2000
AGG_WIN = (60.0, 300.0)
started = time.time()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def sample(candidates, y, cap):
    n = np.bincount(y[candidates], minlength=K)
    rate = np.minimum(1.0, np.maximum(cap / len(candidates), (cap / (2 * K)) / np.maximum(n, 1)))
    keep = candidates[rng.random(len(candidates)) < rate[y[candidates]]]
    return keep, 1.0 / rate[y[keep]]


def stage_matrix(f, rel, rows):
    return np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), rel[rows]], 1)


def fit(X, y, w, seed):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.08, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": seed}
    return lgb.train(params, lgb.Dataset(X, y, weight=w), num_boost_round=300)


def predict(booster, X, chunk=400_000):
    return np.concatenate([booster.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)]).astype(np.float32)


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
    feats = []
    for g in (src, dst, key(src, dst)):
        order = np.lexsort((idx, t0, g))
        gs, ts = g[order], t0[order]
        rank = np.empty(N, np.int64)
        rank[order] = idx
        take = rank[pos0]
        starts = {w: window_starts(gs, ts, w)[take] for w in AGG_WIN}
        for w in AGG_WIN:
            feats.append(np.log1p(take - starts[w]).astype(np.float32))
        for k in range(K):
            pk = P0[order, k]
            c = np.cumsum(pk, dtype=np.float64)
            own = pk[take]
            for w in AGG_WIN:
                s = starts[w]
                n = take - s + 1
                tot = c[take] - np.where(s > 0, c[np.maximum(s - 1, 0)], 0.0)
                feats.append(np.where(n > 1, (tot - own) / np.maximum(n - 1, 1), np.nan).astype(np.float32))
    return np.column_stack(feats)


def macro(y, p):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    return float(mcc[t > 0].mean())


# ---- training split: out-of-fold stage-1 probabilities for every flow
f = load("train")
y_all = f["y"].to_numpy()
rel = relational(f).to_numpy(np.float32)
lab = np.flatnonzero(y_all >= 0)
counts = np.bincount(y_all[lab], minlength=K)
prior = counts / counts.sum()
rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, 1e-12)).argmax(1)}
stream_ = f["stream"].to_numpy().astype(np.int64)
order_ = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), stream_))
position = np.empty(len(f), np.int64)
position[order_] = np.arange(len(f)) - np.searchsorted(stream_[order_], stream_[order_])
block_size = np.clip(np.minimum(BLOCK, np.bincount(stream_)[stream_] // 10), 1, None)
fold = ((position // block_size) % 2).astype(np.int8)
coverage = np.stack([np.bincount(y_all[lab][fold[lab] == k_], minlength=K) for k_ in (0, 1)])
present = counts > 0
if (coverage[:, present] < 20).any():
    raise SystemExit(f"a class is (almost) absent from one fold: {coverage.tolist()}")
P_tr = np.empty((len(f), K), np.float32)
for k_ in (0, 1):
    rows_fit, w_fit = sample(lab[fold[lab] == k_], y_all, MAX_TRAIN // 2)
    booster = fit(stage_matrix(f, rel, rows_fit), y_all[rows_fit], w_fit, SEEDS[0])
    other = np.flatnonzero(fold != k_)
    P_tr[other] = predict_rows(booster, f, rel, other)
    log("oof fold", k_, len(rows_fit), "->", len(other), round(time.time() - started), "s")
    del booster
s_full, w_full = sample(lab, y_all, MAX_TRAIN)
X1_tr = stage_matrix(f, rel, s_full)
y_tr = y_all[s_full]
stage1 = fit(X1_tr, y_tr, w_full, SEEDS[0])
Z_tr = np.concatenate([X1_tr, P_tr[s_full], neighbour(f, P_tr, s_full)], 1)
del f, rel, P_tr
log("train stage built", Z_tr.shape, round(time.time() - started), "s")


def build(split, ids=False):
    f = load(split, ids=ids)
    y = f["y"].to_numpy()
    rel = relational(f).to_numpy(np.float32)
    rows = np.flatnonzero(y >= 0)
    P = predict_rows(stage1, f, rel, np.arange(len(f)))
    Z = np.concatenate([stage_matrix(f, rel, rows), P[rows], neighbour(f, P, rows)], 1)
    extra = {"stream": f["stream"].to_numpy()[rows]}
    if ids:
        extra["ids"] = f["source_id"].to_numpy()[rows].astype(str)
    del f, rel, P
    log("built", split, Z.shape, round(time.time() - started), "s")
    return Z, y[rows], extra


Zva, yva, eva = build("val", ids=True)
Zca, yca, _ = build("calibration")
nf, n1 = len(FLOW), X1_tr.shape[1]
CANDIDATES = {"flow": np.arange(nf), "flow+relational": np.arange(n1), "collective": np.arange(Z_tr.shape[1]),
              "flow+relational+ET": np.arange(n1), "collective+ET": np.arange(Z_tr.shape[1])}
et_rows = np.sort(rng.choice(len(y_tr), min(ET_ROWS, len(y_tr)), replace=False))


def fit_et(sl, seed):
    et = ExtraTreesClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=20, max_depth=30, n_jobs=-1, random_state=seed)
    et.fit(np.nan_to_num(Z_tr[et_rows][:, sl], nan=-1e9, posinf=1e9, neginf=-1e9), y_tr[et_rows], sample_weight=w_full[et_rows])
    return et


def et_predict(et, Z, sl, chunk=300_000):
    out = np.zeros((len(Z), K), np.float32)
    for i in range(0, len(Z), chunk):
        out[i:i + chunk][:, et.classes_] = et.predict_proba(np.nan_to_num(Z[i:i + chunk][:, sl], nan=-1e9, posinf=1e9, neginf=-1e9))
    return out


class Het:
    """Unweighted mean of a LightGBM booster and an ExtraTrees model on the same columns."""
    def __init__(self, booster, et, sl):
        self.booster, self.et, self.sl = booster, et, sl

    def predict(self, Z):
        return (predict(self.booster, Z[:, self.sl]) + et_predict(self.et, Z, self.sl)) / 2
report = {"model": "CT-GBDT v9 (v7 candidates + LightGBM/ExtraTrees heterogeneous means; double held-out maximin rule)",
          "classes": classes, "fold_class_coverage": coverage.tolist(), "selection": {}}
fitted = {}
for name, sl in CANDIDATES.items():
    if name == "flow+relational":
        booster = stage1
    elif name.endswith("+ET"):
        base = fitted["flow+relational" if name.startswith("flow+relational") else "collective"]
        booster = Het(base, fit_et(sl, SEEDS[0]), sl)
    else:
        booster = fit(Z_tr[:, sl], y_tr, w_full, SEEDS[0])
    res = {}
    for s, Z, y in (("val", Zva, yva), ("calibration", Zca, yca)):
        P = booster.predict(Z) if isinstance(booster, Het) else predict(booster, Z[:, sl])
        res[s] = {r: macro(y, f_(P)) for r, f_ in rules.items()}
    rule = max(rules, key=lambda r: res["val"][r])
    report["selection"][name] = {"validation": res["val"], "calibration": res["calibration"], "rule_on_val": rule}
    fitted[name] = booster
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    log("candidate", name, {s: {r: round(v, 4) for r, v in d.items()} for s, d in res.items()})
f0 = report["selection"]["flow"]
qualified = {}
for name, r_ in report["selection"].items():
    if name == "flow":
        continue
    gv = r_["validation"][r_["rule_on_val"]] - f0["validation"][f0["rule_on_val"]]
    gc = r_["calibration"][r_["rule_on_val"]] - f0["calibration"][f0["rule_on_val"]]
    r_["gain_validation"], r_["gain_calibration"] = gv, gc
    if gv >= 0.002 and gc >= 0.002:
        qualified[name] = min(gv, gc)
chosen = max(qualified, key=qualified.get) if qualified else "flow"
report.update(chosen_feature_set=chosen, rule="maximin of (gain_val, gain_cal) over candidates with both >= 0.002 vs flow")
json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
log("chosen", chosen, qualified)
if chosen == "flow":
    log("no candidate passed the double held-out rule; CT-GBDT v4 remains the context branch")
    raise SystemExit(0)
sl = CANDIDATES[chosen]
if chosen.endswith("+ET"):
    boosters = [fitted[chosen]] + [Het(fit(Z_tr[:, sl], y_tr, w_full, seed), fit_et(sl, seed), sl) for seed in SEEDS[1:]]
else:
    boosters = [fitted[chosen]] + [fit(Z_tr[:, sl], y_tr, w_full, seed) for seed in SEEDS[1:]]
del Z_tr, Zca
Zte, yte, ete = build("test", ids=True)
stream_names = np.asarray(sorted(STREAMS, key=STREAMS.get), dtype=str)
for seed, booster in zip(SEEDS, boosters):
    pv, pt = (booster.predict(Zva), booster.predict(Zte)) if isinstance(booster, Het) else (predict(booster, Zva[:, sl]), predict(booster, Zte[:, sl]))
    np.savez_compressed(os.path.join(OUT, f"ctgbdt_seed{seed}.npz"), val_ids=eva["ids"], val_y=yva.astype(np.int8), val_proba=pv,
                        test_ids=ete["ids"], test_y=yte.astype(np.int8), test_proba=pt,
                        test_stream=stream_names[ete["stream"]], prior=prior)
    report[f"seed{seed}"] = {"validation_macro_mcc": {r: macro(yva, f_(pv)) for r, f_ in rules.items()}}
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    log("seed", seed, "saved")
log("ctgbdt9 done")

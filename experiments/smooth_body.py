
# CAUSAL TEMPORAL SMOOTHING of the context branch (ToN). Lab 4 found p' = 0.5 p + 0.5 EMA_tau=10s(probabilities of the
# same source's EARLIER flows) above the unsmoothed flow branch on BOTH validation and calibration.
# PRE-REGISTERED ADOPTION: (a) the check is repeated with a lab-size flow model on validation / calibration in two
# variants: segmented (source key within its segment, as in lab 4) and MERGED (one monitor timeline: source key without
# segment, global time); both must gain >= 0.002 on BOTH splits. (b) If they do, the frozen CT-GBDT v4 validation and
# test probabilities are smoothed with the MERGED variant and written in the CT-GBDT branch format for the post-hoc
# (equal-weight fusion unchanged). Labels are used only for the validation / calibration check; test labels never.
OUT = "/data/output/context_gbdt"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261008)
LAB_TRAIN = 600_000
TAU = 10.0
MIX = 0.5
KEY = globals().get("KEY", "src")  # "src" (ToN, lab 4) or "pair" (5G BS1, lab 5)
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


def meta(f, rows):
    tm, _, _ = merged_clock(f)
    return {"s0": f["s0"].to_numpy()[rows], "s1": f["s1"].to_numpy()[rows], "d0": f["d0"].to_numpy()[rows],
            "d1": f["d1"].to_numpy()[rows], "seg": f["stream"].to_numpy().astype(np.int64)[rows],
            "t": f["hi"].to_numpy(np.float64)[rows], "t_merged": tm[rows], "so": f["source_order"].to_numpy()[rows]}


def smooth(m, P, merged):
    seg = np.zeros_like(m["seg"]) if merged else m["seg"]
    t = m["t_merged"] if merged else m["t"]
    g = key(m["s0"], m["s1"], seg)
    if KEY == "pair":
        g = key(g, key(m["d0"], m["d1"], seg))
    order = np.lexsort((m["so"], t, g))
    ema, has = ema_same_group(g[order], t[order], P[order].astype(np.float64), TAU)
    out = np.empty_like(P, dtype=np.float64)
    out[order] = np.where(has[:, None] > 0, (1 - MIX) * P[order] + MIX * ema, P[order])
    return out


def macro(y, P, rule, prior):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}


# FROZEN_ADOPTED: the smoothing was already adopted (ton-smooth787-full); a refit of the context branch (e.g. on cleaned
# labels) only re-applies the frozen smoothing to its probabilities.
if globals().get("FROZEN_ADOPTED"):
    report = {"tau_s": TAU, "mix": MIX, "key": KEY, "check": "frozen: adopted by ton-smooth787-full", "adopted": True}
else:
    # ---- (a) check with a lab-size flow model
    f = load("train")
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    n = np.bincount(y[rows], minlength=K)
    prior = n / n.sum()
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(rows), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rows[rng.random(len(rows)) < rate[y[rows]]]
    Xtr = np.column_stack([f[c].to_numpy(np.float32)[keep] for c in FLOW])
    booster = lgb.train({"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
                         "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
                         "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7},
                        lgb.Dataset(Xtr, y[keep], weight=1.0 / rate[y[keep]]), num_boost_round=200)
    del f, Xtr
    report = {"tau_s": TAU, "mix": MIX, "key": KEY, "check": {}}
    P_split, M_split, Y_split = {}, {}, {}
    for split in ("val", "calibration"):
        f = load(split)
        y = f["y"].to_numpy()
        r = np.flatnonzero(y >= 0)
        X = np.column_stack([f[c].to_numpy(np.float32)[r] for c in FLOW])
        P_split[split] = np.concatenate([booster.predict(X[i:i + 400_000]) for i in range(0, len(X), 400_000)])
        M_split[split], Y_split[split] = meta(f, r), y[r]
        del f, X
    rule = max(("bayes", "balanced"), key=lambda r_: macro(Y_split["val"], P_split["val"], r_, prior)[0])
    passes = {}
    for variant, merged in (("segmented", False), ("merged", True)):
        res = {}
        for split in ("val", "calibration"):
            m0, _ = macro(Y_split[split], P_split[split], rule, prior)
            m1, pc1 = macro(Y_split[split], smooth(M_split[split], P_split[split], merged), rule, prior)
            res[split] = {"base": m0, "smoothed": m1, "gain": m1 - m0, "per_class": pc1}
        passes[variant] = bool(min(res["val"]["gain"], res["calibration"]["gain"]) >= 0.002)
        report["check"][variant] = {**res, "passes": passes[variant]}
        log(variant, {s: (round(v["base"], 4), round(v["smoothed"], 4)) for s, v in res.items()}, "PASS" if passes[variant] else "-")
    adopt = passes["segmented"] and passes["merged"]
    report["adopted"] = adopt
    json.dump(report, open(os.path.join(OUT, "smooth.json"), "w"), indent=1, default=float)
    if not adopt:
        log("smoothing NOT adopted (pre-registered robustness condition failed)")
        raise SystemExit(0)


# ---- (b) smooth the frozen CT-GBDT v4 branch (merged variant) and write it in the branch format
src_files = sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))
log("CT-GBDT v4 files:", src_files)
metas = {}
for split in ("val", "test"):
    f = load(split, ids=True)
    y = f["y"].to_numpy()
    r = np.flatnonzero(y >= 0)
    ids = f["source_id"].to_numpy()[r].astype(str)
    order = np.argsort(ids)
    metas[split] = (ids[order], {k: v[order] for k, v in meta(f, r).items()})
    del f


def aligned_meta(split, ids):
    ref_ids, m = metas[split]
    pos = np.searchsorted(ref_ids, ids)
    assert np.all(pos < len(ref_ids)) and np.array_equal(ref_ids[pos], ids), f"{split} ids do not match the prepared split"
    return {k: v[pos] for k, v in m.items()}


for path in src_files:
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    for split, pk, ik in (("val", "val_proba", "val_ids"), ("test", "test_proba", "test_ids")):
        m = aligned_meta(split, z[ik].astype(str))
        out[pk] = smooth(m, z[pk].astype(np.float64), merged=True).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, os.path.basename(path)), **out)
    log("smoothed", os.path.basename(path))
report["smoothed_files"] = [os.path.basename(p) for p in src_files]
json.dump(report, open(os.path.join(OUT, "smooth.json"), "w"), indent=1, default=float)
log("smoothing done")

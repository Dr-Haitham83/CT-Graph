
# LAB 5 (train / validation / calibration only; the test split is never opened).
#  SMOOTH-KEYS  causal temporal smoothing p' = 0.5 p + 0.5 EMA_tau=10s over the earlier flows sharing a KEY, for keys
#               src (lab-4 replicate), dst, pair, and src+dst (mean of the two EMAs); each on the segmented and on the
#               merged monitor timeline. No parameter is tuned (tau and mix fixed from lab 4).
#  HIER         hierarchical classification: P(class) = P(benign) for Benign, P(attack) * P(type | attack) for attacks;
#               binary model on all rows, attack-type model on attack rows only (same features, same LightGBM settings).
# Every variant is compared with its unsmoothed / flat model on validation AND calibration (>= +0.002 on both).
OUT = "/data/output/lab5"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261009)
LAB_TRAIN = 600_000
TAU, MIX = 10.0, 0.5
started = time.time()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


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


def build(split, sample=None):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    flow = np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW])
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    d = {"y": y[rows], "flow": flow, "flow+REL": np.concatenate([flow, rel], 1)}
    if sample is None:
        hi = f["hi"].to_numpy(np.float64)
        hm, _, how = merged_clock(f)
        d.update(s0=f["s0"].to_numpy()[rows], s1=f["s1"].to_numpy()[rows], d0=f["d0"].to_numpy()[rows], d1=f["d1"].to_numpy()[rows],
                 seg=f["stream"].to_numpy().astype(np.int64)[rows], t=hi[rows], t_merged=hm[rows], so=f["source_order"].to_numpy()[rows])
        log("merged clock:", how)
    del f
    log("built", split, len(rows), round(time.time() - started), "s")
    return d


def train_sample(y):
    n = np.bincount(y, minlength=K)
    train_sample.counts = n
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(y), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


STATE_CODES = []
TR = build("train", train_sample)
wtr = train_sample.weight
prior = train_sample.counts / train_sample.counts.sum()
EV = {"val": build("val"), "calibration": build("calibration")}
BENIGN = classes.index("Benign") if "Benign" in classes else int(np.argmax(prior))


def params(k):
    return {"objective": "multiclass" if k > 2 else "binary", **({"num_class": k} if k > 2 else {}), "learning_rate": 0.1,
            "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1,
            "lambda_l2": 1.0, "max_bin": 127, "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}


def macro(y, P, rule):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}


def predict(b, X, chunk=400_000):
    return np.concatenate([b.predict(X[i:i + chunk]) for i in range(0, len(X), chunk)])


def smooth(d, P, keyname, merged):
    seg = np.zeros_like(d["seg"]) if merged else d["seg"]
    t = d["t_merged"] if merged else d["t"]
    keys = {"src": [key(d["s0"], d["s1"], seg)], "dst": [key(d["d0"], d["d1"], seg)],
            "pair": [key(key(d["s0"], d["s1"], seg), key(d["d0"], d["d1"], seg))],
            "src+dst": [key(d["s0"], d["s1"], seg), key(d["d0"], d["d1"], seg)]}[keyname]
    emas, hass = [], []
    for g in keys:
        order = np.lexsort((d["so"], t, g))
        e, h = ema_same_group(g[order], t[order], P[order].astype(np.float64), TAU)
        ee, hh = np.empty_like(e), np.empty_like(h)
        ee[order], hh[order] = e, h
        emas.append(ee)
        hass.append(hh)
    num = sum(e * h[:, None] for e, h in zip(emas, hass))
    cnt = sum(hass)
    ema = np.divide(num, np.maximum(cnt, 1)[:, None])
    return np.where(cnt[:, None] > 0, (1 - MIX) * P + MIX * ema, P)


report = {"prepared": PREPARED_NAME, "tau_s": TAU, "mix": MIX, "protocol": "train/validation/calibration only; test never opened",
          "views": {}}
for view in ("flow", "flow+REL"):
    flat = lgb.train(params(K), lgb.Dataset(TR[view], TR["y"], weight=wtr), num_boost_round=200)
    P = {s: predict(flat, EV[s][view]) for s in EV}
    rule = max(("bayes", "balanced"), key=lambda r: macro(EV["val"]["y"], P["val"], r)[0])
    base = {s: macro(EV[s]["y"], P[s], rule)[0] for s in EV}
    out = {"rule": rule, "flat": base, "smooth": {}}
    for keyname in ("src", "dst", "pair", "src+dst"):
        for merged in (False, True):
            res = {s: macro(EV[s]["y"], smooth(EV[s], P[s], keyname, merged), rule) for s in EV}
            gains = {s: res[s][0] - base[s] for s in EV}
            name = f"{keyname}{' [merged]' if merged else ''}"
            out["smooth"][name] = {"val": res["val"][0], "calibration": res["calibration"][0], "gain_val": gains["val"],
                                   "gain_cal": gains["calibration"], "passes": bool(min(gains.values()) >= 0.002),
                                   "cal_per_class": res["calibration"][1]}
            log(view, "smooth", name, {s: round(g, 4) for s, g in gains.items()})
    # hierarchical: benign vs attack, then attack type on attack rows
    yb = (TR["y"] != BENIGN).astype(int)
    binm = lgb.train(params(2), lgb.Dataset(TR[view], yb, weight=wtr), num_boost_round=200)
    att = TR["y"] != BENIGN
    att_classes = [k for k in range(K) if k != BENIGN and (TR["y"][att] == k).any()]
    remap = {k: i for i, k in enumerate(att_classes)}
    ya = np.array([remap[v] for v in TR["y"][att]])
    typem = lgb.train(params(len(att_classes)), lgb.Dataset(TR[view][att], ya, weight=wtr[att]), num_boost_round=200)
    hier = {}
    for s in EV:
        pa = predict(binm, EV[s][view])
        pt = predict(typem, EV[s][view])
        pt = pt if pt.ndim == 2 else np.column_stack([1 - pt, pt])
        H = np.zeros((len(pa), K))
        H[:, BENIGN] = 1 - pa
        for k, i in remap.items():
            H[:, k] = pa * pt[:, i]
        hier[s] = H
    hrule = max(("bayes", "balanced"), key=lambda r: macro(EV["val"]["y"], hier["val"], r)[0])
    hres = {s: macro(EV[s]["y"], hier[s], hrule) for s in EV}
    out["hierarchical"] = {"rule": hrule, "val": hres["val"][0], "calibration": hres["calibration"][0],
                           "gain_val": hres["val"][0] - base["val"], "gain_cal": hres["calibration"][0] - base["calibration"],
                           "cal_per_class": hres["calibration"][1]}
    out["hierarchical"]["passes"] = bool(min(out["hierarchical"]["gain_val"], out["hierarchical"]["gain_cal"]) >= 0.002)
    report["views"][view] = out
    json.dump(report, open(os.path.join(OUT, "lab5.json"), "w"), indent=1, default=float)
for view, o in report["views"].items():
    print(f"== {view}: flat val {o['flat']['val']:.4f} cal {o['flat']['calibration']:.4f} (rule {o['rule']})")
    for name, r in o["smooth"].items():
        print(f"   smooth {name:18s} val {r['val']:.4f} ({r['gain_val']:+.4f}) cal {r['calibration']:.4f} ({r['gain_cal']:+.4f}) {'PASS' if r['passes'] else '-'}")
    h = o["hierarchical"]
    print(f"   hierarchical        val {h['val']:.4f} ({h['gain_val']:+.4f}) cal {h['calibration']:.4f} ({h['gain_cal']:+.4f}) {'PASS' if h['passes'] else '-'}")
log("lab5 done")

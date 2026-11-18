
# LAB 4 (train / validation / calibration only; the test split is never opened).
#  REFIT  Does adding more CAMPAIGNS to training help the next unseen campaign? Same total training size (cap), so any
#         gain comes from campaign diversity, not sample size:  train -> calibration  vs  train + validation -> calibration,
#         and  train -> validation  vs  train + calibration -> validation. Counts only if BOTH directions gain >= 0.002.
#         If it does, the final context branch may be refitted on train + validation + calibration with its configuration
#         and decision rule frozen beforehand ("refit after selection").
#  SMOOTH Causal temporal smoothing of the context-branch probabilities: p' = 0.5 p + 0.5 EMA(tau = 10 s) of the
#         probabilities of the SAME source's earlier flows (non-learned counterpart of the collective stage).
OUT = "/data/output/lab4"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261007)
CAP = 600_000
TAU = 10.0
started = time.time()
self_check()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


@njit(cache=False)
def ema_same_group(g, t, P, tau):
    """For rows sorted by (group, time): EMA of the probabilities of EARLIER rows of the same group (0 weight if none)."""
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


PRE = 2_000_000  # training rows kept (class-floored, weighted) so that very large training splits fit in memory


def build(split):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    base = np.ones(len(rows))
    if split == "train" and len(rows) > PRE:
        n = np.bincount(y[rows], minlength=K)
        rate = np.minimum(1.0, np.maximum(PRE / len(rows), (PRE / (2 * K)) / np.maximum(n, 1)))
        keep = rng.random(len(rows)) < rate[y[rows]]
        base = 1.0 / rate[y[rows][keep]]
        rows = rows[keep]
    rel_all = relational(f)
    flow = np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW])
    rel = rel_all.iloc[rows].to_numpy(np.float32)
    del rel_all
    st = f["stream"].to_numpy().astype(np.int64)[rows]
    t = f["hi"].to_numpy(np.float64)[rows]
    so = f["source_order"].to_numpy()[rows]
    src = key(f["s0"].to_numpy()[rows], f["s1"].to_numpy()[rows], st)
    del f
    log("built", split, len(rows), round(time.time() - started), "s")
    return {"y": y[rows], "base": base, "flow": flow, "rel": rel, "src": src, "t": t, "so": so, "seg": st}


D = {s: build(s) for s in ("train", "val", "calibration")}
PARAMS = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}
n_tr = np.bincount(D["train"]["y"], weights=D["train"]["base"], minlength=K)
prior = n_tr / n_tr.sum()


def sample(parts):
    """Class-floored sample of about CAP rows from the union of the splits (rows drawn with probability proportional to
    their base weight, so the union's class-conditional distribution is kept); weights restore the union's prior."""
    y = np.concatenate([D[p]["y"] for p in parts])
    b = np.concatenate([D[p]["base"] for p in parts])
    N = np.bincount(y, weights=b, minlength=K)
    rows, w = [], []
    for c in range(K):
        idx = np.flatnonzero(y == c)
        if not len(idx):
            continue
        m = int(min(len(idx), max(CAP * N[c] / N.sum(), CAP / (2 * K))))
        keys = np.log(rng.random(len(idx))) / b[idx]  # Efraimidis-Spirakis weighted sampling without replacement
        pick = idx[np.argpartition(-keys, m - 1)[:m]] if m < len(idx) else idx
        rows.append(pick)
        w.append(np.full(m, N[c] / m))
    return np.concatenate(rows), np.concatenate(w)


def view_of(d, view, idx=None):
    parts = [d["flow"]] + ([d["rel"]] if view == "flow+REL" else [])
    return np.concatenate([x if idx is None else x[idx] for x in parts], 1) if len(parts) > 1 else (parts[0] if idx is None else parts[0][idx])


def matrix(parts, view, rows):
    offs = np.cumsum([0] + [len(D[p]["y"]) for p in parts])
    out = []
    for i, p in enumerate(parts):
        local = rows[(rows >= offs[i]) & (rows < offs[i + 1])] - offs[i]
        out.append(view_of(D[p], view, local))
    return np.concatenate(out)


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


def smooth(split, P):
    d = D[split]
    order = np.lexsort((d["so"], d["t"], d["src"]))
    ema, has = ema_same_group(d["src"][order], d["t"][order], P[order].astype(np.float64), TAU)
    out = P.copy()
    mix = np.where(has[:, None] > 0, 0.5 * P[order] + 0.5 * ema, P[order])
    out[order] = mix
    return out


report = {"prepared": PREPARED_NAME, "cap": CAP, "tau_s": TAU, "protocol": "train/validation/calibration only; test never opened",
          "refit": {}, "smooth": {}}
for view in ("flow", "flow+REL"):
    res = {}
    fitted_P = {}
    for train_parts, target in ((("train",), "calibration"), (("train", "val"), "calibration"),
                                (("train",), "val"), (("train", "calibration"), "val")):
        rows, w = sample(train_parts)
        yy = np.concatenate([D[p]["y"] for p in train_parts])[rows]
        order = np.argsort(rows)
        rows, w, yy = rows[order], w[order], yy[order]
        b = lgb.train(PARAMS, lgb.Dataset(matrix(train_parts, view, rows), yy, weight=w), num_boost_round=200)
        P = predict(b, view_of(D[target], view))
        fitted_P[(train_parts, target)] = P
        res["+".join(train_parts) + " -> " + target] = P
        log("fitted", view, train_parts, "->", target)
    # decision rule fixed from the train-only model on validation
    rule = max(("bayes", "balanced"), key=lambda r: macro(D["val"]["y"], fitted_P[(("train",), "val")], r)[0])
    out = {"rule": rule}
    for k_, P in res.items():
        tgt = k_.split(" -> ")[1]
        m, pc_ = macro(D[tgt]["y"], P, rule)
        out[k_] = {"macro_mcc": m, "per_class": pc_}
    g_cal = out["train+val -> calibration"]["macro_mcc"] - out["train -> calibration"]["macro_mcc"]
    g_val = out["train+calibration -> val"]["macro_mcc"] - out["train -> val"]["macro_mcc"]
    out["gain_cal"], out["gain_val"] = g_cal, g_val
    out["passes"] = bool(min(g_cal, g_val) >= 0.002)
    report["refit"][view] = out
    # smoothing of the train-only model's probabilities
    sm = {}
    for tgt, key_ in (("val", (("train",), "val")), ("calibration", (("train",), "calibration"))):
        P = fitted_P[key_]
        m0, _ = macro(D[tgt]["y"], P, rule)
        m1, pc1 = macro(D[tgt]["y"], smooth(tgt, P), rule)
        sm[tgt] = {"base": m0, "smoothed": m1, "gain": m1 - m0, "per_class": pc1}
    sm["passes"] = bool(min(sm["val"]["gain"], sm["calibration"]["gain"]) >= 0.002)
    report["smooth"][view] = sm
    json.dump(report, open(os.path.join(OUT, "lab4.json"), "w"), indent=1, default=float)
for view in ("flow", "flow+REL"):
    r, s_ = report["refit"][view], report["smooth"][view]
    print(f"== {view} (rule {r['rule']})")
    print(f"   REFIT  cal: train {r['train -> calibration']['macro_mcc']:.4f} -> train+val {r['train+val -> calibration']['macro_mcc']:.4f} ({r['gain_cal']:+.4f}) | "
          f"val: train {r['train -> val']['macro_mcc']:.4f} -> train+cal {r['train+calibration -> val']['macro_mcc']:.4f} ({r['gain_val']:+.4f})  {'PASS' if r['passes'] else '-'}")
    print(f"   SMOOTH val {s_['val']['base']:.4f} -> {s_['val']['smoothed']:.4f} ({s_['val']['gain']:+.4f}) | cal {s_['calibration']['base']:.4f} -> "
          f"{s_['calibration']['smoothed']:.4f} ({s_['calibration']['gain']:+.4f})  {'PASS' if s_['passes'] else '-'}")
    print("   cal per-class train-only :", r["train -> calibration"]["per_class"])
    print("   cal per-class train+val  :", r["train+val -> calibration"]["per_class"])
log("lab4 done")

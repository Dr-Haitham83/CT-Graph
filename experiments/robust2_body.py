
# BOUNDARY ROBUSTNESS OF ALL CONTEXT FEATURES (train / validation / calibration only; the test split is never opened).
# The P10 check showed that segment-reset running statistics can encode "position within a single-attack capture".
# Every relational / temporal window also restarts at a segment boundary. Here every context feature is recomputed on a
# MERGED monitor timeline: all segments of a split form one stream (absolute clocks keep their times; capture-relative
# clocks are laid back to back in segment-name order, 1 s apart), so windows, novelty tables and running statistics
# flow across boundaries as a deployed sensor would see them. Configs are fitted on segmented and on merged features
# (one fit seed; the P10 check measured fit SD <= 0.006 on these splits). A gain that survives merging is not a
# boundary artefact. Nothing is selected here.
OUT = "/data/output/robust2"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261004)
LAB_TRAIN = 600_000
EVAL_CAP = 1_000_000
started = time.time()
self_check()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def features(f, rows):
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    T, t_names, t_groups = temporal(f, rows, STATE_CODES)
    return np.concatenate([rel, T], 1), list(relational_names), t_names, t_groups


def build(split, sample=None, cap=None):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    if cap is not None and len(rows) > cap:
        rows = np.sort(rng.choice(rows, cap, replace=False))
    flow = np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW])
    seg_id = f["stream"].to_numpy()[rows]
    ctx_seg, REL_, TN_, TG_ = features(f, rows)
    hi0, st0 = f["hi"].to_numpy().copy(), f["stream"].to_numpy().copy()
    hi_m, st_m, how = merged_clock(f)
    f["hi"], f["stream"] = hi_m, st_m
    ctx_mrg, *_ = features(f, rows)
    f["hi"], f["stream"] = hi0, st0
    del f
    log("built", split, len(rows), "rows |", how, round(time.time() - started), "s")
    return flow, ctx_seg, ctx_mrg, y[rows], seg_id, REL_, TN_, TG_, how


def train_sample(y):
    n = np.bincount(y, minlength=K)
    train_sample.counts = n
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(y), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


STATE_CODES = top_state_codes(load("train")) if STATE else []
D = {}
D["train"] = build("train", train_sample)
wtr = train_sample.weight
prior = train_sample.counts / train_sample.counts.sum()
D["val"] = build("val", cap=EVAL_CAP)
D["calibration"] = build("calibration", cap=EVAL_CAP)
REL, TN, TG, HOW = D["train"][5], D["train"][6], D["train"][7], D["train"][8]
nf, nr = len(FLOW), len(REL)
ctx_idx = {"REL": list(range(nr))}
for g in GROUPS:
    ctx_idx[g] = [nr + i for i, gg in enumerate(TG) if gg == g]
ALL_T = [j for g in GROUPS for j in ctx_idx[g]]
SETS = {"REL": ctx_idx["REL"], "T_segz": ctx_idx["T_segz"], "all": ctx_idx["REL"] + ALL_T,
        "all_minus_T_segz": ctx_idx["REL"] + [j for g in GROUPS if g != "T_segz" for j in ctx_idx[g]]}


def matrix(split, mode, ctx_cols):
    flow, seg, mrg = D[split][0], D[split][1], D[split][2]
    ctx = seg if mode == "segmented" else mrg
    return np.concatenate([flow, ctx[:, ctx_cols]], 1) if ctx_cols else flow


def macro(y, P, rule):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}


def fit_predict(mode, ctx_cols):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
              "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}
    ytr = D["train"][3]
    b = lgb.train(params, lgb.Dataset(matrix("train", mode, ctx_cols), ytr, weight=wtr), num_boost_round=200)
    return {s: b.predict(matrix(s, mode, ctx_cols)).astype(np.float32) for s in ("val", "calibration")}


report = {"prepared": PREPARED_NAME, "merge": HOW, "protocol": "train/validation/calibration only; test never opened",
          "eval_rows": {s: int(len(D[s][3])) for s in ("val", "calibration")}, "configs": {}}
P = {("flow", "-"): fit_predict("segmented", [])}
for mode in ("segmented", "merged"):
    for name, cc in SETS.items():
        P[(name, mode)] = fit_predict(mode, cc)
        log("fitted", name, mode, round(time.time() - started), "s")
for mode in ("segmented", "merged"):
    P[("multiview(REL,T_segz,all)", mode)] = {s: np.mean([P[(v, mode)][s] for v in ("REL", "T_segz", "all")], 0) for s in ("val", "calibration")}
    P[("multiview_noZ(REL,all-T_segz)", mode)] = {s: np.mean([P[(v, mode)][s] for v in ("REL", "all_minus_T_segz")], 0)
                                                 for s in ("val", "calibration")}
yv, yc = D["val"][3], D["calibration"][3]
for (name, mode), Pm in P.items():
    rule = max(("bayes", "balanced"), key=lambda r: macro(yv, Pm["val"], r)[0])
    v, _ = macro(yv, Pm["val"], rule)
    c, cpc = macro(yc, Pm["calibration"], rule)
    report["configs"][f"{name} [{mode}]"] = {"rule": rule, "val": v, "calibration": c, "cal_per_class": cpc}
f0 = report["configs"]["flow [-]"]
for k, r in report["configs"].items():
    r["gain_vs_flow"] = (r["val"] - f0["val"], r["calibration"] - f0["calibration"])
    r["passes_rule"] = bool(k != "flow [-]" and min(r["gain_vs_flow"]) >= 0.002)
json.dump(report, open(os.path.join(OUT, "robust2.json"), "w"), indent=1, default=float)
print("merge:", HOW, "| eval rows", report["eval_rows"])
print(f"{'config':45s} {'val':>7s} {'cal':>7s}   gain vs flow        rule")
for k, r in report["configs"].items():
    print(f"{k:45s} {r['val']:.4f} {r['calibration']:.4f}   ({r['gain_vs_flow'][0]:+.4f},{r['gain_vs_flow'][1]:+.4f})  {'PASS' if r['passes_rule'] else '-'}")
for k in ("flow [-]", "REL [segmented]", "REL [merged]", "multiview(REL,T_segz,all) [segmented]", "multiview(REL,T_segz,all) [merged]",
          "multiview_noZ(REL,all-T_segz) [merged]"):
    print(k, "cal per-class", report["configs"][k]["cal_per_class"])
log("boundary robustness done")


# P10 ROBUSTNESS CHECK: segment-reset vs continuous normalisation of the temporal window counts (train / validation /
# calibration only; the test split is never opened). P10 = flow + T_segz, whose z-scores reset at every segment start.
# A deployed monitor sees no segment boundaries, so the same 15 window counts are also normalised
#   Z1 continuously (causal cumulative mean / sd over the whole split, no reset),
#   Z2 with an exponentially forgetting mean / sd over ~50k flows across boundaries (the graph model's context_norm),
#   Z3 not at all (raw log counts; a fixed train-fitted z-score is a no-op for trees).
# Every config is fitted with three seeds (fit variance) and as a seed ensemble. Also: macro-MCC in the first 5,000
# flows of each segment (where reset statistics are cold) vs the rest, and the rank correlation of Z0 with Z1 / Z2.
OUT = "/data/output/robust"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261002)
LAB_TRAIN = 600_000
FIT_SEEDS = (7, 8, 9)
HEAD = 5000
started = time.time()
self_check()


def segment_rank(f):
    st = f["stream"].to_numpy().astype(np.int64)
    order = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), st))
    pos = np.empty(len(f), np.int64)
    pos[order] = np.arange(len(f)) - np.searchsorted(st[order], st[order])
    return pos


def build(split, sample=None):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample is not None:
        rows = rows[sample(y[rows])]
    T, names_, groups_ = temporal(f, rows, STATE_CODES, groups=list(SEGZ_GROUPS))
    X = np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), T], 1)
    pos = segment_rank(f)[rows]
    del f, T
    log("built", split, X.shape, round(time.time() - started), "s")
    return X, y[rows], pos, names_, groups_


def train_sample(y):
    n = np.bincount(y, minlength=K)
    train_sample.counts = n
    rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(y), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
    keep = rng.random(len(y)) < rate[y]
    train_sample.weight = 1.0 / rate[y[keep]]
    return keep


STATE_CODES = top_state_codes(load("train")) if STATE else []
Xtr, ytr, _, TN, TG = build("train", train_sample)
wtr = train_sample.weight
prior = train_sample.counts / train_sample.counts.sum()
Xva, yva, pva, *_ = build("val")
Xca, yca, pca, *_ = build("calibration")
nf = len(FLOW)
cols = {"flow": list(range(nf))}
for g in SEGZ_GROUPS:
    cols[g] = [nf + i for i, gg in enumerate(TG) if gg == g]
CONFIGS = {"flow": ["flow"], "P10_Z0_segment_reset": ["flow", "T_segz"], "P10_Z1_continuous": ["flow", "T_segz_cont"],
           "P10_Z2_ewma_50k": ["flow", "T_segz_ewma"], "P10_Z3_raw_counts": ["flow", "T_segz_static"]}


def macro(y, P, rule):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}


def fit(c, seed):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
              "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": seed}
    return lgb.train(params, lgb.Dataset(Xtr[:, c], ytr, weight=wtr), num_boost_round=200)


def predict(booster, X, c, chunk=400_000):
    return np.concatenate([booster.predict(X[i:i + chunk][:, c]) for i in range(0, len(X), chunk)]).astype(np.float32)


report = {"prepared": PREPARED_NAME, "protocol": "train/validation/calibration only; test never opened",
          "ewma_window_flows": EWMA_FLOWS, "fit_seeds": FIT_SEEDS, "configs": {}}
ens = {}
for name, parts in CONFIGS.items():
    c = sorted({j for p in parts for j in cols[p]})
    per_seed = []
    acc = {"val": 0, "calibration": 0}
    for seed in FIT_SEEDS:
        booster = fit(c, seed)
        P = {"val": predict(booster, Xva, c), "calibration": predict(booster, Xca, c)}
        rule = max(("bayes", "balanced"), key=lambda r: macro(yva, P["val"], r)[0])
        per_seed.append({"rule": rule, "val": macro(yva, P["val"], rule)[0], "calibration": macro(yca, P["calibration"], rule)[0]})
        for s in acc:
            acc[s] = acc[s] + P[s] / len(FIT_SEEDS)
    rule = max(("bayes", "balanced"), key=lambda r: macro(yva, acc["val"], r)[0])
    ens[name] = (acc, rule)
    vm, vpc = macro(yva, acc["val"], rule)
    cm_, cpc = macro(yca, acc["calibration"], rule)
    head = {s: {"head": macro(y[p < HEAD], P_[p < HEAD], rule)[0] if (p < HEAD).any() else None,
                "rest": macro(y[p >= HEAD], P_[p >= HEAD], rule)[0] if (p >= HEAD).any() else None,
                "head_rows": int((p < HEAD).sum())}
            for s, y, p, P_ in (("val", yva, pva, acc["val"]), ("calibration", yca, pca, acc["calibration"]))}
    report["configs"][name] = {
        "features": len(c), "seeds": per_seed,
        "val_mean": float(np.mean([r["val"] for r in per_seed])), "val_sd": float(np.std([r["val"] for r in per_seed])),
        "cal_mean": float(np.mean([r["calibration"] for r in per_seed])), "cal_sd": float(np.std([r["calibration"] for r in per_seed])),
        "seed_ensemble": {"rule": rule, "val": vm, "calibration": cm_, "val_per_class": vpc, "cal_per_class": cpc},
        "segment_head_vs_rest": head}
    json.dump(report, open(os.path.join(OUT, "robust.json"), "w"), indent=1)
    r = report["configs"][name]
    log(name, "val", f"{r['val_mean']:.4f}±{r['val_sd']:.4f}", "cal", f"{r['cal_mean']:.4f}±{r['cal_sd']:.4f}",
        "| seed-ensemble", round(vm, 4), round(cm_, 4), "| head/rest cal", head["calibration"])
# gains and the double held-out rule, on the seed-mean scores and on the seed ensembles
f0 = report["configs"]["flow"]
for name, r in report["configs"].items():
    r["gain_vs_flow_seedmean"] = (r["val_mean"] - f0["val_mean"], r["cal_mean"] - f0["cal_mean"])
    r["gain_vs_flow_ensemble"] = (r["seed_ensemble"]["val"] - f0["seed_ensemble"]["val"],
                                  r["seed_ensemble"]["calibration"] - f0["seed_ensemble"]["calibration"])
    r["passes_rule_seedmean"] = bool(name != "flow" and min(r["gain_vs_flow_seedmean"]) >= 0.002)
# rank correlation of the reset z-scores with the continuous / EWMA ones on validation
import scipy.stats as ss
corr = {}
for other in ("T_segz_cont", "T_segz_ewma", "T_segz_static"):
    rs = []
    for j0 in cols["T_segz"]:
        base = TN[j0 - nf][2:]
        match = [j for j in cols[other] if TN[j - nf].split("_", 1)[1] == base]
        if match:
            sub = rng.choice(len(yva), min(100_000, len(yva)), replace=False)
            rs.append(ss.spearmanr(Xva[sub, j0], Xva[sub, match[0]], nan_policy="omit")[0])
    corr[other] = {"median_spearman": float(np.nanmedian(rs)) if rs else None, "min": float(np.nanmin(rs)) if rs else None}
report["rank_correlation_with_segment_reset_z"] = corr
json.dump(report, open(os.path.join(OUT, "robust.json"), "w"), indent=1, default=float)
print(f"{'config':24s} {'val mean±sd':>16s} {'cal mean±sd':>16s} {'ens val':>8s} {'ens cal':>8s}  gain(seed-mean) rule  head/rest (cal)")
for name, r in report["configs"].items():
    h = r["segment_head_vs_rest"]["calibration"]
    print(f"{name:24s} {r['val_mean']:.4f}±{r['val_sd']:.4f}  {r['cal_mean']:.4f}±{r['cal_sd']:.4f}  {r['seed_ensemble']['val']:.4f}  "
          f"{r['seed_ensemble']['calibration']:.4f}  ({r['gain_vs_flow_seedmean'][0]:+.4f},{r['gain_vs_flow_seedmean'][1]:+.4f}) "
          f"{'PASS' if r['passes_rule_seedmean'] else '-':4s}  {h['head'] if h['head'] is None else round(h['head'], 4)}/{h['rest'] if h['rest'] is None else round(h['rest'], 4)} ({h['head_rows']} head rows)")
print("rank correlation with segment-reset z:", corr)
for name in CONFIGS:
    print(name, "cal per-class", report["configs"][name]["seed_ensemble"]["cal_per_class"])
log("robustness check done")

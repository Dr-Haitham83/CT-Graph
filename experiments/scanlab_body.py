
# SCAN-TYPE-INVARIANT PORT-SCAN DETECTOR (ToN; train / validation / calibration only; the test split is never opened).
# Motivation (ToN test-error analysis, development set): the test scanning campaign is a fast VERTICAL port sweep
# (one source, one target, ~3,000 ports per source in minutes) while training scanning is slow web-port probing, so no
# learned model maps it to "scanning" although burst novelty features separate it almost perfectly.
# Detector (classic IDS reasoning, sfPortscan / threshold-random-walk style): a flow is flagged when a causal 10-second
# burst statistic of its source / pair exceeds the 99.99th percentile of TRAINING Benign flows (no scanning label is used
# to set the threshold, so any scan shape can trigger it). Candidates (pre-registered):
#   D1 pair_dnewdport_10s  new destination ports of the (src, dst) pair in 10 s        (vertical scan)
#   D2 src_dnewsport_10s   new source ports (= new connections) of the source in 10 s   (connection burst)
#   D3 src_dnewpeer_10s    new peers of the source in 10 s                              (horizontal scan)
#   D4 D1 or D3
# A flagged flow is re-labelled "scanning" after the model's decision. Scored on validation AND calibration with a
# lab-size flow model and flow + REL model, and on validation with the DEPLOYED hybrid (v4b graph + smoothed context).
# Adoption: macro-MCC gain >= +0.002 on validation AND calibration (lab flow model) AND on the deployed hybrid (val).
OUT = "/data/output/scanlab"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261011)
LAB_TRAIN = 600_000
started = time.time()
DET = {"D1": ["pair_dnewdport_10s"], "D2": ["src_dnewsport_10s"], "D3": ["src_dnewpeer_10s"], "D4": ["pair_dnewdport_10s", "src_dnewpeer_10s"]}
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def build(split, cap=None, ids=False):
    f = load(split, ids=ids)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if cap and len(rows) > cap:
        sc = y[rows] == classes.index("scanning")
        rows = np.sort(np.r_[rows[sc], rng.choice(rows[~sc], cap, replace=False)])
    T, tn, _ = temporal(f, rows, [], groups=["T_count"])
    det = {n: T[:, tn.index(n)] for n in {c for v in DET.values() for c in v}}
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    flow = np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW])
    d = {"y": y[rows], "det": det, "flow": flow, "flow+REL": np.concatenate([flow, rel], 1)}
    if ids:
        d["ids"] = f["source_id"].to_numpy()[rows].astype(str)
    del f, T, rel
    log("built", split, len(rows), round(time.time() - started), "s")
    return d


TR = build("train", cap=1_500_000)
VA = build("val", ids=True)
CA = build("calibration")
SC, BE = classes.index("scanning"), classes.index("Benign")
n = np.bincount(TR["y"], minlength=K)
prior = n / n.sum()
# thresholds from TRAINING Benign flows only
thr = {name: float(np.quantile(TR["det"][name][TR["y"] == BE], 0.9999)) for name in TR["det"]}
log("thresholds (train Benign 99.99%)", thr)


def flag(d, det):
    return np.any([d["det"][c] > thr[c] for c in DET[det]], axis=0)


def macro(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    return float(mcc[t > 0].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if t[i] > 0}


report = {"thresholds": thr, "detectors": DET, "protocol": "train/validation/calibration only; test never opened", "results": {}}
# detector anatomy on each split
for split, d in (("train", TR), ("val", VA), ("calibration", CA)):
    report.setdefault("detector_hits", {})[split] = {
        det: {"scanning_recall": float(flag(d, det)[d["y"] == SC].mean()) if (d["y"] == SC).any() else None,
              "benign_fpr": float(flag(d, det)[d["y"] == BE].mean()),
              "other_attack_flag_rate": float(flag(d, det)[(d["y"] != SC) & (d["y"] != BE)].mean())} for det in DET}
log("detector hits", json.dumps(report["detector_hits"]))
# lab models
params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 7}
rate = np.minimum(1.0, np.maximum(LAB_TRAIN / len(TR["y"]), (LAB_TRAIN / (2 * K)) / np.maximum(n, 1)))
keep = rng.random(len(TR["y"])) < rate[TR["y"]]
for view in ("flow", "flow+REL"):
    b = lgb.train(params, lgb.Dataset(TR[view][keep], TR["y"][keep], weight=1.0 / rate[TR["y"][keep]]), num_boost_round=200)
    res = {}
    for split, d in (("val", VA), ("calibration", CA)):
        P = np.concatenate([b.predict(d[view][i:i + 400_000]) for i in range(0, len(d["y"]), 400_000)])
        base = P.argmax(1)
        res[split] = {"base": macro(d["y"], base)[0]}
        for det in DET:
            pr = np.where(flag(d, det), SC, base)
            m, pc = macro(d["y"], pr)
            res[split][det] = {"macro": m, "gain": m - res[split]["base"], "scanning_mcc": pc.get("scanning")}
    report["results"][view] = res
    log(view, {s: {k: (round(v["gain"], 4) if isinstance(v, dict) else round(v, 4)) for k, v in r.items()} for s, r in res.items()})
# deployed hybrid on validation (v4b graph seeds + smoothed context branch)
def read_preds(directory):
    files = sorted(glob.glob(os.path.join(directory, "predictions-rank*.parquet")))
    t = pa.concat_tables([pq.read_table(p, columns=["source_id", "label", "probabilities"]) for p in files])
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object).astype(str)
    o = np.argsort(ids)
    return ids[o], t["label"].to_numpy()[o], t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1)[o]


runs = [p for p in sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True)) if "-v4bs" in p]
gv = [read_preds(os.path.join(os.path.dirname(p), "validation")) for p in runs]
ids_v, y_v = gv[0][0], gv[0][1].astype(np.int64)
G = np.mean([g[2] for g in gv], 0)
kv = y_v >= 0
C = []
for c in [np.load(p, allow_pickle=False) for p in sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))]:
    cid = c["val_ids"].astype(str)
    o = np.argsort(cid)
    pos = np.searchsorted(cid[o], ids_v[kv])
    C.append(c["val_proba"][o][pos])
H = 0.5 * G[kv] + 0.5 * np.mean(C, 0)
o = np.argsort(VA["ids"])
pos = np.searchsorted(VA["ids"][o], ids_v[kv])
assert np.array_equal(VA["ids"][o][pos], ids_v[kv])
base = H.argmax(1)
yk = y_v[kv]
hyb = {"base": macro(yk, base)[0]}
for det in DET:
    fl = flag(VA, det)[o][pos]
    m, pc = macro(yk, np.where(fl, SC, base))
    hyb[det] = {"macro": m, "gain": m - hyb["base"], "scanning_mcc": pc.get("scanning"), "per_class": pc}
report["deployed_hybrid_val"] = hyb
dec = {}
for det in DET:
    g = [report["results"]["flow"][s][det]["gain"] for s in ("val", "calibration")] + [hyb[det]["gain"]]
    dec[det] = {"gains_flow_val_cal_hybrid_val": g, "adopt": bool(min(g) >= 0.002)}
report["decision"] = dec
json.dump(report, open(os.path.join(OUT, "scanlab.json"), "w"), indent=1, default=float)
print("thresholds", thr)
print("detector hits", json.dumps(report["detector_hits"], indent=1))
for view, r in report["results"].items():
    print(view, json.dumps({s: {k: (v if not isinstance(v, dict) else {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in x.items()} for s, x in r.items()}))
print("deployed hybrid val", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "per_class"}) for k, v in hyb.items()}, default=float))
print("DECISION", json.dumps(dec, indent=1))
log("scan lab done")

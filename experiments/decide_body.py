# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log)
# v6r ADOPTION DECISION (validation only; no test prediction file is opened). Pre-registered rule: adopt the v6r graph
# branch for CIC iff (i) its 3-seed ensemble validation macro-MCC >= the v4b ensemble's + 0.002 AND (ii) the
# equal-weight hybrid with the CIC context branch has validation macro-MCC >= the v4b hybrid's + 0.002. The decision
# rule (Bayes / prior-balanced) is taken as the better of the two on validation for every model, symmetrically.
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = "/data/output/decide"
os.makedirs(OUT, exist_ok=True)


FAMILIES = globals().get("FAMILIES", ("v6r8", "v6r", "v4b"))
REF = globals().get("REF", "v4b")
RULE = globals().get("RULE", "val")  # "val": gains on validation; "cal": saturated validation -> calibration decides


def family(path):
    for tag in FAMILIES:
        if f"-{tag}s" in path:
            return tag
    return None


def read_validation(results_path, sub="validation"):
    vdir = os.path.join(os.path.dirname(results_path), sub)
    files = sorted(glob.glob(os.path.join(vdir, "predictions-rank*.parquet")))
    t = pa.concat_tables([pq.read_table(p, columns=["source_id", "label", "probabilities"]) for p in files])
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object)
    o = np.argsort(ids)
    P = t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1)
    return ids[o], t["label"].to_numpy()[o], P[o].astype(np.float64)


runs = {tag: [] for tag in FAMILIES}
for path in sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True)):
    fam = family(path)
    if fam and glob.glob(os.path.join(os.path.dirname(path), "validation", "predictions-rank*.parquet")):
        runs[fam].append(path)
log({k: len(v) for k, v in runs.items()}, runs)
res0 = json.load(open(runs[REF][0]))
classes = res0["classes"]
K = len(classes)
counts = np.asarray([res0["split_class_counts"]["train"].get(c, 0) for c in classes], float)
prior = counts / counts.sum()


def macro(y, P, rule):
    pred = P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return float(mcc[present].mean()), {classes[i]: round(float(mcc[i]), 4) for i in range(K) if present[i]}


def best(y, P):
    scores = {r: macro(y, P, r) for r in ("bayes", "balanced")}
    rule = max(scores, key=lambda r: scores[r][0])
    return {"rule": rule, "macro_mcc": scores[rule][0], "per_class": scores[rule][1], "both_rules": {r: v[0] for r, v in scores.items()}}


ct_paths = sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))
CONTEXT_KEYS = globals().get("CONTEXT_KEYS", ("ctgbdt7", "ctgbdt5"))
CONTEXTS = {name: [np.load(p, allow_pickle=False) for p in ct_paths if f"-{name}-" in p] for name in CONTEXT_KEYS}
CONTEXTS = {k: v for k, v in CONTEXTS.items() if v}
log("context branch files", ct_paths, {k: len(v) for k, v in CONTEXTS.items()})
report = {"protocol": "validation predictions only; no test file opened", "runs": runs, "context_files": ct_paths}
runs = {k: v for k, v in runs.items() if v}
for fam, paths in runs.items():
    vals = [read_validation(p) for p in paths]
    ids, y = vals[0][0], vals[0][1]
    for v in vals[1:]:
        assert np.array_equal(v[0], ids)
    keep = y >= 0
    G = np.mean([v[2] for v in vals], 0)[keep]
    ids_k, y_k = ids[keep], y[keep].astype(np.int64)
    seeds = [best(y_k, v[2][keep])["macro_mcc"] for v in vals]
    report[fam] = {"seed_val_macro_mcc": seeds, "graph_ensemble": best(y_k, G), "hybrid": {}}
    if all(glob.glob(os.path.join(os.path.dirname(p), "calibration", "predictions-rank*.parquet")) for p in paths):
        cals = [read_validation(p, "calibration") for p in paths]
        yc = cals[0][1]
        kc = yc >= 0
        report[fam]["graph_ensemble_calibration"] = best(yc[kc].astype(np.int64), np.mean([c[2] for c in cals], 0)[kc])
        report[fam]["seed_cal_macro_mcc"] = [best(yc[kc].astype(np.int64), c[2][kc])["macro_mcc"] for c in cals]
    for cname, ct in CONTEXTS.items():
        C = []
        for c in ct:
            cid = c["val_ids"].astype(object)
            o = np.argsort(cid)
            pos = np.searchsorted(cid[o], ids_k)
            assert np.all(pos < len(cid)) and np.array_equal(cid[o][pos], ids_k)
            C.append(c["val_proba"].astype(np.float64)[o][pos])
        C = np.mean(C, 0)
        report[fam]["hybrid"][cname] = {"hybrid_equal_weight": best(y_k, 0.5 * G + 0.5 * C), "context_branch": best(y_k, C)}
    log(fam, "seeds", [round(s, 4) for s in seeds], "ensemble", round(report[fam]["graph_ensemble"]["macro_mcc"], 4),
        "hybrid", {c: round(v["hybrid_equal_weight"]["macro_mcc"], 4) for c, v in report[fam]["hybrid"].items()})
report["decisions"] = {}
for fam in runs:
    if fam == REF:
        continue
    g_gain = report[fam]["graph_ensemble"]["macro_mcc"] - report[REF]["graph_ensemble"]["macro_mcc"]
    h_gain = {c: report[fam]["hybrid"][c]["hybrid_equal_weight"]["macro_mcc"] - report[REF]["hybrid"][c]["hybrid_equal_weight"]["macro_mcc"]
              for c in CONTEXTS}
    d = {"graph_gain": g_gain, "hybrid_gain": h_gain, "rule": RULE}
    if RULE == "cal":
        c_gain = report[fam]["graph_ensemble_calibration"]["macro_mcc"] - report[REF]["graph_ensemble_calibration"]["macro_mcc"]
        d["graph_gain_calibration"] = c_gain
        d["adopt"] = bool(c_gain >= 0.002 and g_gain >= -0.002 and all(v >= -0.002 for v in h_gain.values()))
    else:
        d["adopt"] = bool(g_gain >= 0.002 and all(v >= 0.002 for v in h_gain.values()))
    report["decisions"][fam] = d
json.dump(report, open(os.path.join(OUT, "decide.json"), "w"), indent=1, default=float)
print(json.dumps(report["decisions"], indent=1))
for fam in runs:
    print(fam, "graph per-class", report[fam]["graph_ensemble"]["per_class"])
log("decision done")

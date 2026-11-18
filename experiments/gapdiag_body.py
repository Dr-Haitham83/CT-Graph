# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log)
# VALIDATION -> TEST GENERALISATION-GAP DIAGNOSIS for the DEVELOPMENT dataset (ToN; its test split was already used
# during development, CIC stays the untouched confirmation set). Per class and per branch (graph seeds, graph ensemble,
# context GBDT, fusions): F1 / recall / precision on validation vs test, confusion destinations, the test-minus-val
# prior shift, EM-estimated test prior (label-free), per-segment test macro-MCC, calibration (NLL, ECE), and the
# fusion weight that validation vs test would prefer (showing whether validation-based selection transfers).
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = "/data/output/gapdiag"
os.makedirs(OUT, exist_ok=True)
EPS = 1e-12


def read(directory):
    files = sorted(glob.glob(os.path.join(directory, "predictions-rank*.parquet")))
    t = pa.concat_tables([pq.read_table(p, columns=["source_id", "stream", "label", "probabilities"]) for p in files])
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object)
    o = np.argsort(ids)
    proba = t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1)
    return {"ids": ids[o], "stream": np.asarray(t["stream"].to_pylist(), dtype=object)[o], "y": t["label"].to_numpy()[o],
            "proba": proba[o].astype(np.float64)}


runs = [p for p in sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True))
        if glob.glob(os.path.join(os.path.dirname(p), "validation", "predictions-rank*.parquet"))]
res0 = json.load(open(runs[0]))
classes = res0["classes"]
K = len(classes)
counts = np.asarray([res0["split_class_counts"]["train"].get(c, 0) for c in classes], float)
prior = counts / counts.sum()
split = {"val": [read(os.path.join(os.path.dirname(p), "validation")) for p in runs],
         "test": [read(os.path.dirname(p)) for p in runs]}
ct = [np.load(p, allow_pickle=False) for p in sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))]


def aligned(ids_ref, ids, values):
    o = np.argsort(ids)
    pos = np.searchsorted(ids[o], ids_ref)
    assert np.all(pos < len(ids)) and np.array_equal(ids[o][pos], ids_ref)
    return values[o][pos]


def per_class(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, p = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, p - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(K), where=(2 * tp + fp + fn) > 0)
    rec = np.divide(tp, t, out=np.zeros(K), where=t > 0)
    prec = np.divide(tp, p, out=np.zeros(K), where=p > 0)
    present = t > 0
    top = {classes[i]: sorted([(int(cm[i, j]), classes[j]) for j in range(K) if j != i and cm[i, j] > 0], reverse=True)[:3] for i in range(K) if present[i]}
    return {"macro_mcc": float(mcc[present].mean()), "f1": dict(zip(classes, np.round(f1, 4).tolist())),
            "recall": dict(zip(classes, np.round(rec, 4).tolist())), "precision": dict(zip(classes, np.round(prec, 4).tolist())),
            "confusions": top}


rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, EPS)).argmax(1)}
report = {"classes": classes, "graph_runs": len(runs), "ctgbdt_seeds": len(ct)}
data = {}
for name in ("val", "test"):
    base = split[name][0]
    keep = base["y"] >= 0
    ids, y = base["ids"][keep], base["y"][keep].astype(np.int64)
    G_seeds = [r["proba"][keep] for r in split[name]]
    G = np.mean(G_seeds, 0)
    key = "val" if name == "val" else "test"
    C = np.mean([aligned(ids, c[f"{key}_ids"].astype(object), c[f"{key}_proba"].astype(np.float64)) for c in ct], 0) if ct else None
    data[name] = {"y": y, "G": G, "G_seeds": G_seeds, "C": C, "stream": base["stream"][keep]}
    report[f"{name}_prior"] = dict(zip(classes, np.round(np.bincount(y, minlength=K) / len(y), 5).tolist()))
# The deployed rule per branch is chosen on validation (as in the post-hoc), then applied to both splits.
choice = {}
for branch in ("G", "C"):
    if data["val"][branch] is None:
        continue
    choice[branch] = max(rules, key=lambda r: per_class(data["val"]["y"], rules[r](data["val"][branch]))["macro_mcc"])
report["branches"] = {}
for branch, label in (("G", "graph_ensemble"), ("C", "context_gbdt")):
    if branch not in choice:
        continue
    report["branches"][label] = {s: per_class(data[s]["y"], rules[choice[branch]](data[s][branch])) for s in ("val", "test")}
for i in range(len(runs)):
    report["branches"][f"graph_seed{i}"] = {s: per_class(data[s]["y"], rules[choice["G"]](data[s]["G_seeds"][i])) for s in ("val", "test")}
# Fusion weights: which weight would each split prefer (the post-hoc chooses on validation only).
if "C" in choice:
    curve = {}
    for s in ("val", "test"):
        curve[s] = {}
        for w in np.round(np.linspace(0, 1, 11), 2):
            P = w * data[s]["G"] + (1 - w) * data[s]["C"]
            curve[s][str(w)] = {r: round(per_class(data[s]["y"], rules[r](P))["macro_mcc"], 4) for r in rules}
    report["fusion_weight_curves"] = curve
    for s in ("val", "test"):
        wbest = max(((v[r], w, r) for w, v in curve[s].items() for r in v))
        report[f"fusion_best_weight_on_{s}"] = {"macro_mcc": wbest[0], "weight_graph": float(wbest[1]), "rule": wbest[2]}
    # Per-class weight preference: F1 of graph vs context per class on each split.
    report["per_class_branch_f1"] = {c: {s: {"graph": report["branches"]["graph_ensemble"][s]["f1"][c],
                                             "context": report["branches"]["context_gbdt"][s]["f1"][c]} for s in ("val", "test")}
                                    for c in classes}
# Calibration and label-free prior estimate on test.
def em(P, iters=200):
    q = prior.copy()
    for _ in range(iters):
        w = P * (q / np.maximum(prior, EPS)); w /= w.sum(1, keepdims=True); q = w.mean(0)
    return q
for s in ("val", "test"):
    G, y = data[s]["G"], data[s]["y"]
    nll = float(-np.log(np.maximum(G[np.arange(len(y)), y], EPS)).mean())
    conf = G.max(1); correct = G.argmax(1) == y
    bins = np.minimum((conf * 15).astype(int), 14)
    ece = float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(15) if (bins == b).any()))
    report[f"graph_calibration_{s}"] = {"nll": nll, "ece": ece, "em_prior": dict(zip(classes, np.round(em(G), 5).tolist()))}
# Per-segment test performance of both branches (where does the graph lose?).
seg = {}
for sname in np.unique(data["test"]["stream"]):
    m = data["test"]["stream"] == sname
    y = data["test"]["y"][m]
    counts_ = np.bincount(y, minlength=K)
    seg[str(sname)] = {"rows": int(m.sum()), "classes": {classes[i]: int(counts_[i]) for i in range(K) if counts_[i]},
                       "graph_acc": float((rules[choice["G"]](data["test"]["G"][m]) == y).mean())}
    if "C" in choice:
        seg[str(sname)]["context_acc"] = float((rules[choice["C"]](data["test"]["C"][m]) == y).mean())
report["test_segments"] = seg
json.dump(report, open(os.path.join(OUT, "gapdiag.json"), "w"), indent=1, default=float)
for b, v in report["branches"].items():
    print(f"{b:16s} val {v['val']['macro_mcc']:.4f} test {v['test']['macro_mcc']:.4f}")
print(json.dumps({k: report.get(k) for k in ("fusion_best_weight_on_val", "fusion_best_weight_on_test")}, indent=1))
print(json.dumps(report.get("per_class_branch_f1"), indent=0))
for s in ("val", "test"):
    print(s, "graph F1", report["branches"]["graph_ensemble"][s]["f1"])
    print(s, "graph confusions", report["branches"]["graph_ensemble"][s]["confusions"])
print(json.dumps(report["test_segments"], indent=0)[:6000])
log("gap diagnosis done")

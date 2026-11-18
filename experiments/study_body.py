# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# VALIDATION-ONLY design study for the fusion of the proposed CT-Graph hybrid. No test file is opened.
#   1. per-class behaviour of the graph branch, the context branch and the current fusion on validation;
#   2. flow-feature ambiguity: for every validation row, the training-label distribution of its exact feature
#      vector (a lookup built from TRAIN labels only) -> novel / unambiguous / label-conflicting rows;
#   3. cross-fitted (block-grouped K-fold over validation) estimates of candidate fusion rules, so the rule can be
#      chosen and frozen before any test result exists.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = "/data/output/study"
os.makedirs(OUT, exist_ok=True)
FOLDS = 5


def cm_summary(cm):
    t, p = cm.sum(1).astype(float), cm.sum(0).astype(float)
    s, c = cm.sum(), np.trace(cm)
    tp = np.diag(cm).astype(float)
    fn, fp = t - tp, p - tp
    tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_k = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    f1_k = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    present = t > 0
    den_all = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return {"macro_mcc": float(mcc_k[present].mean()) if present.any() else 0.0,
            "macro_f1": float(f1_k[present].mean()) if present.any() else 0.0, "accuracy": float(c / max(s, 1)),
            "mcc": float((c * s - (p * t).sum()) / den_all) if den_all > 0 else 0.0,
            "per_class_f1": f1_k.tolist(), "per_class_mcc": mcc_k.tolist(), "support": t.astype(int).tolist()}


def score(y, pred, k):
    return cm_summary(np.bincount(y * k + pred, minlength=k * k).reshape(k, k))


# ---------------------------------------------------------------- inputs (validation only)
runs = sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True))
results = [json.load(open(p)) for p in runs]
classes = results[0]["classes"]
K = len(classes)
BENIGN = [c.lower() for c in classes].index("benign")
block_s = float(results[0].get("split_plan", {}).get("block_s") or 600.0)


def read_validation(directory):
    tables = [pq.read_table(p, columns=["source_id", "stream", "first", "label", "probabilities"])
              for p in sorted(glob.glob(os.path.join(directory, "validation", "predictions-rank*.parquet")))]
    t = pa.concat_tables(tables)
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object)
    o = np.argsort(ids)
    return {"ids": ids[o], "stream": np.asarray(t["stream"].to_pylist(), dtype=object)[o], "first": t["first"].to_numpy()[o],
            "y": t["label"].to_numpy()[o], "proba": np.asarray(t["probabilities"].to_pylist(), dtype=np.float64)[o]}


graph = [read_validation(os.path.dirname(p)) for p in runs]
for g in graph[1:]:
    assert np.array_equal(g["ids"], graph[0]["ids"])
g0 = graph[0]
keep = g0["y"] >= 0
ids, y = g0["ids"][keep], g0["y"][keep].astype(np.int64)
pg = np.mean([g["proba"] for g in graph], 0)[keep]
units = np.asarray([f"{s}|{int(np.floor(f / block_s))}" for s, f in zip(g0["stream"][keep], g0["first"][keep])], dtype=object)
ct = [np.load(p, allow_pickle=False) for p in sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))]


def aligned(values, source_ids):
    o = np.argsort(source_ids)
    position = np.searchsorted(source_ids[o], ids)
    assert np.all(position < len(source_ids)) and np.array_equal(source_ids[o][position], ids)
    return values[o][position]


pc_ = np.mean([aligned(c["val_proba"].astype(np.float64), c["val_ids"].astype(object)) for c in ct], 0)
prior = np.asarray(ct[0]["prior"], dtype=np.float64)
log("validation rows", len(y), "graph runs", len(graph), "context seeds", len(ct), "units", len(np.unique(units)))

# ---------------------------------------------------------------- flow-feature ambiguity (train labels only)
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
width = len(prep["means"])
label_set = pa.array(prep["classes"])


def shard_path(split, shard):
    return shard["path"] if os.path.exists(shard["path"]) else os.path.join(root, "prepared", split, os.path.basename(shard["path"]))


def matrix(t):
    x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
    cats = []
    for j, vocab in enumerate(prep["vocabs"]):
        values = pc.list_element(t["categories"], j)
        index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
        cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
    lab = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False).astype(np.int64)
    return (np.concatenate([x, np.column_stack(cats)], 1) if cats else x), lab


usable = None
for shard in manifest["shards"]["train"]:
    x, _ = matrix(pq.read_table(shard_path("train", shard), columns=["numeric", "categories", "label"]))
    observed = ~np.all(np.isnan(x), 0)
    usable = observed if usable is None else usable | observed


def hashes(x):
    return pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(x[:, usable], nan=-7.77e30)), index=False).to_numpy()


pairs = []
for shard in manifest["shards"]["train"]:
    x, lab = matrix(pq.read_table(shard_path("train", shard), columns=["numeric", "categories", "label"]))
    h = hashes(x)[lab >= 0]
    pairs.append(pd.DataFrame({"h": h, "y": lab[lab >= 0]}).value_counts().reset_index())
counts = pd.concat(pairs).groupby(["h", "y"], sort=False)["count"].sum().reset_index()
table = counts.pivot_table(index="h", columns="y", values="count", fill_value=0)
table = table.reindex(columns=range(K), fill_value=0)
train_hash = table.index.to_numpy()
train_counts = table.to_numpy(np.float64)
del counts, table, pairs
val_hash, val_ids = [], []
for shard in manifest["shards"]["val"]:
    t = pq.read_table(shard_path("val", shard), columns=["numeric", "categories", "label", "source_id"])
    x, lab = matrix(t)
    val_hash.append(hashes(x)[lab >= 0])
    val_ids.append(np.asarray(t["source_id"].to_pylist(), dtype=object)[lab >= 0])
val_hash = aligned(np.concatenate(val_hash), np.concatenate(val_ids))
position = np.minimum(np.searchsorted(train_hash, val_hash), len(train_hash) - 1)
seen = train_hash[position] == val_hash
dist = np.where(seen[:, None], train_counts[position], 0.0)
n_seen = dist.sum(1)
p_lab = np.divide(dist, n_seen[:, None], out=np.zeros_like(dist), where=n_seen[:, None] > 0)
entropy = -np.sum(np.where(p_lab > 0, p_lab * np.log(np.maximum(p_lab, 1e-300)), 0.0), 1)
group = np.where(~seen, "novel", np.where(entropy > 1e-9, "conflicting", "unambiguous")).astype(object)
report = {"classes": classes, "rows": int(len(y)), "units": int(len(np.unique(units))),
          "ambiguity_groups": {g: int((group == g).sum()) for g in ("novel", "unambiguous", "conflicting")}}

# ---------------------------------------------------------------- per-class and per-group diagnostics
rules = {"bayes": lambda p: p.argmax(1), "balanced": lambda p: (p / np.maximum(prior, 1e-12)).argmax(1)}


def best_rule(p):
    s = {r: score(y, f(p), K)["macro_mcc"] for r, f in rules.items()}
    return max(s, key=s.get)


branches = {"graph": pg, "context": pc_, "fusion_w0.5": 0.5 * pg + 0.5 * pc_}
preds = {n: rules[best_rule(p)](p) for n, p in branches.items()}
report["branches"] = {}
for n, pred in preds.items():
    whole = score(y, pred, K)
    report["branches"][n] = {"all": {m: whole[m] for m in ("macro_mcc", "mcc", "accuracy", "macro_f1")},
                             "per_class_f1": dict(zip(classes, whole["per_class_f1"])),
                             "groups": {g: {**{m: v for m, v in score(y[group == g], pred[group == g], K).items()
                                               if m in ("macro_mcc", "mcc", "accuracy")}, "rows": int((group == g).sum())}
                                        for g in ("novel", "unambiguous", "conflicting") if (group == g).any()}}
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    off = [(int(cm[i, j]), classes[i], classes[j]) for i in range(K) for j in range(K) if i != j and cm[i, j]]
    report["branches"][n]["top_confusions"] = sorted(off, reverse=True)[:8]
    log(n, report["branches"][n]["all"], {g: round(v["accuracy"], 4) for g, v in report["branches"][n]["groups"].items()})
agree = preds["graph"] == preds["context"]
report["complementarity"] = {"agree": float(agree.mean()),
                             "graph_right_context_wrong": float(((preds["graph"] == y) & (preds["context"] != y)).mean()),
                             "context_right_graph_wrong": float(((preds["context"] == y) & (preds["graph"] != y)).mean()),
                             "by_group": {g: {"graph_only": float(((preds["graph"] == y) & (preds["context"] != y))[group == g].mean()),
                                              "context_only": float(((preds["context"] == y) & (preds["graph"] != y))[group == g].mean())}
                                          for g in ("novel", "unambiguous", "conflicting") if (group == g).any()}}

# ---------------------------------------------------------------- cross-fitted fusion candidates
names, unit_index = np.unique(units, return_inverse=True)
rng = np.random.default_rng(20260925)
fold_of_unit = rng.permutation(len(names)) % FOLDS
fold = fold_of_unit[unit_index]
lg, lc = np.log(np.maximum(pg, 1e-9)), np.log(np.maximum(pc_, 1e-9))
amb = np.clip(entropy / np.log(K), 0, 1)
weights = np.round(np.linspace(0, 1, 11), 2)


def fit_global(tr):
    best = (-1, None)
    for w in weights:
        p = w * pg[tr] + (1 - w) * pc_[tr]
        for r, f in rules.items():
            v = score(y[tr], f(p), K)["macro_mcc"]
            if v > best[0]:
                best = (v, (w, r))
    w, r = best[1]
    return lambda idx: rules[r](w * pg[idx] + (1 - w) * pc_[idx])


def fit_ambiguity(tr):
    """Separate fusion weight for ambiguous (label-conflicting) and other feature vectors."""
    best = (-1, None)
    conflicting = group == "conflicting"
    for w_amb in weights:
        for w_other in weights:
            w = np.where(conflicting, w_amb, w_other)[:, None]
            p = w[tr] * pg[tr] + (1 - w[tr]) * pc_[tr]
            for r, f in rules.items():
                v = score(y[tr], f(p), K)["macro_mcc"]
                if v > best[0]:
                    best = (v, (w_amb, w_other, r))
    w_amb, w_other, r = best[1]
    return lambda idx: rules[r](np.where(conflicting[idx], w_amb, w_other)[:, None] * pg[idx]
                                + (1 - np.where(conflicting[idx], w_amb, w_other)[:, None]) * pc_[idx])


def fit_stack(tr, with_ambiguity):
    """Multinomial logistic stacking on branch log-probabilities (optionally x ambiguity); C by inner block CV."""
    from sklearn.linear_model import LogisticRegression

    def features(idx):
        f = [lg[idx], lc[idx]]
        if with_ambiguity:
            f += [lg[idx] * amb[idx, None], lc[idx] * amb[idx, None], amb[idx, None]]
        return np.concatenate(f, 1)

    sub = tr if tr.sum() <= 200000 else tr & (rng.random(len(tr)) < 200000 / tr.sum())
    inner = fold_of_unit[unit_index] % 2
    best = (-1, 1.0)
    for C in (0.01, 0.1, 1.0):
        v = []
        for f in (0, 1):
            a, b = sub & (inner != f), sub & (inner == f)
            if not a.any() or not b.any():
                continue
            m = LogisticRegression(C=C, max_iter=300, class_weight="balanced").fit(features(a), y[a])
            p = np.zeros((b.sum(), K))
            p[:, m.classes_] = m.predict_proba(features(b))
            v.append(score(y[b], p.argmax(1), K)["macro_mcc"])
        if v and np.mean(v) > best[0]:
            best = (float(np.mean(v)), C)
    m = LogisticRegression(C=best[1], max_iter=300, class_weight="balanced").fit(features(sub), y[sub])

    def predict(idx):
        p = np.zeros((idx.sum(), K))
        p[:, m.classes_] = m.predict_proba(features(idx))
        return p.argmax(1)
    return predict


candidates = {"global_weight (current)": fit_global, "ambiguity_weight": fit_ambiguity,
              "stack_lr": lambda tr: fit_stack(tr, False), "stack_lr_ambiguity": lambda tr: fit_stack(tr, True)}
report["cross_fitted"] = {}
for name, fitter in candidates.items():
    started = time.time()
    oof = np.full(len(y), -1)
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        if te.any():
            oof[te] = fitter(tr)(te)
    whole = score(y, oof, K)
    fold_scores = [score(y[fold == f], oof[fold == f], K)["macro_mcc"] for f in range(FOLDS) if (fold == f).any()]
    report["cross_fitted"][name] = {"oof_macro_mcc": whole["macro_mcc"], "oof_mcc": whole["mcc"], "oof_accuracy": whole["accuracy"],
                                    "fold_macro_mcc": fold_scores, "seconds": time.time() - started,
                                    "groups": {g: score(y[group == g], oof[group == g], K)["accuracy"]
                                               for g in ("novel", "unambiguous", "conflicting") if (group == g).any()}}
    log("cross-fitted", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report["cross_fitted"][name].items()
                               if k in ("oof_macro_mcc", "oof_mcc", "oof_accuracy")})
json.dump(report, open(os.path.join(OUT, "study.json"), "w"), indent=1)
print(json.dumps({k: report[k] for k in ("ambiguity_groups", "complementarity", "cross_fitted")}, indent=1))
for n, b in report["branches"].items():
    print(n, json.dumps(b["all"]), "\n  per-class F1", {c: round(v, 3) for c, v in b["per_class_f1"].items()},
          "\n  groups", json.dumps(b["groups"]), "\n  confusions", b["top_confusions"])
log("study kernel done")

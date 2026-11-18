# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# Context branch of the proposed CT-Graph hybrid: gradient-boosted trees on the flow features PLUS the
# label-free causal graph context exported by run_pipeline --mode export_context. Same frozen split,
# fit on train, early stopping and decision rule on validation, test scored once. Validation and test
# probabilities are saved for the validation-weighted fusion with the graph branch.
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
# The context kernel's output also carries a manifest copy; the prepared shards live beside the other one.
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
# FLOW_ONLY: the frozen feature set of CT-GBDT v4 on ToN / 5G was "flow" (chosen on validation); refits of that frozen
# branch (e.g. on cleaned labels) skip the context export and the feature-set selection.
FLOW_ONLY = globals().get("FLOW_ONLY", False)
context_root = None if FLOW_ONLY else os.path.dirname(sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/context/context.json", recursive=True))[0])
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
context_names = [] if FLOW_ONLY else json.load(open(os.path.join(context_root, "context.json")))["features"]
classes = prep["classes"]
K = len(classes)
OUT = "/data/output/context_gbdt"
os.makedirs(OUT, exist_ok=True)


def shard_path(split, shard):
    path = shard["path"]
    return path if os.path.exists(path) else os.path.join(root, "prepared", split, os.path.basename(path))


def load(split, keep_ids=True):
    xs, ys, ids, orders, streams = [], [], [], [], []
    width = len(prep["means"])
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=["source_id", "source_order", "numeric", "categories", "label", "stream"])
        x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
        cats = []
        for j, vocab in enumerate(prep["vocabs"]):
            values = pc.list_element(t["categories"], j)
            index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
            cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
        xs.append(np.concatenate([x, np.column_stack(cats)], 1))
        ys.append(pc.fill_null(pc.index_in(t["label"], value_set=pa.array(classes)), -1).to_numpy(zero_copy_only=False).astype(np.int64))
        orders.append(t["source_order"].to_numpy())
        if keep_ids:
            ids.append(np.asarray(t["source_id"].to_pylist(), dtype=object))
            streams.append(np.asarray(t["stream"].to_pylist(), dtype=object))
        del t
    x, y, orders = np.concatenate(xs), np.concatenate(ys), np.concatenate(orders)
    del xs
    ids = np.concatenate(ids) if keep_ids else None
    streams = np.concatenate(streams) if keep_ids else None
    keep = y >= 0
    if FLOW_ONLY:
        return (x, y, ids, streams) if keep.all() else (x[keep], y[keep], (ids[keep] if keep_ids else None),
                                                         (streams[keep] if keep_ids else None))
    context = pa.concat_tables([pq.read_table(p) for p in sorted(glob.glob(os.path.join(context_root, split, "*.parquet")))])
    corder = context["source_order"].to_numpy()
    sort = np.argsort(corder)
    position = np.searchsorted(corder[sort], orders)
    if np.any(position >= len(corder)) or not np.array_equal(corder[sort][position], orders):
        raise SystemExit(f"Context rows do not match the {split} split")
    take = sort[position]
    full = np.empty((len(y), x.shape[1] + len(context_names)), np.float32)
    full[:, :x.shape[1]] = x
    del x
    for j, n in enumerate(context_names):
        full[:, -len(context_names) + j] = context[n].to_numpy()[take]
    del context
    keep = y >= 0
    if keep.all():  # avoid a second full-size copy of the training matrix
        return full, y, ids, streams
    return full[keep], y[keep], (ids[keep] if keep_ids else None), (streams[keep] if keep_ids else None)


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
    return {"macro_mcc": float(mcc_k[present].mean()), "macro_f1": float(f1_k[present].mean()), "accuracy": float(c / s),
            "mcc": float((c * s - (p * t).sum()) / den_all) if den_all > 0 else 0.0,
            "per_class_f1": dict(zip(classes, f1_k.tolist()))}


log("loading flow features + causal context")
Xtr, ytr, _, _ = load("train", keep_ids=False)
Xva, yva, idva, _ = load("val")
Xte, yte, idte, ste = load("test")
usable = ~np.all(np.isnan(Xtr), 0)
Xtr, Xva, Xte = Xtr[:, usable], Xva[:, usable], Xte[:, usable]
log("shapes", Xtr.shape, Xva.shape, Xte.shape)
prior = np.bincount(ytr, minlength=K) / len(ytr)
import lightgbm as lgb  # noqa: E402

report = {"model": "CT-GBDT (flow + causal graph context, feature set chosen on validation)",
          "context_features": context_names, "classes": classes}
ctx_usable = usable[len(usable) - len(context_names):]  # (usable[-0:] would be the whole vector when FLOW_ONLY)
n_flow = Xtr.shape[1] - int(np.sum(ctx_usable))
context_used = [n for n, u in zip(context_names, ctx_usable) if u]
short = [j for j, n in enumerate(context_used) if n.endswith("_60s") or "rhythm" in n or n.endswith("seen_before")
         or n.startswith(("rel_", "same_")) or n == "log_degree"]
feature_sets = ({"flow": np.arange(n_flow)} if FLOW_ONLY else
                {"flow": np.arange(n_flow), "flow+short_context": np.r_[np.arange(n_flow), n_flow + np.asarray(short, int)],
                 "flow+full_context": np.arange(Xtr.shape[1])})
rules = {"bayes": lambda p: p.argmax(1), "balanced": lambda p: (p / np.maximum(prior, 1e-12)).argmax(1)}


def take(X, columns):
    return X if len(columns) == X.shape[1] else X[:, columns]


def fit(columns, seed):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": seed, "max_bin": 255}
    booster = lgb.train(params, lgb.Dataset(take(Xtr, columns), ytr), num_boost_round=1000,
                        valid_sets=[lgb.Dataset(take(Xva, columns), yva)],
                        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(100)])
    pv = booster.predict(take(Xva, columns), num_iteration=booster.best_iteration)
    val = {r: cm_summary(np.bincount(yva * K + f(pv), minlength=K * K).reshape(K, K))["macro_mcc"] for r, f in rules.items()}
    return booster, pv, val


# 1) Feature-set selection on VALIDATION only (seed of the first run).
selection, fitted = {}, {}
for name, columns in feature_sets.items():
    started = time.time()
    booster, pv, val = fit(columns, SEEDS[0])
    selection[name] = {"validation_macro_mcc": val, "best": max(val.values()), "trees": booster.best_iteration,
                       "fit_seconds": time.time() - started}
    fitted[name] = (booster, pv, val)
    log("feature set", name, {k: round(v, 4) for k, v in val.items()})
chosen = max(selection, key=lambda n: selection[n]["best"])
report["feature_set_selection"] = selection
report["chosen_feature_set"] = chosen
json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
# 2) The chosen feature set with every seed; test scored once per seed.
columns = feature_sets[chosen]
all_names = [n for n, u in zip(prep["feature_names"] + prep["schema"]["categorical"] + context_names, usable) if u]
for seed in SEEDS:
    started = time.time()
    booster, pv, val = fitted[chosen] if seed == SEEDS[0] else fit(columns, seed)
    rule = max(val, key=val.get)
    pt = booster.predict(take(Xte, columns), num_iteration=booster.best_iteration)
    test = cm_summary(np.bincount(yte * K + rules[rule](pt), minlength=K * K).reshape(K, K))
    report[f"seed{seed}"] = {"rule": rule, "validation_macro_mcc": val, "test": test, "trees": booster.best_iteration,
                             "fit_seconds": time.time() - started,
                             "top_features_gain": sorted(zip([all_names[j] for j in columns], booster.feature_importance("gain").tolist()),
                                                         key=lambda t: -t[1])[:20]}
    np.savez_compressed(os.path.join(OUT, f"ctgbdt_seed{seed}.npz"), val_ids=idva.astype(str), val_y=yva.astype(np.int8),
                        val_proba=pv.astype(np.float32), test_ids=idte.astype(str), test_y=yte.astype(np.int8),
                        test_proba=pt.astype(np.float32), test_stream=ste.astype(str), prior=prior)
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    log("CT-GBDT", chosen, "seed", seed, rule, {m: round(test[m], 4) for m in ("macro_mcc", "macro_f1", "accuracy", "mcc")})
log("context gbdt kernel done")
# Cleaned-label ablation gate (pre-registered): the branch refitted on cleaned labels must beat the frozen standard
# CT-GBDT v4 (attached as input) by >= +0.002 macro-MCC on the SAME cleaned validation rows before the graph branch is
# retrained on cleaned labels (GPU). Validation only; test labels are not used here.
if globals().get("COMPARE_TO_STANDARD"):
    std_files = sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))
    ids_c = idva.astype(str)
    P_clean = np.mean([np.load(os.path.join(OUT, f"ctgbdt_seed{s}.npz"), allow_pickle=False)["val_proba"].astype(np.float64)
                       for s in SEEDS], 0)
    P_std = []
    for path in std_files:
        z = np.load(path, allow_pickle=False)
        sid = z["val_ids"].astype(str)
        o = np.argsort(sid)
        pos = np.searchsorted(sid[o], ids_c)
        assert np.all(pos < len(sid)) and np.array_equal(sid[o][pos], ids_c), "standard branch lacks cleaned validation rows"
        P_std.append(z["val_proba"][o][pos].astype(np.float64))
        std_prior = np.asarray(z["prior"], dtype=np.float64)
    P_std = np.mean(P_std, 0)

    def val_macro(P, pr):
        return {r: cm_summary(np.bincount(yva * K + f(P), minlength=K * K).reshape(K, K))["macro_mcc"]
                for r, f in (("bayes", lambda p: p.argmax(1)), ("balanced", lambda p: (p / np.maximum(pr, 1e-12)).argmax(1)))}

    s_std, s_clean = val_macro(P_std, std_prior), val_macro(P_clean, prior)
    gate = {"standard_files": std_files, "cleaned_validation_rows": int(len(ids_c)), "standard_branch": s_std,
            "cleaned_label_branch": s_clean, "gain": max(s_clean.values()) - max(s_std.values())}
    gate["helps"] = bool(gate["gain"] >= 0.002)
    report["cleaned_label_gate"] = gate
    json.dump(report, open(os.path.join(OUT, "ctgbdt.json"), "w"), indent=1)
    print("CLEANED-LABEL GATE", json.dumps(gate, indent=1, default=float))

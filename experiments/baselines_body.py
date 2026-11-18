# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# Body of the CPU baselines kernel (included by build_runs.py after COMMON; PREPARED_NAME is defined there).
# Tabular baselines on the EXACT prepared split used by the graph model: fit on train, early stopping
# and decision-rule choice on validation, alarm threshold on calibration benign rows, test scored once.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))  # noqa: F821
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))  # noqa: F821
manifest = json.load(open(os.path.join(root, "manifest.json")))  # noqa: F821
prep = json.load(open(os.path.join(root, "preprocess.json")))  # noqa: F821
classes = prep["classes"]
K = len(classes)
OUT = "/data/output/baselines"
os.makedirs(OUT, exist_ok=True)  # noqa: F821


def shard_path(split, shard):
    path = shard["path"]
    return path if os.path.exists(path) else os.path.join(root, "prepared", split, os.path.basename(path))  # noqa: F821


def load(split, ids=None):
    xs, ys, streams = [], [], []
    width = len(prep["means"])
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=["numeric", "categories", "label", "stream", "source_id"])
        if ids is not None:
            ids.append(np.asarray(t["source_id"].to_pylist(), dtype=object))
        numeric = t["numeric"].combine_chunks()
        x = numeric.values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
        cats = []
        for j, vocab in enumerate(prep["vocabs"]):
            values = pc.list_element(t["categories"], j)
            index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
            code = pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab)))
            cats.append(code.to_numpy(zero_copy_only=False).astype(np.float32))
        xs.append(np.concatenate([x, np.column_stack(cats)], 1))
        ys.append(pc.fill_null(pc.index_in(t["label"], value_set=pa.array(classes)), -1).to_numpy(zero_copy_only=False).astype(np.int64))
        streams.append(np.asarray(t["stream"].to_pylist(), dtype=object))
    x, y, s = np.concatenate(xs), np.concatenate(ys), np.concatenate(streams)
    keep = y >= 0
    if ids is not None:
        ids[:] = [np.concatenate(ids)[keep]]
    return x[keep], y[keep], s[keep]


def row_hash(x):
    return pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(x, nan=-7.77e30)), index=False).to_numpy()


def cm_summary(cm):
    t, p = cm.sum(1).astype(float), cm.sum(0).astype(float)
    s, c = cm.sum(), np.trace(cm)
    tp = np.diag(cm).astype(float)
    fn, fp = t - tp, p - tp
    tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_k = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    f1_k = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
    prec = np.divide(tp, p, out=np.zeros_like(tp), where=p > 0)
    rec = np.divide(tp, t, out=np.zeros_like(tp), where=t > 0)
    present = t > 0
    den_all = np.sqrt((s * s - (p * p).sum()) * (s * s - (t * t).sum()))
    return {"macro_mcc": float(mcc_k[present].mean()), "macro_f1": float(f1_k[present].mean()), "accuracy": float(c / s),
            "mcc": float((c * s - (p * t).sum()) / den_all) if den_all > 0 else 0.0,
            "weighted_f1": float((f1_k * t).sum() / t.sum()), "balanced_accuracy": float(rec[present].mean()),
            "detection_rate": float(cm[1:, 1:].sum() / max(cm[1:].sum(), 1)),
            "false_alarm_rate": float(cm[0, 1:].sum() / max(cm[0].sum(), 1)),
            "per_class": {classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1_k[i]),
                                       "mcc": float(mcc_k[i]), "support": int(t[i])} for i in range(len(t))}}


def bootstrap(y, pred, streams, draws=2000):
    names, index = np.unique(streams, return_inverse=True)
    cms = np.bincount(index * K * K + y * K + pred, minlength=len(names) * K * K).reshape(len(names), K, K)
    rng = np.random.default_rng(42)
    samples = [cm_summary(cms[rng.integers(0, len(names), len(names))].sum(0)) for _ in range(draws)]
    return {m: [float(np.percentile([d[m] for d in samples], 2.5)), float(np.percentile([d[m] for d in samples], 97.5))]
            for m in ("macro_mcc", "macro_f1", "accuracy", "mcc")}, int(len(names))


def binary_scores(y, proba):
    from sklearn.metrics import average_precision_score, roc_auc_score
    target, score = (y != 0).astype(int), 1 - proba[:, 0]
    return {"binary_roc_auc": float(roc_auc_score(target, score)), "binary_pr_auc": float(average_precision_score(target, score))}


log("loading prepared splits")  # noqa: F821
Xtr, ytr, _ = load("train")
Xva, yva, _ = load("val")
Xca, yca, _ = load("calibration")
test_ids = []
Xte, yte, ste = load("test", test_ids)
test_ids = test_ids[0]
log("rows", {"train": len(ytr), "val": len(yva), "calibration": len(yca), "test": len(yte)})  # noqa: F821
usable = ~np.all(np.isnan(Xtr), 0)
Xtr, Xva, Xca, Xte = Xtr[:, usable], Xva[:, usable], Xca[:, usable], Xte[:, usable]
novel = ~np.isin(row_hash(Xte), np.unique(row_hash(Xtr)))
log("test rows whose feature vector never occurs in train", float(novel.mean()))  # noqa: F821
if globals().get("LABEL_CONTROL"):
    # Leakage sanity control: training labels randomly permuted (marginals kept, feature-label link broken).
    ytr = np.random.default_rng(7).permutation(ytr)
    OUT = "/data/output/baselines_label_control"
    os.makedirs(OUT, exist_ok=True)
prior = np.bincount(ytr, minlength=K) / len(ytr)
results = {"protocol": manifest["split_plan"].get("scope"), "block_s": manifest["split_plan"].get("block_s"),
           "sizes": manifest["sizes"], "classes": classes, "features_used": int(usable.sum()),
           "test_novel_fraction": float(novel.mean()),
           "protocol_note": "fit on train; early stopping and decision rule on validation; threshold on calibration; test once"}


def finish(name, proba_val, proba_cal, proba_test, extra):
    # Decision rule on validation only: posterior argmax (bayes) vs prior-divided argmax (balanced).
    rules = {"bayes": lambda p: p.argmax(1), "balanced": lambda p: (p / np.maximum(prior, 1e-12)).argmax(1)}
    val_scores = {r: cm_summary(np.bincount(yva * K + f(proba_val), minlength=K * K).reshape(K, K))["macro_mcc"]
                  for r, f in rules.items()}
    rule = max(val_scores, key=val_scores.get)
    pred = rules[rule](proba_test)
    cm = np.bincount(yte * K + pred, minlength=K * K).reshape(K, K)
    report = cm_summary(cm)
    ci, segments = bootstrap(yte, pred, ste)
    other = "balanced" if rule == "bayes" else "bayes"
    alt = cm_summary(np.bincount(yte * K + rules[other](proba_test), minlength=K * K).reshape(K, K))
    cal_benign = np.sort(1 - proba_cal[yca == 0, 0])
    k = int(np.ceil((len(cal_benign) + 1) * 0.999))
    threshold = float(cal_benign[k - 1]) if k <= len(cal_benign) else None
    score = 1 - proba_test[:, 0]
    operating = ({"alpha": 0.001, "threshold": threshold, "fpr": float((score[yte == 0] > threshold).mean()),
                  "tpr": float((score[yte != 0] > threshold).mean())} if threshold is not None else None)
    novel_cm = np.bincount(yte[novel] * K + pred[novel], minlength=K * K).reshape(K, K)
    results[name] = {**report, **binary_scores(yte, proba_test), "decision_rule": rule, "validation_macro_mcc": val_scores,
                     "ci95_segment_bootstrap": ci, "test_segments": segments, "operating_point": operating,
                     "alternative_rule": {m: alt[m] for m in ("macro_mcc", "macro_f1", "accuracy", "mcc")},
                     "novel_feature_vector_subset": {**{m: v for m, v in cm_summary(novel_cm).items() if m != "per_class"},
                                                     "rows": int(novel.sum())},
                     "confusion_matrix": cm.tolist(), **extra}
    json.dump(results, open(os.path.join(OUT, "baselines.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(OUT, f"{name}_test_predictions.npz"), source_id=test_ids.astype(str),
                        label=yte.astype(np.int8), prediction=pred.astype(np.int8), benign_probability=proba_test[:, 0].astype(np.float32),
                        novel=novel, stream=ste.astype(str))
    log(name, rule, {m: round(report[m], 4) for m in ("macro_mcc", "macro_f1", "accuracy", "mcc")}, "CI", ci)  # noqa: F821


import lightgbm as lgb  # noqa: E402

started = time.time()  # noqa: F821
params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": 42, "max_bin": 255}  # noqa: F821
booster = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=1000, valid_sets=[lgb.Dataset(Xva, yva)],
                    callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(50)])
fit = time.time() - started  # noqa: F821
t0 = time.time()  # noqa: F821
proba_test = booster.predict(Xte, num_iteration=booster.best_iteration)
infer = time.time() - t0  # noqa: F821
finish("LightGBM", booster.predict(Xva, num_iteration=booster.best_iteration),
       booster.predict(Xca, num_iteration=booster.best_iteration), proba_test,
       {"trees_per_class": booster.best_iteration, "fit_seconds": fit, "test_inference_seconds": infer,
        "train_rows": int(len(ytr)), "params": params})
del booster
if globals().get("LABEL_CONTROL"):
    log("label-randomised control done")
    raise SystemExit(0)

import xgboost as xgb  # noqa: E402

started = time.time()  # noqa: F821
dtrain = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256)
dval = xgb.QuantileDMatrix(Xva, yva, ref=dtrain)
xparams = {"objective": "multi:softprob", "num_class": K, "tree_method": "hist", "max_depth": 8, "eta": 0.1,
           "subsample": 0.8, "colsample_bytree": 0.9, "min_child_weight": 5, "eval_metric": "mlogloss",
           "nthread": os.cpu_count() or 4, "seed": 42}  # noqa: F821
model = xgb.train(xparams, dtrain, num_boost_round=600, evals=[(dval, "val")], early_stopping_rounds=40, verbose_eval=50)
fit = time.time() - started  # noqa: F821
best = (0, model.best_iteration + 1)
t0 = time.time()  # noqa: F821
proba_test = model.predict(xgb.DMatrix(Xte), iteration_range=best)
infer = time.time() - t0  # noqa: F821
finish("XGBoost", model.predict(xgb.DMatrix(Xva), iteration_range=best), model.predict(xgb.DMatrix(Xca), iteration_range=best),
       proba_test, {"rounds": best[1], "fit_seconds": fit, "test_inference_seconds": infer, "params": xparams})
log("baselines kernel done")  # noqa: F821

# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log, PREPARED_NAME)
# FORMAL DEGRADATION ANALYSIS of the proposed model on VALIDATION data (train labels only for the irreducible-entropy
# lookup; the test split is never opened). Per class c:
#   Information theory   H(Y|X) (identical-vector label distributions from TRAIN), Fano bound, irreducible
#                        class-conditional entropy, "entropic bleed" 1 - E[p_c(x)|Y=c] and its destinations,
#                        H(Yhat|Y=c), I(Y;Yhat), excess cross-entropy = CE_c - irreducible entropy.
#   Decision boundary    log-odds margins m = log p_y - max_{k!=y} log p_k, boundary-proximal mass, flip displacement,
#   / loss landscape     seed disagreement (BALD mutual information), epoch-to-epoch F1 volatility and collapses.
#   Bayesian decision    EM label-shift prior (Saerens et al. 2002) vs train prior, optimal one-vs-rest log-odds
#                        threshold shift with a block-bootstrap 95% CI, per-class expected calibration error.
#   Branch attribution   graph / context-GBDT / fusion per class, oracle bound, branch deletion, cross-fitted
#                        class-conditional mixture-of-experts gate, cross-fitted per-class bias head.
# Every class receives a mechanism diagnosis mapped to an architectural remedy.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = "/data/output/formal"
os.makedirs(OUT, exist_ok=True)
EPS = 1e-12
rng = np.random.default_rng(20260926)


def read_validation(directory):
    files = sorted(glob.glob(os.path.join(directory, "validation", "predictions-rank*.parquet")))
    t = pa.concat_tables([pq.read_table(p, columns=["source_id", "stream", "first", "label", "probabilities"]) for p in files])
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object)
    o = np.argsort(ids)
    proba = np.stack(t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1))
    return {"ids": ids[o], "stream": np.asarray(t["stream"].to_pylist(), dtype=object)[o], "first": t["first"].to_numpy()[o],
            "y": t["label"].to_numpy()[o], "proba": proba[o].astype(np.float64)}


runs = sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True))
runs = [p for p in runs if glob.glob(os.path.join(os.path.dirname(p), "validation", "predictions-rank*.parquet"))]
results = [json.load(open(p)) for p in runs]
classes = results[0]["classes"]
K = len(classes)
block_s = float(results[0].get("split_plan", {}).get("block_s") or 600.0)
graph = [read_validation(os.path.dirname(p)) for p in runs]
ref = graph[0]
for g in graph[1:]:
    assert np.array_equal(g["ids"], ref["ids"]), "graph runs cover different validation rows"
keep = ref["y"] >= 0
ids, y = ref["ids"][keep], ref["y"][keep].astype(np.int64)
G_seeds = np.stack([g["proba"][keep] for g in graph])
G = G_seeds.mean(0)
units = np.asarray([f"{s}|{int(np.floor(f / block_s))}" for s, f in zip(ref["stream"][keep], ref["first"][keep])], dtype=object)
unit_names, unit_idx = np.unique(units, return_inverse=True)
counts_train = np.asarray([results[0].get("split_class_counts", {}).get("train", {}).get(c, 0) for c in classes], dtype=np.float64)
prior = counts_train / counts_train.sum()


def aligned(values, source_ids):
    o = np.argsort(source_ids)
    position = np.searchsorted(source_ids[o], ids)
    assert np.all(position < len(source_ids)) and np.array_equal(source_ids[o][position], ids)
    return values[o][position]


ct_files = sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))
C = np.mean([aligned(np.load(p)["val_proba"].astype(np.float64), np.load(p)["val_ids"].astype(object)) for p in ct_files], 0) if ct_files else None
report = {"prepared": PREPARED_NAME, "classes": classes, "graph_runs": runs, "graph_seeds": len(graph),
          "context_branch_seeds": len(ct_files), "validation_rows": int(len(y)), "block_units": int(len(unit_names)),
          "protocol": "validation predictions + train labels for the irreducible-entropy lookup; test never opened"}
log("rows", len(y), "graph seeds", len(graph), "ctgbdt seeds", len(ct_files))


def save():
    json.dump(report, open(os.path.join(OUT, "formal.json"), "w"), indent=1, default=float)


def confusion(yt, yp):
    return np.bincount(yt * K + yp, minlength=K * K).reshape(K, K).astype(np.float64)


def per_class_scores(cm):
    t, p = cm.sum(1), cm.sum(0)
    s = cm.sum()
    tp = np.diag(cm)
    fn, fp = t - tp, p - tp
    tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(K), where=(2 * tp + fp + fn) > 0)
    prec = np.divide(tp, p, out=np.zeros(K), where=p > 0)
    rec = np.divide(tp, t, out=np.zeros(K), where=t > 0)
    return mcc, f1, prec, rec, t > 0


def macro_mcc(yt, yp):
    mcc, _, _, _, present = per_class_scores(confusion(yt, yp))
    return float(mcc[present].mean())


rules = {"bayes": lambda P: P.argmax(1), "balanced": lambda P: (P / np.maximum(prior, EPS)).argmax(1)}


def best_rule(P):
    return max(rules, key=lambda r: macro_mcc(y, rules[r](P)))


# ---------------------------------------------------------------------------- fusion as currently deployed
if C is not None:
    grid = [(macro_mcc(y, rules[r](w * G + (1 - w) * C)), w, r) for w in np.round(np.linspace(0, 1, 11), 2) for r in rules]
    _, w_star, r_star = max(grid, key=lambda t: (t[0], -abs(t[1] - 0.5)))
    F = w_star * G + (1 - w_star) * C
    report["deployed_fusion"] = {"graph_weight": w_star, "rule": r_star}
else:
    F, r_star = G, best_rule(G)
branches = {"graph": (G, best_rule(G)), "fusion": (F, r_star)}
if C is not None:
    branches["context_gbdt"] = (C, best_rule(C))
pred = {name: rules[r](P) for name, (P, r) in branches.items()}

# ---------------------------------------------------------------------------- A. information theory
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
width = len(prep["means"])
label_set = pa.array(classes)


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


def hashes(x):
    return pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(x, nan=-7.77e30)), index=False).to_numpy()


pairs = []
for shard in manifest["shards"]["train"]:
    x, lab = matrix(pq.read_table(shard_path("train", shard), columns=["numeric", "categories", "label"]))
    ok = lab >= 0
    pairs.append(pd.DataFrame({"h": hashes(x)[ok], "y": lab[ok]}).value_counts().reset_index())
table = pd.concat(pairs).groupby(["h", "y"], sort=False)["count"].sum().reset_index().pivot_table(
    index="h", columns="y", values="count", fill_value=0).reindex(columns=range(K), fill_value=0)
th, tc = table.index.to_numpy(), table.to_numpy(np.float64)
del pairs, table
vh, vid = [], []
for shard in manifest["shards"]["val"]:
    t = pq.read_table(shard_path("val", shard), columns=["numeric", "categories", "label", "source_id"])
    x, lab = matrix(t)
    vh.append(hashes(x)[lab >= 0])
    vid.append(np.asarray(t["source_id"].to_pylist(), dtype=object)[lab >= 0])
val_hash = aligned(np.concatenate(vh), np.concatenate(vid))
pos = np.minimum(np.searchsorted(th, val_hash), len(th) - 1)
seen = th[pos] == val_hash
dist = np.where(seen[:, None], tc[pos], 0.0)
dist = dist / np.maximum(dist.sum(1, keepdims=True), EPS)
H_row = -np.sum(np.where(dist > 0, dist * np.log2(np.maximum(dist, EPS)), 0.0), 1)  # H(Y|x) on seen vectors, bits
bayes_err_row = 1 - dist.max(1)
p_true_row = dist[np.arange(len(y)), y]  # train frequency of the true label on this exact vector
H_Y = float(-(np.bincount(y, minlength=K) / len(y) * np.log2(np.maximum(np.bincount(y, minlength=K) / len(y), EPS))).sum())
H_YX = float(H_row[seen].mean()) if seen.any() else None
fano = max(0.0, (H_YX - 1) / np.log2(K - 1)) if H_YX is not None and K > 2 else None
info = {"H_Y_bits": H_Y, "H_Y_given_X_flow_bits_on_seen": H_YX, "seen_fraction": float(seen.mean()),
        "fano_lower_bound_on_per_flow_error": fano, "per_flow_bayes_error_on_seen": float(bayes_err_row[seen].mean()) if seen.any() else None}
for name, (P, r) in branches.items():
    cm = confusion(y, pred[name])
    pj = cm / cm.sum()
    py, pyh = pj.sum(1, keepdims=True), pj.sum(0, keepdims=True)
    nz = pj > 0
    info[f"I_Y_Yhat_bits_{name}"] = float((pj[nz] * np.log2(pj[nz] / (py @ pyh)[nz])).sum())
per_class = {}
for c in range(K):
    m = y == c
    if not m.any():
        continue
    row = {"support": int(m.sum()), "seen_fraction": float(seen[m].mean()),
           "irreducible_entropy_bits": float(H_row[m & seen].mean()) if (m & seen).any() else None,
           "forced_error_rate": float((1 - p_true_row[m & seen]).mean()) if (m & seen).any() else None}
    for name, (P, r) in branches.items():
        pc_ = P[m, c]
        leak = P[m].mean(0)
        leak[c] = 0
        top = np.argsort(-leak)[:3]
        yh = pred[name][m]
        dist_hat = np.bincount(yh, minlength=K) / len(yh)
        ce = -np.log2(np.maximum(pc_, EPS))
        row[name] = {"entropic_bleed": float(1 - pc_.mean()),
                     "bleed_to": [(classes[k], round(float(leak[k]), 4)) for k in top if leak[k] > 0],
                     "H_Yhat_given_Y_bits": float(-(dist_hat[dist_hat > 0] * np.log2(dist_hat[dist_hat > 0])).sum()),
                     "cross_entropy_bits": float(ce.mean()),
                     "excess_cross_entropy_bits": float(ce[seen[m]].mean() - H_row[m & seen].mean()) if (m & seen).any() else None}
    per_class[classes[c]] = row
log("A done", info)

# ---------------------------------------------------------------------------- B. decision boundary / loss landscape
logF = np.log(np.maximum(F, EPS))
if r_star == "balanced":
    logF = logF - np.log(np.maximum(prior, EPS))
true_logit = logF[np.arange(len(y)), y]
masked = logF.copy()
masked[np.arange(len(y)), y] = -np.inf
margin = true_logit - masked.max(1)
Hs = -(G_seeds * np.log(np.maximum(G_seeds, EPS))).sum(2)
bald = -(G * np.log(np.maximum(G, EPS))).sum(1) - Hs.mean(0)
seed_preds = G_seeds.argmax(2)
disagree = (seed_preds != seed_preds[0]).any(0)
histories = []
for f in glob.glob("/data/input/**/best.training.json", recursive=True):
    try:
        histories.append([[p.get("f1", 0.0) for p in e["validation"]["per_class"]] for e in json.load(open(f))["epochs"]])
    except Exception:
        pass
for c in range(K):
    if classes[c] not in per_class:
        continue
    m = y == c
    mg = margin[m]
    wrong = mg < 0
    volatility, collapses = [], 0
    for h in histories:
        f1 = np.asarray([e[c] for e in h if len(e) > c])
        if len(f1) > 1:
            volatility.append(float(f1.std()))
            collapses += int(np.sum(np.diff(f1) < -0.3))
    per_class[classes[c]]["boundary"] = {
        "margin_q10_q50_q90": [float(np.quantile(mg, q)) for q in (0.1, 0.5, 0.9)],
        "misclassified_fraction": float(wrong.mean()), "boundary_proximal_fraction_abs_margin_lt_0.5": float((np.abs(mg) < 0.5).mean()),
        "median_flip_displacement_log_odds": float(np.median(-mg[wrong])) if wrong.any() else 0.0,
        "seed_bald_bits": float(bald[m].mean() / np.log(2)), "seed_disagreement": float(disagree[m].mean()),
        "epoch_f1_std_mean": float(np.mean(volatility)) if volatility else None, "epoch_collapses": collapses}
log("B done")

# ---------------------------------------------------------------------------- C. Bayesian decision theory
def em_prior(P, train_prior, iters=100):
    """Saerens, Latinne & Decaestecker (2002): maximum-likelihood target prior under label shift."""
    q = train_prior.copy()
    for _ in range(iters):
        w = P * (q / np.maximum(train_prior, EPS))
        w /= np.maximum(w.sum(1, keepdims=True), EPS)
        q_new = w.mean(0)
        if np.abs(q_new - q).max() < 1e-7:
            break
        q = q_new
    return q


true_val_prior = np.bincount(y, minlength=K) / len(y)
q_em = em_prior(G, prior)
report["C_label_shift"] = {"train_prior": dict(zip(classes, prior.round(6).tolist())),
                           "validation_prior": dict(zip(classes, true_val_prior.round(6).tolist())),
                           "em_estimated_validation_prior_from_graph": dict(zip(classes, q_em.round(6).tolist())),
                           "kl_val_vs_train_bits": float(np.sum(np.where(true_val_prior > 0, true_val_prior * np.log2(np.maximum(true_val_prior, EPS) / np.maximum(prior, EPS)), 0)))}
shifts = np.arange(-4.0, 4.01, 0.25)
for c in range(K):
    if classes[c] not in per_class:
        continue
    others = np.delete(logF, c, axis=1).max(1)
    score = logF[:, c] - others  # one-vs-rest log-odds of the deployed decision
    target = (y == c)

    # Per-unit confusion counts for every shift: the block bootstrap then only sums unit rows (fast, exact).
    U = len(unit_names)
    tp_u = np.zeros((len(shifts), U)); fp_u = np.zeros((len(shifts), U)); fn_u = np.zeros((len(shifts), U))
    n_u = np.bincount(unit_idx, minlength=U).astype(np.float64)
    for i, d in enumerate(shifts):
        p = score + d > 0
        tp_u[i] = np.bincount(unit_idx, weights=(p & target), minlength=U)
        fp_u[i] = np.bincount(unit_idx, weights=(p & ~target), minlength=U)
        fn_u[i] = np.bincount(unit_idx, weights=(~p & target), minlength=U)

    def mcc_from(tp, fp, fn, n):
        tn = n - tp - fp - fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return np.divide(tp * tn - fp * fn, den, out=np.zeros_like(tp), where=den > 0)

    curve = mcc_from(tp_u.sum(1), fp_u.sum(1), fn_u.sum(1), np.full(len(shifts), n_u.sum()))
    best = float(shifts[curve.argmax()])
    draws = rng.integers(0, U, (200, U))
    counts_ = np.stack([np.bincount(d, minlength=U) for d in draws]).astype(np.float64)  # multiplicity of each unit
    boot_curves = mcc_from(counts_ @ tp_u.T, counts_ @ fp_u.T, counts_ @ fn_u.T, (counts_ @ n_u)[:, None])
    boot = shifts[boot_curves.argmax(1)]
    conf = F[:, c]
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(target[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    per_class[classes[c]]["bayes_decision"] = {"ovr_mcc_at_zero_shift": float(curve[np.argmin(np.abs(shifts))]),
                                               "ovr_mcc_at_optimal_shift": float(curve.max()), "optimal_log_odds_shift": best,
                                               "shift_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                               "ece": float(ece), "prior_train": float(prior[c]), "prior_val": float(true_val_prior[c])}
log("C done")
save()

# ---------------------------------------------------------------------------- D. branch attribution, deletion, gates
fold = rng.permutation(len(unit_names))[unit_idx] % 5
scores = {}
for name in pred:
    mcc, f1, prec, rec, present = per_class_scores(confusion(y, pred[name]))
    scores[name] = {"macro_mcc": float(mcc[present].mean()), "f1": f1, "precision": prec, "recall": rec}
for c in range(K):
    if classes[c] in per_class:
        per_class[classes[c]]["branch_f1"] = {n: round(float(s["f1"][c]), 4) for n, s in scores.items()}
        per_class[classes[c]]["branch_precision_recall"] = {n: [round(float(s["precision"][c]), 4), round(float(s["recall"][c]), 4)] for n, s in scores.items()}
branch = {"macro_mcc": {n: s["macro_mcc"] for n, s in scores.items()}}
if C is not None:
    oracle = (pred["graph"] == y) | (pred["context_gbdt"] == y)
    branch["oracle_either_branch_accuracy"] = float(oracle.mean())

    def fit_gate(idx):
        """Class-conditional mixture of experts: p(k) ~ w_k g_k + (1 - w_k) c_k, weights by coordinate ascent."""
        w = np.full(K, w_star)
        for _ in range(2):
            for k in range(K):
                best_ = (-1, w[k])
                for v in np.round(np.linspace(0, 1, 11), 2):
                    w2 = w.copy(); w2[k] = v
                    P = w2 * G[idx] + (1 - w2) * C[idx]
                    val = macro_mcc(y[idx], rules[r_star](P / P.sum(1, keepdims=True)))
                    if val > best_[0]:
                        best_ = (val, v)
                w[k] = best_[1]
        return w

    def fit_bias(idx):
        b = np.zeros(K)
        S = logF[idx]
        for _ in range(2):
            for k in range(K):
                trial = []
                for v in np.arange(-2.0, 2.01, 0.25):
                    b2 = b.copy(); b2[k] = v
                    trial.append((macro_mcc(y[idx], (S + b2).argmax(1)), -abs(v), v))
                b[k] = max(trial)[2]
        return b

    oof_gate, oof_bias, oof_global = np.zeros(len(y), np.int64), np.zeros(len(y), np.int64), np.zeros(len(y), np.int64)
    gate_weights = []
    for f in range(5):
        tr, te = fold != f, fold == f
        if not te.any():
            continue
        w = fit_gate(np.flatnonzero(tr))
        gate_weights.append(w.tolist())
        P = w * G[te] + (1 - w) * C[te]
        oof_gate[te] = rules[r_star](P / P.sum(1, keepdims=True))
        oof_bias[te] = (logF[te] + fit_bias(np.flatnonzero(tr))).argmax(1)
        wg = max(((macro_mcc(y[tr], rules[r](wt * G[tr] + (1 - wt) * C[tr])), wt, r) for wt in np.round(np.linspace(0, 1, 11), 2) for r in rules))
        oof_global[te] = rules[wg[2]](wg[1] * G[te] + (1 - wg[1]) * C[te])
    branch["cross_fitted_oof_macro_mcc"] = {"global_weight_fusion": macro_mcc(y, oof_global),
                                            "class_conditional_gate": macro_mcc(y, oof_gate),
                                            "per_class_bias_head": macro_mcc(y, oof_bias)}
    branch["gate_weights_per_fold"] = gate_weights
    branch["deletion"] = {"delete_context_branch (graph only)": scores["graph"]["macro_mcc"],
                          "delete_graph_branch (context only)": scores["context_gbdt"]["macro_mcc"],
                          "keep_both (deployed fusion)": scores["fusion"]["macro_mcc"]}
report["D_branches"] = branch
log("D done", json.dumps(branch.get("cross_fitted_oof_macro_mcc", {})))

# ---------------------------------------------------------------------------- E. mechanism diagnosis -> remedy
remedies = {"irreducible": "per-flow Bayes-limited: only context separates it -> graph/context branch, never a per-flow fix",
            "threshold": "decision threshold misplaced (significant optimal log-odds shift) -> cross-fitted per-class bias head",
            "branch_selection": "a branch is better than the fusion for this class -> class-conditional gate between branches",
            "variance": "unstable across seeds/epochs -> more seeds, weight averaging, class-balanced replay",
            "capacity": "per-flow learnable (trees succeed) but the graph fails -> numeric embeddings (PLE) / distillation",
            "context_needed": "graph succeeds where per-flow trees fail -> raise graph weight for this class",
            "unlearned": "low recall with no irreducible or threshold explanation -> more data/rehearsal; likely out of distribution"}
diagnosis = {}
for name, row in per_class.items():
    tags = []
    b, d = row["boundary"], row["bayes_decision"]
    if (row["forced_error_rate"] or 0) > 0.2 or (row["irreducible_entropy_bits"] or 0) > 0.5:
        tags.append("irreducible")
    if (d["shift_ci95"][0] > 0 or d["shift_ci95"][1] < 0) and d["ovr_mcc_at_optimal_shift"] - d["ovr_mcc_at_zero_shift"] > 0.05:
        tags.append("threshold")
    bf = row["branch_f1"]
    if max(bf.values()) - bf["fusion"] > 0.05:
        tags.append("branch_selection")
    if b["seed_disagreement"] > 0.1 or (b["epoch_f1_std_mean"] or 0) > 0.2 or b["epoch_collapses"] > 1:
        tags.append("variance")
    if "context_gbdt" in bf and bf["context_gbdt"] - bf["graph"] > 0.3:
        tags.append("capacity")
    if "context_gbdt" in bf and bf["graph"] - bf["context_gbdt"] > 0.2:
        tags.append("context_needed")
    if bf["fusion"] < 0.5 and not tags:
        tags.append("unlearned")
    diagnosis[name] = {"mechanisms": tags, "remedies": [remedies[t] for t in tags], "fusion_f1": bf["fusion"]}
report["A_information"] = info
report["per_class"] = per_class
report["E_diagnosis"] = diagnosis
save()
print(json.dumps({"A_information": info, "C_label_shift_kl": report["C_label_shift"]["kl_val_vs_train_bits"],
                  "D_branches": {k: v for k, v in branch.items() if k != "gate_weights_per_fold"}}, indent=1, default=float))
for name, dgn in diagnosis.items():
    row = per_class[name]
    print(f"{name:26s} F1 {dgn['fusion_f1']:.3f} | irreducible H {row['irreducible_entropy_bits']} forced {row['forced_error_rate']} | "
          f"bleed {row['fusion']['entropic_bleed']:.3f} -> {row['fusion']['bleed_to'][:2]} | shift {row['bayes_decision']['optimal_log_odds_shift']} "
          f"CI {row['bayes_decision']['shift_ci95']} | seedDis {row['boundary']['seed_disagreement']:.3f} | {dgn['mechanisms']}")
log("formal analysis done")

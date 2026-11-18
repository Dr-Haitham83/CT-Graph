# flake8: noqa -- executed after the COMMON prelude of build_runs.py (glob, os, json, time, log)
# Post-hoc comparison on the frozen test split. No model is refitted and nothing is selected on test:
# decision rules come from validation (single runs: training; ensembles: validation predictions).
#
# Dependence-aware inference. Test rows share graph state and temporal context, so rows are not
# independent. Resampling units are (stream, time block of the split's own block length): the protocol
# already treats such blocks as separable up to an embargo. Segment-level results are reported too.
#   * paired percentile bootstrap of metric differences over units (2000 draws);
#   * paired sign-flip randomisation test over units (2000 draws, two-sided) for the null of
#     exchangeable model labels within every unit.
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OUT = os.environ.get("POSTHOC_OUT", "/data/output/posthoc")
INPUT = os.environ.get("POSTHOC_INPUT", "/data/input")
os.makedirs(OUT, exist_ok=True)
RESAMPLES = 2000
metrics = ("macro_mcc", "macro_f1", "accuracy", "mcc")


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
            "per_class_f1": f1_k.tolist(), "per_class_mcc": mcc_k.tolist()}


def unit_cms(y, pred, unit, units, k):
    return np.bincount(unit * k * k + y * k + pred, minlength=units * k * k).reshape(units, k, k)


def paired_inference(y, a, b, unit, k, seed=42):
    """Bootstrap CIs of metric(a) - metric(b) and a sign-flip randomisation p-value over units."""
    names, index = np.unique(unit, return_inverse=True)
    ca, cb = unit_cms(y, a, index, len(names), k), unit_cms(y, b, index, len(names), k)
    point = {m: cm_summary(ca.sum(0))[m] - cm_summary(cb.sum(0))[m] for m in metrics}
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in metrics}
    for _ in range(RESAMPLES):
        draw = rng.integers(0, len(names), len(names))
        sa, sb = cm_summary(ca[draw].sum(0)), cm_summary(cb[draw].sum(0))
        for m in metrics:
            boot[m].append(sa[m] - sb[m])
    null = {m: [] for m in metrics}
    for _ in range(RESAMPLES):
        flip = rng.random(len(names)) < 0.5
        sa = cm_summary(np.where(flip[:, None, None], cb, ca).sum(0))
        sb = cm_summary(np.where(flip[:, None, None], ca, cb).sum(0))
        for m in metrics:
            null[m].append(sa[m] - sb[m])
    return {"units": int(len(names)), "delta": point,
            "ci95": {m: [float(np.percentile(boot[m], 2.5)), float(np.percentile(boot[m], 97.5))] for m in metrics},
            "p_bootstrap_delta_le_0": {m: float(np.mean(np.asarray(boot[m]) <= 0)) for m in metrics},
            "p_randomisation_two_sided": {m: float((1 + np.sum(np.abs(null[m]) >= abs(point[m]) - 1e-12)) / (RESAMPLES + 1))
                                          for m in metrics}}


def read_predictions(directory, rule):
    tables = [pq.read_table(p, columns=["source_id", "stream", "first", "label", "probabilities",
                                        "balanced_prediction", "bayes_prediction"])
              for p in sorted(glob.glob(os.path.join(directory, "predictions-rank*.parquet")))]
    t = pa.concat_tables(tables)
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object)
    order = np.argsort(ids)
    probabilities = np.asarray(t["probabilities"].to_pylist(), dtype=np.float64)[order]
    return {"ids": ids[order], "stream": np.asarray(t["stream"].to_pylist(), dtype=object)[order],
            "first": t["first"].to_numpy()[order], "y": t["label"].to_numpy()[order],
            "pred": t[f"{rule}_prediction"].to_numpy()[order], "proba": probabilities}


runs = sorted(glob.glob(f"{INPUT}/**/evaluations/*/results.json", recursive=True))
if not runs:
    raise SystemExit("No graph evaluation found")
results = [json.load(open(p)) for p in runs]
classes = results[0]["classes"]
K = len(classes)
BEN_IDX = [c.lower() for c in classes].index("benign")
SCORES = {}  # attack scores (1 - P(benign)) for the binary ROC / PR AUC of the ALL_METRICS block
block_s = float(results[0].get("split_plan", {}).get("block_s") or 600.0)
graph_runs = []
for path, result in zip(runs, results):
    run = read_predictions(os.path.dirname(path), result["decision_rule"])
    run["validation"] = (read_predictions(os.path.join(os.path.dirname(path), "validation"), result["decision_rule"])
                         if glob.glob(os.path.join(os.path.dirname(path), "validation", "predictions-rank*.parquet")) else None)
    run["rule"], run["path"] = result["decision_rule"], path
    graph_runs.append(run)
base = graph_runs[0]
for run in graph_runs[1:]:
    if not np.array_equal(run["ids"], base["ids"]):
        raise SystemExit("Graph runs were evaluated on different test rows")
report = {"classes": classes, "graph_runs": [r["path"] for r in graph_runs], "block_unit_s": block_s,
          "inference": "paired bootstrap + sign-flip randomisation over (stream, block) units; segment level as well",
          "comparisons": {}}
y = base["y"]
keep = y >= 0
# Optional CLEANED-BENCHMARK ablation (the standard benchmark is always reported first, by the kernel without CLEAN_RULE):
# test rows listed by the cleaning kernel under the chosen rule are dropped before any metric is computed.
CLEAN = globals().get("CLEAN_RULE")
if CLEAN:
    clean_file = sorted(glob.glob(f"{INPUT}/**/clean_test_drop.npz", recursive=True))[0]
    dropped = np.isin(np.asarray(base["ids"], dtype=str), np.load(clean_file, allow_pickle=False)[CLEAN])
    report["cleaning"] = {"rule": CLEAN, "file": clean_file, "dropped_rows": int((dropped & keep).sum()),
                          "dropped_by_class": {classes[k]: int((dropped & (y == k)).sum()) for k in range(K) if (dropped & (y == k)).any()},
                          "kept_rows": int((keep & ~dropped).sum())}
    keep = keep & ~dropped
    log("cleaned benchmark", json.dumps(report["cleaning"]))
units = {"block": np.asarray([f"{s}|{int(np.floor(f / block_s))}" for s, f in zip(base["stream"], base["first"])], dtype=object),
         "segment": base["stream"]}
candidates = {f"graph_seed{i}": r["pred"] for i, r in enumerate(graph_runs)}
report["seed_runs"] = {name: {m: cm_summary(np.bincount(y[keep] * K + p[keep], minlength=K * K).reshape(K, K))[m] for m in metrics}
                       for name, p in candidates.items()}
if len(graph_runs) > 1:
    values = {m: [v[m] for v in report["seed_runs"].values()] for m in metrics}
    report["seed_mean_sd"] = {m: [float(np.mean(v)), float(np.std(v, ddof=1))] for m, v in values.items()}
    # Probability-averaged deep ensemble; decision rule chosen on the ensemble's VALIDATION predictions.
    if all(r["validation"] is not None for r in graph_runs):
        val = graph_runs[0]["validation"]
        train_counts = results[0].get("split_class_counts", {}).get("train", {})
        prior = np.asarray([train_counts.get(c, 0) for c in classes], dtype=np.float64)
        prior = prior / prior.sum() if prior.sum() else np.full(K, 1.0 / K)
        mean_val = np.mean([r["validation"]["proba"] for r in graph_runs], 0)
        mean_test = np.mean([r["proba"] for r in graph_runs], 0)
        rules = {"bayes": lambda p: p.argmax(1), "balanced": lambda p: (p / np.maximum(prior, 1e-12)).argmax(1)}
        vy = val["y"]
        vk = vy >= 0
        scores = {r: cm_summary(np.bincount(vy[vk] * K + f(mean_val)[vk], minlength=K * K).reshape(K, K))["macro_mcc"]
                  for r, f in rules.items()}
        rule = max(scores, key=scores.get)
        candidates["graph_ensemble"] = rules[rule](mean_test)
        SCORES["graph_ensemble"] = 1 - mean_test[:, BEN_IDX]
        report["ensemble"] = {"members": len(graph_runs), "rule_selected_on_validation": rule, "validation_macro_mcc": scores,
                              "test": {m: cm_summary(np.bincount(y[keep] * K + candidates["graph_ensemble"][keep],
                                                                 minlength=K * K).reshape(K, K))[m] for m in metrics}}
# CT-Graph hybrid: validation-weighted fusion of the graph branch (seed ensemble) and the context branch
# (CT-GBDT seeds). Weight and decision rule are chosen on VALIDATION predictions only.
ct_paths = sorted(glob.glob(f"{INPUT}/**/ctgbdt_seed*.npz", recursive=True))
if ct_paths and all(r["validation"] is not None for r in graph_runs):
    ct = [np.load(p, allow_pickle=False) for p in ct_paths]

    def aligned(ids_ref, ids, values):
        o = np.argsort(ids)
        position = np.searchsorted(ids[o], ids_ref)
        if np.any(position >= len(ids)) or not np.array_equal(ids[o][position], ids_ref):
            raise SystemExit("Context branch rows do not match the graph predictions")
        return values[o][position]

    gv = graph_runs[0]["validation"]
    vmask = gv["y"] >= 0
    # A context branch refitted on CLEANED labels (cleanprep) has validation predictions only for its labelled rows:
    # the fusion is then selected on the rows labelled for both branches (the cleaned validation set).
    ct_val_ids = [c["val_ids"].astype(str) for c in ct]
    if any(len(v) < int(vmask.sum()) for v in ct_val_ids):
        gids = np.asarray(gv["ids"], dtype=str)
        for v in ct_val_ids:
            vmask &= np.isin(gids, v)
        report["hybrid_validation_rows"] = {"graph_labelled": int((gv["y"] >= 0).sum()), "used": int(vmask.sum())}
        log("hybrid validation restricted to rows labelled for both branches", report["hybrid_validation_rows"])
    g_val = np.mean([r["validation"]["proba"] for r in graph_runs], 0)[vmask]
    g_test = np.mean([r["proba"] for r in graph_runs], 0)
    vids = gv["ids"][vmask]
    c_val = np.mean([aligned(vids, c["val_ids"].astype(object), c["val_proba"].astype(np.float64)) for c in ct], 0)
    c_seeds = [aligned(base["ids"], c["test_ids"].astype(object), c["test_proba"].astype(np.float64)) for c in ct]
    c_test = np.mean(c_seeds, 0)
    prior = np.asarray(ct[0]["prior"], dtype=np.float64)
    vy = gv["y"][vmask]
    rules = {"bayes": lambda p: p.argmax(1), "balanced": lambda p: (p / np.maximum(prior, 1e-12)).argmax(1)}
    vstreams = gv["stream"][vmask]
    seg_names, seg_idx = np.unique(vstreams, return_inverse=True)

    def macro(yv, pred):
        return cm_summary(np.bincount(yv * K + pred, minlength=K * K).reshape(K, K))["macro_mcc"]

    # Distributionally robust selection (pre-registered after the ToN development gap analysis): test campaigns are
    # NEW campaigns, so each validation segment (~ one campaign) is a draw from the campaign distribution and the
    # fusion is chosen to maximise CVaR_0.25 over segments (mean of the worst quarter of per-segment macro-MCC),
    # not the pooled validation score that a few large campaigns dominate (Rockafellar & Uryasev 2000; Duchi & Namkoong 2021).
    def seg_scores(yv, pred, sidx):
        out = []
        for s_ in np.unique(sidx):
            m_ = sidx == s_
            if m_.sum() >= 100 and len(np.unique(yv[m_])) >= 2:
                out.append(macro(yv[m_], pred[m_]))
        return np.asarray(out)

    def robust(yv, pred, sidx, alpha=0.25):
        scores = seg_scores(yv, pred, sidx)
        if not len(scores):
            return -1.0
        return float(np.sort(scores)[:max(1, int(np.ceil(alpha * len(scores))))].mean())

    grid, robust_grid = {}, {}
    for w in np.round(np.linspace(0, 1, 11), 2):
        fused = w * g_val + (1 - w) * c_val
        for rule, f in rules.items():
            pred_ = f(fused)
            grid[f"{w}|{rule}"] = macro(vy, pred_)
            robust_grid[f"{w}|{rule}"] = robust(vy, pred_, seg_idx)
    closeness = lambda k: -abs(float(k.split("|")[0]) - 0.5)  # noqa: E731
    pooled_best = max(grid, key=lambda k: (grid[k], closeness(k)))
    best = max(robust_grid, key=lambda k: (robust_grid[k], closeness(k)))
    w, rule = float(best.split("|")[0]), best.split("|")[1]
    candidates["ct_graph_hybrid"] = rules[rule](w * g_test + (1 - w) * c_test)
    pw, pr = float(pooled_best.split("|")[0]), pooled_best.split("|")[1]
    candidates["ct_graph_hybrid_pooled_selection"] = rules[pr](pw * g_test + (1 - pw) * c_test)
    er = max(rules, key=lambda r: robust_grid[f"0.5|{r}"])
    candidates["ct_graph_hybrid_equal_weight"] = rules[er](0.5 * g_test + 0.5 * c_test)
    if globals().get("RULE_DIAG"):
        # report-only diagnostic (revision): the equal-weight fusion under every decision rule; the selection above is unchanged
        for r_ in rules:
            candidates[f"diag_equal_weight_{r_}"] = rules[r_](0.5 * g_test + 0.5 * c_test)
    candidates["context_branch"] = rules[max(("bayes", "balanced"), key=lambda r: grid[f"0.0|{r}"])](c_test)
    SCORES["context_branch"] = 1 - c_test[:, BEN_IDX]
    # individual CT-GBDT seeds (same decision rule as the context-branch ensemble), for the ALL_METRICS block
    c_rule = rules[max(("bayes", "balanced"), key=lambda r: grid[f"0.0|{r}"])]
    for i_, cs in enumerate(c_seeds):
        candidates[f"context_seed{i_}"] = c_rule(cs)
        SCORES[f"context_seed{i_}"] = 1 - cs[:, BEN_IDX]
    SCORES["ct_graph_hybrid"] = 1 - (0.5 * g_test + 0.5 * c_test)[:, BEN_IDX]

    # Per-class decision bias (vector scaling of the fused log-scores) for the macro metric, adopted only if its
    # 5-fold block-cross-fitted VALIDATION macro-MCC beats the cross-fitted global-weight head (no test data used).
    def log_scores(wt, rl, g, c):
        s_ = np.log(np.maximum(wt * g + (1 - wt) * c, 1e-12))
        return s_ - np.log(np.maximum(prior, 1e-12)) if rl == "balanced" else s_

    def fit_global(idx):
        best_ = max(((robust(vy[idx], rules[rl](wt * g_val[idx] + (1 - wt) * c_val[idx]), seg_idx[idx]), -abs(wt - 0.5), wt, rl)
                     for wt in np.round(np.linspace(0, 1, 11), 2) for rl in rules))
        return best_[2], best_[3]

    def fit_bias(S, yv):
        b = np.zeros(K)
        for _ in range(2):
            for k in range(K):
                trial = []
                for v in np.arange(-2.0, 2.01, 0.25):
                    b2 = b.copy(); b2[k] = v
                    trial.append((macro(yv, (S + b2).argmax(1)), -abs(v), v))
                b[k] = max(trial)[2]
        return b

    def fit_gate(idx, wt, rl):
        """Class-conditional mixture of experts: p(k) ~ w_k g_k + (1 - w_k) c_k (weights by coordinate ascent)."""
        wv = np.full(K, wt)
        for _ in range(2):
            for k in range(K):
                best_ = (-1.0, wv[k])
                for v in np.round(np.linspace(0, 1, 11), 2):
                    w2 = wv.copy(); w2[k] = v
                    P = w2 * g_val[idx] + (1 - w2) * c_val[idx]
                    val_ = macro(vy[idx], rules[rl](P / P.sum(1, keepdims=True)))
                    if val_ > best_[0] + 1e-12:
                        best_ = (val_, v)
                wv[k] = best_[1]
        return wv

    vunits = np.asarray([f"{st}|{int(np.floor(fs / block_s))}" for st, fs in zip(gv["stream"][vmask], gv["first"][vmask])], dtype=object)
    # Folds group whole validation SEGMENTS (independent episodes; a campaign never straddles fit and score) when
    # there are at least five; otherwise (stream, block) units. Block folds would let a head fitted on part of a
    # campaign be scored on the rest of the same campaign and overstate its benefit.
    group_units = vstreams if len(np.unique(vstreams)) >= 5 else vunits
    uname, uidx = np.unique(group_units, return_inverse=True)
    vfold = np.random.default_rng(20260925).permutation(len(uname))[uidx] % 5
    oof_global, oof_bias, oof_gate = np.zeros(len(vy), np.int64), np.zeros(len(vy), np.int64), np.zeros(len(vy), np.int64)
    for f in range(5):
        tr_, te_ = vfold != f, vfold == f
        if not te_.any():
            continue
        wt, rl = fit_global(tr_)
        oof_global[te_] = rules[rl](wt * g_val[te_] + (1 - wt) * c_val[te_])
        b = fit_bias(log_scores(wt, rl, g_val[tr_], c_val[tr_]), vy[tr_])
        oof_bias[te_] = (log_scores(wt, rl, g_val[te_], c_val[te_]) + b).argmax(1)
        wv = fit_gate(np.flatnonzero(tr_), wt, rl)
        P = wv * g_val[te_] + (1 - wv) * c_val[te_]
        oof_gate[te_] = rules[rl](P / P.sum(1, keepdims=True))
    head = {"global_oof_macro_mcc": macro(vy, oof_global), "bias_oof_macro_mcc": macro(vy, oof_bias),
            "gate_oof_macro_mcc": macro(vy, oof_gate), "fold_unit": "segment" if group_units is vstreams else "block",
            "folds_groups": int(len(uname)), "criterion": "CVaR_0.25 over validation segments (out-of-fold)",
            "global_oof_cvar": robust(vy, oof_global, seg_idx), "bias_oof_cvar": robust(vy, oof_bias, seg_idx),
            "gate_oof_cvar": robust(vy, oof_gate, seg_idx)}
    best_head = max(("class_bias", "class_gate"), key=lambda h: head[{"class_bias": "bias_oof_cvar", "class_gate": "gate_oof_cvar"}[h]])
    best_value = head["bias_oof_cvar" if best_head == "class_bias" else "gate_oof_cvar"]
    head["adopted"] = best_head if best_value > head["global_oof_cvar"] + 0.002 else "global_weight"
    if head["adopted"] == "class_gate":
        gate = fit_gate(np.arange(len(vy)), w, rule)
        head["gate_graph_weight_per_class"] = dict(zip(classes, gate.tolist()))
        candidates["ct_graph_hybrid_global"] = candidates["ct_graph_hybrid"]
        P = gate * g_test + (1 - gate) * c_test
        candidates["ct_graph_hybrid"] = rules[rule](P / P.sum(1, keepdims=True))
    if head["adopted"] == "class_bias":
        bias = fit_bias(log_scores(w, rule, g_val, c_val), vy)
        head["bias"] = dict(zip(classes, bias.tolist()))
        candidates["ct_graph_hybrid_global"] = candidates["ct_graph_hybrid"]
        candidates["ct_graph_hybrid"] = (log_scores(w, rule, g_test, c_test) + bias).argmax(1)
    log("fusion head", json.dumps(head))
    report["hybrid_head_selection"] = head
    # FINAL (frozen before the confirmatory CIC test was read): the proposed hybrid is the equal-weight (1/N)
    # probability average of the two branch ensembles; only its decision rule is chosen on validation (segment CVaR).
    # Validation-tuned weights/heads did not transfer under campaign shift on the development dataset (forecast
    # combination puzzle) and are reported as ablations.
    candidates["ct_graph_hybrid_tuned"] = candidates["ct_graph_hybrid"]
    candidates["ct_graph_hybrid"] = candidates["ct_graph_hybrid_equal_weight"]
    report["hybrid_primary"] = {"fusion": "equal weight (0.5 / 0.5)", "rule": er, "rule_selection": "CVaR_0.25 over validation segments"}
    report["hybrid"] = {"weight_graph": w, "rule": rule, "selection": "CVaR_0.25 over validation segments",
                        "pooled_selection": {"weight_graph": pw, "rule": pr}, "validation_grid_macro_mcc": grid,
                        "validation_grid_segment_cvar": robust_grid,
                        "graph_seeds": len(graph_runs), "context_seeds": len(ct),
                        "test": {name: {m: cm_summary(np.bincount(y[keep] * K + candidates[name][keep], minlength=K * K).reshape(K, K))[m]
                                        for m in metrics} for name in candidates if name.startswith(("ct_graph_hybrid", "context_branch"))}}
baselines, detectors = {}, {}
BENIGN = [c.lower() for c in classes].index("benign")
for path in sorted(glob.glob(f"{INPUT}/**/baselines/*_test_predictions.npz", recursive=True)):
    name = os.path.basename(path).replace("_test_predictions.npz", "")
    data = np.load(path, allow_pickle=True)  # baseline files are written by this project's own kernels
    ids = data["source_id"].astype(object)
    o = np.argsort(ids)
    if not np.array_equal(ids[o], base["ids"]) or not np.array_equal(data["label"][o].astype(np.int64), y):
        report["comparisons"][name] = {"status": "row sets differ; skipped"}
        continue
    if "binary_only" in data.files:  # unsupervised anomaly detectors (EULER): benign vs attack only
        detectors[name] = {"attack": (data["prediction"][o] != BENIGN).astype(np.int64), "novel": data["novel"][o],
                           "score": data["attack_score"][o].astype(np.float64)}
        continue
    baselines[name] = {"pred": data["prediction"][o].astype(np.int64), "novel": data["novel"][o]}
    if "benign_probability" in data.files:
        SCORES[name] = 1 - data["benign_probability"][o].astype(np.float64)
novel = next(iter(baselines.values()))["novel"] if baselines else (
    next(iter(detectors.values()))["novel"] if detectors else np.zeros(len(y), bool))
primary = ("ct_graph_hybrid" if "ct_graph_hybrid" in candidates else
           "graph_ensemble" if "graph_ensemble" in candidates else "graph_seed0")
report["primary_graph"] = primary
for name, b in baselines.items():
    entry = {"graph": {m: cm_summary(np.bincount(y[keep] * K + candidates[primary][keep], minlength=K * K).reshape(K, K))[m] for m in metrics},
             "baseline": {m: cm_summary(np.bincount(y[keep] * K + b["pred"][keep], minlength=K * K).reshape(K, K))[m] for m in metrics}}
    for level, unit in units.items():
        entry[f"all_rows_{level}"] = paired_inference(y[keep], candidates[primary][keep], b["pred"][keep], unit[keep], K)
        subset = keep & novel
        entry[f"novel_rows_{level}"] = paired_inference(y[subset], candidates[primary][subset], b["pred"][subset], unit[subset], K)
    entry["novel_rows"] = int((keep & novel).sum())
    entry["variants_block"] = {v: paired_inference(y[keep], candidates[v][keep], b["pred"][keep], units["block"][keep], K)
                               for v in candidates if v != primary}
    # Benign-vs-attack view of the same predictions (collapsing attack types).
    entry["binary_block"] = paired_inference((y[keep] != BENIGN).astype(np.int64), (candidates[primary][keep] != BENIGN).astype(np.int64),
                                             (b["pred"][keep] != BENIGN).astype(np.int64), units["block"][keep], 2)
    report["comparisons"][name] = entry
    log(name, json.dumps({k: entry[k]["delta"] for k in ("all_rows_block",)}), json.dumps(entry["all_rows_block"]["p_randomisation_two_sided"]))
# OPTIONAL SINGLE-SEED ABLATION (cost of the three-seed ensembles; reported as an ablation, the proposed model is unchanged).
# Each graph seed alone is fused, with equal weights, with (a) the three-seed context branch and (b) the context seed of the
# same index. The decision rule is chosen exactly as for the proposed model (CVaR_0.25 over validation segments); nothing
# is chosen on test. Paired tests against the GBDTs and against the three-seed CT-Graph use the headline units.
if globals().get("SINGLE_SEED") and "ct_graph_hybrid" in candidates:
    single = {}
    c_val_seeds = [aligned(vids, c["val_ids"].astype(object), c["val_proba"].astype(np.float64)) for c in ct]
    for i_, run_ in enumerate(graph_runs):
        g_val_i, g_test_i = run_["validation"]["proba"][vmask], run_["proba"]
        for tag, cv, ctst in (("ctx3", c_val, c_test), ("ctx1", c_val_seeds[i_], c_seeds[i_])):
            fused_val = 0.5 * g_val_i + 0.5 * cv
            cvar = {r: robust(vy, f(fused_val), seg_idx) for r, f in rules.items()}
            rl = max(rules, key=lambda r: cvar[r])
            name = f"ct_graph_g{i_}_{tag}"
            fused_test = 0.5 * g_test_i + 0.5 * ctst
            candidates[name] = rules[rl](fused_test)
            SCORES[name] = 1 - fused_test[:, BEN_IDX]
            entry = {"graph_seed_index": i_, "graph_run": run_["path"], "context": "three seeds" if tag == "ctx3" else f"seed index {i_}",
                     "rule": rl, "validation_segment_cvar": cvar,
                     "test": {m: cm_summary(np.bincount(y[keep] * K + candidates[name][keep], minlength=K * K).reshape(K, K))[m]
                              for m in metrics}, "vs": {}}
            refs = [("CT-Graph (3+3 seeds)", candidates["ct_graph_hybrid"])] + [(b, baselines[b]["pred"]) for b in ("LightGBM", "XGBoost")
                                                                                if b in baselines]
            for ref, pred_ref in refs:
                entry["vs"][ref] = {lv: paired_inference(y[keep], candidates[name][keep], pred_ref[keep], u[keep], K)
                                    for lv, u in units.items()}
            single[name] = entry
            log("single-seed", name, rl, json.dumps({m: round(v, 4) for m, v in entry["test"].items()}),
                json.dumps({ref: round(e["block"]["delta"]["macro_mcc"], 4) for ref, e in entry["vs"].items()}))
    report["single_seed"] = single
# OPTIONAL STATISTICAL POWER / DESIGN ANALYSIS of every proposed-vs-baseline comparison (no test-tuned choice depends on
# it): bootstrap standard error of the macro-MCC difference, minimum detectable effect at 80 % power (two-sided 0.05),
# analytic power for plausible true effects, Gelman & Carlin (2014) design analysis (Type S = probability that a
# significant estimate has the wrong sign; Type M = expected exaggeration of a significant estimate), the simulated power
# of the ACTUAL sign-flip randomisation test at the observed effect (resampling units with replacement), the smallest
# p-value the randomisation test can reach with this many units, and Holm-adjusted p-values over the baselines.
if globals().get("POWER"):
    from math import erf, sqrt

    def Phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    def macro_batch(cms):  # (..., K, K) -> macro-MCC over present classes
        cms = cms.astype(np.float64)
        t, pp = cms.sum(-1), cms.sum(-2)
        tp = np.diagonal(cms, axis1=-2, axis2=-1)
        s = cms.sum((-1, -2))[..., None]
        fn, fp = t - tp, pp - tp
        tn = s - tp - fn - fp
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
        present = t > 0
        return (mcc * present).sum(-1) / np.maximum(present.sum(-1), 1)

    def power_analysis(yv, a, b, unit, k, worlds=300, flips=999, seed=11):
        names, index = np.unique(unit, return_inverse=True)
        U = len(names)
        ca, cb = unit_cms(yv, a, index, U, k), unit_cms(yv, b, index, U, k)
        obs = float(macro_batch(ca.sum(0)) - macro_batch(cb.sum(0)))
        rng_ = np.random.default_rng(seed)
        draws = rng_.integers(0, U, (2000, U))
        boot = macro_batch(ca[draws].sum(1)) - macro_batch(cb[draws].sum(1))
        se = float(np.std(boot, ddof=1))
        z, zb = 1.959964, 0.841621
        out = {"units": int(U), "observed_delta": obs, "bootstrap_se": se, "mde_80": (z + zb) * se,
               "min_attainable_p_two_sided": float(2.0 / 2 ** U) if U <= 60 else 0.0}
        sims = rng_.standard_normal(200_000)
        design = {}
        for d in (0.01, 0.02, 0.05):
            pw = Phi(d / se - z) + Phi(-d / se - z) if se > 0 else 1.0
            est = d + se * sims
            sig = np.abs(est) > z * se
            design[str(d)] = {"power": float(pw), "type_s": float((est[sig] < 0).mean()) if sig.any() else None,
                              "exaggeration": float(np.abs(est[sig]).mean() / d) if sig.any() else None}
        out["design_analysis"] = design
        # simulated power of the actual sign-flip test, true effect = observed (units resampled with replacement)
        rejections = 0
        for _ in range(worlds):
            dr = rng_.integers(0, U, U)
            A, B = ca[dr], cb[dr]
            point = macro_batch(A.sum(0)) - macro_batch(B.sum(0))
            F = rng_.random((flips, U)) < 0.5
            SA = np.where(F[:, :, None, None], B[None], A[None]).sum(1)
            SB = np.where(F[:, :, None, None], A[None], B[None]).sum(1)
            null = macro_batch(SA) - macro_batch(SB)
            pval = (1 + np.sum(np.abs(null) >= abs(point) - 1e-12)) / (flips + 1)
            rejections += pval < 0.05
        out["simulated_power_signflip_at_observed"] = rejections / worlds
        return out

    def holm(pvals):
        names_ = sorted(pvals, key=pvals.get)
        m, adj, run = len(names_), {}, 0.0
        for i, nme in enumerate(names_):
            run = max(run, min(1.0, (m - i) * pvals[nme]))
            adj[nme] = run
        return adj

    report["power"] = {"note": "design analysis; alpha 0.05 two-sided; effects in macro-MCC", "comparisons": {}}
    for name, b in baselines.items():
        report["power"]["comparisons"][name] = {level: power_analysis(y[keep], candidates[primary][keep], b["pred"][keep], unit[keep], K)
                                                for level, unit in units.items()}
        log("power", name, json.dumps({lv: {k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in r_.items() if k_ != "design_analysis"}
                                       for lv, r_ in report["power"]["comparisons"][name].items()}))
    for level in units:
        pv = {nm: report["comparisons"][nm][f"all_rows_{level}"]["p_randomisation_two_sided"]["macro_mcc"]
              for nm in baselines if "all_rows_" + level in report["comparisons"].get(nm, {})}
        report["power"][f"holm_{level}"] = {"raw": pv, "adjusted": holm(pv)} if pv else {}
# Binary detection comparison with the unsupervised detectors: the proposed model's decision is "attack" when its
# predicted class is not benign; the detector's own validation-chosen cutoff decides for it.
attack_true = (y != BENIGN).astype(np.int64)
graph_attack = (candidates[primary] != BENIGN).astype(np.int64)
report["binary_comparisons"] = {}
for name, d in detectors.items():
    entry = {"graph": {m: cm_summary(np.bincount(attack_true[keep] * 2 + graph_attack[keep], minlength=4).reshape(2, 2))[m]
                       for m in ("mcc", "accuracy", "macro_f1")},
             "baseline": {m: cm_summary(np.bincount(attack_true[keep] * 2 + d["attack"][keep], minlength=4).reshape(2, 2))[m]
                          for m in ("mcc", "accuracy", "macro_f1")}}
    for tag, pred in (("graph", graph_attack), ("baseline", d["attack"])):
        t, p = attack_true[keep], pred[keep]
        entry[tag]["detection_rate"] = float((p[t == 1] == 1).mean()) if (t == 1).any() else None
        entry[tag]["false_alarm_rate"] = float((p[t == 0] == 1).mean()) if (t == 0).any() else None
    for level, unit in units.items():
        entry[f"all_rows_{level}"] = paired_inference(attack_true[keep], graph_attack[keep], d["attack"][keep], unit[keep], 2)
        subset = keep & novel
        entry[f"novel_rows_{level}"] = paired_inference(attack_true[subset], graph_attack[subset], d["attack"][subset], unit[subset], 2)
    report["binary_comparisons"][name] = entry
    log(name, "binary", json.dumps(entry["all_rows_block"]["delta"]), json.dumps(entry["all_rows_block"]["p_randomisation_two_sided"]))
def per_class(pred):
    cm = np.bincount(y[keep] * K + pred[keep], minlength=K * K).reshape(K, K)
    t, p = cm.sum(1), cm.sum(0)
    return {classes[i]: {"recall": round(float(cm[i, i] / t[i]), 4), "precision": round(float(cm[i, i] / p[i]), 4) if p[i] else None,
                         "support": int(t[i])} for i in range(K) if t[i]}


report["per_class"] = {n: per_class(candidates[n]) for n in ("ct_graph_hybrid", "graph_ensemble", "context_branch") if n in candidates}
report["per_class"].update({n: per_class(b["pred"]) for n, b in baselines.items()})
# OPTIONAL full metric set for every model on the evaluated rows (same confusion-matrix definitions for all models).
if globals().get("ALL_METRICS"):
    from sklearn.metrics import average_precision_score, roc_auc_score

    def full_metrics(pred, score=None):
        yk, pk = y[keep], pred[keep]
        cm = np.bincount(yk * K + pk, minlength=K * K).reshape(K, K).astype(np.float64)
        t, pp, tp = cm.sum(1), cm.sum(0), np.diag(cm)
        s_, fn, fp = cm.sum(), t - tp, pp - tp
        tn = s_ - tp - fn - fp
        present = t > 0
        prec = np.divide(tp, pp, out=np.zeros(K), where=pp > 0)
        rec = np.divide(tp, t, out=np.zeros(K), where=t > 0)
        f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros(K), where=(2 * tp + fp + fn) > 0)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc_k = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
        c_ = tp.sum()
        den_all = np.sqrt((s_ * s_ - (pp * pp).sum()) * (s_ * s_ - (t * t).sum()))
        pe = (pp * t).sum() / (s_ * s_)
        a_true, a_pred = yk != BEN_IDX, pk != BEN_IDX
        btp, bfp = float((a_true & a_pred).sum()), float((~a_true & a_pred).sum())
        bfn, btn = float((a_true & ~a_pred).sum()), float((~a_true & ~a_pred).sum())
        bden = np.sqrt((btp + bfp) * (btp + bfn) * (btn + bfp) * (btn + bfn))
        out = {"macro_mcc": float(mcc_k[present].mean()), "mcc": float((c_ * s_ - (pp * t).sum()) / den_all) if den_all > 0 else 0.0,
               "accuracy": float(c_ / s_), "balanced_accuracy": float(rec[present].mean()),
               "macro_f1": float(f1[present].mean()), "weighted_f1": float((f1 * t).sum() / t.sum()),
               "macro_precision": float(prec[present].mean()), "macro_recall": float(rec[present].mean()),
               "cohen_kappa": float((c_ / s_ - pe) / (1 - pe)) if pe < 1 else 0.0,
               "binary": {"detection_rate": btp / max(btp + bfn, 1), "false_alarm_rate": bfp / max(bfp + btn, 1),
                          "precision": btp / max(btp + bfp, 1), "f1": 2 * btp / max(2 * btp + bfp + bfn, 1),
                          "mcc": float((btp * btn - bfp * bfn) / bden) if bden > 0 else 0.0},
               "per_class": {classes[i]: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "mcc": float(mcc_k[i]),
                                          "support": int(t[i])} for i in range(K) if present[i]},
               "rows": int(keep.sum()),
               # rows = true class, columns = predicted class, in the order of `classes`
               "confusion_matrix": cm.astype(np.int64).tolist(), "confusion_labels": list(classes),
               # binary: rows = true (Benign, Attack), columns = predicted (Benign, Attack)
               "binary_confusion_matrix": [[int(btn), int(bfp)], [int(bfn), int(btp)]]}
        if score is not None and a_true.any() and (~a_true).any():
            sc = np.nan_to_num(score[keep].astype(np.float64))
            out["binary"]["roc_auc"] = float(roc_auc_score(a_true, sc))
            out["binary"]["pr_auc"] = float(average_precision_score(a_true, sc))
        return out

    allm = {}
    for name in ("ct_graph_hybrid", "graph_ensemble", "context_branch",
                 *[n for n in candidates if n.startswith(("graph_seed", "context_seed", "ct_graph_g"))]):
        if name in candidates:
            allm[name] = full_metrics(candidates[name], SCORES.get(name))
    for name, b in baselines.items():
        allm[name] = full_metrics(b["pred"], SCORES.get(name))
    for name, d in detectors.items():  # binary-only detectors (EULER)
        a_true = (y[keep] != BEN_IDX)
        a_pred = d["attack"][keep].astype(bool)
        btp, bfp = float((a_true & a_pred).sum()), float((~a_true & a_pred).sum())
        bfn, btn = float((a_true & ~a_pred).sum()), float((~a_true & ~a_pred).sum())
        bden = np.sqrt((btp + bfp) * (btp + bfn) * (btn + bfp) * (btn + bfn))
        from sklearn.metrics import average_precision_score as _ap, roc_auc_score as _ra
        allm[name] = {"binary": {"detection_rate": btp / max(btp + bfn, 1), "false_alarm_rate": bfp / max(bfp + btn, 1),
                                 "precision": btp / max(btp + bfp, 1), "f1": 2 * btp / max(2 * btp + bfp + bfn, 1),
                                 "mcc": float((btp * btn - bfp * bfn) / bden) if bden > 0 else 0.0,
                                 "roc_auc": float(_ra(a_true, d["score"][keep])), "pr_auc": float(_ap(a_true, d["score"][keep]))},
                      "rows": int(keep.sum()), "binary_only": True,
                      "binary_confusion_matrix": [[int(btn), int(bfp)], [int(bfn), int(btp)]]}
    report["all_metrics"] = allm
    for name, m in allm.items():
        if "macro_mcc" in m:
            log("ALL", name, {k: round(m[k], 4) for k in ("macro_mcc", "mcc", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa")},
                {k: round(v, 4) for k, v in m["binary"].items()})
        else:
            log("ALL", name, {k: round(v, 4) for k, v in m["binary"].items()})
json.dump(report, open(os.path.join(OUT, "posthoc.json"), "w"), indent=1)
print(json.dumps({k: report[k] for k in ("primary_graph", "seed_runs", "cleaning") if k in report}, indent=1))
for n, pcs in report["per_class"].items():
    print("per-class recall", n, {c: v["recall"] for c, v in pcs.items()})
for name, entry in report["comparisons"].items():
    if "status" in entry:
        continue
    for key in ("all_rows_block", "novel_rows_block", "all_rows_segment"):
        e = entry[key]
        print(name, key, "units", e["units"], "delta", {m: round(v, 4) for m, v in e["delta"].items()},
              "ci", {m: [round(a, 4) for a in v] for m, v in e["ci95"].items()}, "p_rand", e["p_randomisation_two_sided"])
log("posthoc kernel done")
# OPTIONAL REVISION ANALYSES (Computers & Security major revision). Frozen predictions only; nothing is chosen on test.
#  R1 graph-free comparison: the fused model against its own context branch and graph branch (block and segment units)
#  R2 distribution of the paired effect over units and leave-one-unit-out (jackknife) influence on the pooled difference
#  R3 binary operating points with the threshold frozen on SOURCE VALIDATION benign records (FAR 1% / 5%), applied to test
#  R4 attack-score distributions of benign and attack test records, and ROC-AUC per branch
#  R5 confusion counts and Wilson intervals of recall / precision for every class and model
if globals().get("REVISION") and "ct_graph_hybrid" in candidates:
    rev = {}
    yk = y[keep]

    def per_unit_delta(a, b, unit, min_rows=200):
        names_, idx_ = np.unique(unit, return_inverse=True)
        ca, cb = unit_cms(yk, a, idx_, len(names_), K), unit_cms(yk, b, idx_, len(names_), K)
        pooled = cm_summary(ca.sum(0))["macro_mcc"] - cm_summary(cb.sum(0))["macro_mcc"]
        rows_ = ca.sum((1, 2))
        local = [cm_summary(ca[i])["macro_mcc"] - cm_summary(cb[i])["macro_mcc"] for i in range(len(names_))
                 if rows_[i] >= min_rows and (ca[i].sum(1) > 0).sum() >= 2]
        jack = [cm_summary(ca.sum(0) - ca[i])["macro_mcc"] - cm_summary(cb.sum(0) - cb[i])["macro_mcc"] for i in range(len(names_))]
        local = np.asarray(local)
        share = np.sort(rows_)[::-1]
        return {"units": int(len(names_)), "pooled_delta": pooled,
                "local_units": int(len(local)), "local_delta_median": float(np.median(local)) if len(local) else None,
                "local_delta_q25_q75": [float(np.percentile(local, 25)), float(np.percentile(local, 75))] if len(local) else None,
                "local_positive_share": float((local > 0).mean()) if len(local) else None,
                "local_negative_share": float((local < 0).mean()) if len(local) else None,
                "jackknife_min": float(np.min(jack)), "jackknife_max": float(np.max(jack)),
                "largest_unit_row_share": float(share[0] / rows_.sum()), "top5_unit_row_share": float(share[:5].sum() / rows_.sum())}

    a_h = candidates["ct_graph_hybrid"][keep]
    rev["graph_free"] = {}
    for other in ("context_branch", "graph_ensemble"):
        if other in candidates:
            b_ = candidates[other][keep]
            rev["graph_free"][other] = {lvl: paired_inference(yk, a_h, b_, units[lvl][keep], K) for lvl in units}
            rev["graph_free"][other]["binary_block"] = paired_inference((yk != BEN_IDX).astype(np.int64), (a_h != BEN_IDX).astype(np.int64),
                                                                        (b_ != BEN_IDX).astype(np.int64), units["block"][keep], 2)
    rev["effect_distribution"] = {}
    for other, pred_ in [("context_branch", candidates.get("context_branch")), *[(n, b["pred"]) for n, b in baselines.items()]]:
        if pred_ is None:
            continue
        rev["effect_distribution"][other] = {lvl: per_unit_delta(a_h, pred_[keep], units[lvl][keep]) for lvl in units}
    # R3: thresholds on the attack score 1 - p(benign), fixed on validation benign records of the source data.
    vb = vy != BEN_IDX
    val_scores = {"ct_graph_hybrid": 1 - (0.5 * g_val + 0.5 * c_val)[:, BEN_IDX], "graph_ensemble": 1 - g_val[:, BEN_IDX],
                  "context_branch": 1 - c_val[:, BEN_IDX]}
    test_scores = {"ct_graph_hybrid": 1 - (0.5 * g_test + 0.5 * c_test)[:, BEN_IDX], "graph_ensemble": 1 - g_test[:, BEN_IDX],
                   "context_branch": 1 - c_test[:, BEN_IDX]}
    a_true = (yk != BEN_IDX)
    from sklearn.metrics import roc_auc_score as _auc
    rev["operating_points"], rev["score_distribution"] = {}, {}
    for name_, sv in val_scores.items():
        st = test_scores[name_][keep]
        ben_v = np.sort(sv[~vb])
        ops = {}
        for far in (0.001, 0.01, 0.05):
            thr = float(np.quantile(ben_v, 1 - far)) if len(ben_v) else 1.0
            fa = st[~a_true] > thr
            dr = st[a_true] > thr
            ops[f"val_far_{far}"] = {"threshold": thr, "validation_far": float((sv[~vb] > thr).mean()), "validation_dr": float((sv[vb] > thr).mean()),
                                     "test_far": float(fa.mean()) if len(fa) else None, "test_detection_rate": float(dr.mean()) if len(dr) else None,
                                     "test_false_alarms": int(fa.sum()), "test_benign": int((~a_true).sum())}
        ops["argmax_rule_test_far"] = float((candidates[name_][keep][~a_true] != BEN_IDX).mean()) if name_ in candidates else None
        rev["operating_points"][name_] = ops
        rev["score_distribution"][name_] = {
            "test_roc_auc": float(_auc(a_true, st)) if 0 < a_true.mean() < 1 else None,
            "validation_roc_auc": float(_auc(vb, sv)) if 0 < vb.mean() < 1 else None,
            "benign_test_quantiles": [float(np.quantile(st[~a_true], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)] if (~a_true).any() else None,
            "attack_test_quantiles": [float(np.quantile(st[a_true], q)) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9)] if a_true.any() else None,
            "benign_validation_quantiles": [float(np.quantile(sv[~vb], q)) for q in (0.5, 0.9, 0.99, 0.999)]}
    # Where do the benign false alarms of the argmax rule go, and how do their scores compare with true attacks?
    fa_mask = (~a_true) & (a_h != BEN_IDX)
    rev["benign_false_alarms"] = {"count": int(fa_mask.sum()), "benign": int((~a_true).sum()),
                                  "predicted_as": {classes[k]: int((a_h[fa_mask] == k).sum()) for k in range(K) if (a_h[fa_mask] == k).any()},
                                  "hybrid_score_median_false_alarms": float(np.median(test_scores["ct_graph_hybrid"][keep][fa_mask])) if fa_mask.any() else None}

    # R5: confusion counts and Wilson 95% intervals.
    def wilson(k_, n_, z=1.959964):
        if n_ == 0:
            return [None, None]
        p_ = k_ / n_
        d_ = 1 + z * z / n_
        c_ = (p_ + z * z / (2 * n_)) / d_
        h_ = z * np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_)) / d_
        return [float(c_ - h_), float(c_ + h_)]
    rev["class_counts"] = {}
    for name_, pred_ in [("ct_graph_hybrid", a_h), ("graph_ensemble", candidates.get("graph_ensemble")), ("context_branch", candidates.get("context_branch")),
                         *[(n, b["pred"]) for n, b in baselines.items()]]:
        if pred_ is None:
            continue
        p_ = pred_ if len(pred_) == len(yk) else pred_[keep]
        out_ = {}
        for k in range(K):
            tp_ = int(((yk == k) & (p_ == k)).sum()); sup = int((yk == k).sum()); pp = int((p_ == k).sum())
            if sup == 0:
                continue
            out_[classes[k]] = {"support": sup, "tp": tp_, "fn": sup - tp_, "fp": pp - tp_, "recall": tp_ / sup,
                                "recall_ci95": wilson(tp_, sup), "precision": tp_ / pp if pp else None, "precision_ci95": wilson(tp_, pp)}
        rev["class_counts"][name_] = out_
    rev["fusion_rules"] = {"equal_weight_rule": er, "segment_cvar_selection": {"weight_graph": w, "rule": rule},
                           "pooled_selection": {"weight_graph": pw, "rule": pr}}
    # Calibration of the binary attack score on test: Brier score and ECE (15 equal-width bins).
    def _ece(p_, t_, bins=15):
        idx_ = np.clip((p_ * bins).astype(np.int64), 0, bins - 1)
        return float(sum(abs(p_[idx_ == b].mean() - t_[idx_ == b].mean()) * (idx_ == b).mean() for b in range(bins) if (idx_ == b).any()))
    cal_scores = {k_: np.clip(v_[keep], 0, 1) for k_, v_ in test_scores.items()}
    for name_ in ("LightGBM_rel", "XGBoost_rel", "LightGBM", "XGBoost"):
        if name_ in SCORES:
            cal_scores[name_] = np.clip(SCORES[name_][keep], 0, 1)
    tt_ = a_true.astype(np.float64)
    rev["calibration_binary"] = {n_: {"brier": float(np.mean((v_ - tt_) ** 2)), "ece_15": _ece(v_, tt_)} for n_, v_ in cal_scores.items()}
    # Target-site recalibration with a small benign sample (labels of benign records only; no attack label is used):
    # the threshold on the attack score is set at the stated false-alarm rate on 1% / 5% of the target benign records
    # (the earliest ones in stream order, or a random sample, 20 repetitions) and evaluated on all other target records.
    rng_tc = np.random.default_rng(20261118)
    ben_idx = np.flatnonzero(~a_true)
    rev["target_benign_recalibration"] = {}
    for n_, v_ in cal_scores.items():
        out_ = {}
        for frac in tuple(globals().get("RECAL_FRACS", (0.01, 0.05))):
            m_ = max(1, int(frac * len(ben_idx)))
            for far in tuple(globals().get("RECAL_FARS", (0.001, 0.01))):
                for how in ("earliest", "random"):
                    fars, drs = [], []
                    for rep_ in range(1 if how == "earliest" else 20):
                        pick = ben_idx[:m_] if how == "earliest" else rng_tc.choice(ben_idx, m_, replace=False)
                        thr = float(np.quantile(v_[pick], 1 - far))
                        rest = np.ones(len(v_), bool)
                        rest[pick] = False
                        fars.append(float((v_[rest & ~a_true] > thr).mean()))
                        drs.append(float((v_[rest & a_true] > thr).mean()))
                    out_[f"{how}_benign_{frac}_far_{far}"] = {"benign_sample": int(m_), "test_far": float(np.mean(fars)),
                                                             "test_far_range": [float(min(fars)), float(max(fars))],
                                                             "test_detection_rate": float(np.mean(drs)),
                                                             "test_detection_rate_range": [float(min(drs)), float(max(drs))]}
        rev["target_benign_recalibration"][n_] = out_
    # Unsupervised prior re-estimation on the unlabelled target records (EM of Saerens et al., 2002): source prior from
    # the labelled validation split; the posteriors are re-weighted until the target prior estimate converges.
    yv_ok = vy[vy >= 0]
    pi_s = np.bincount(yv_ok, minlength=K).astype(np.float64)
    pi_s = np.maximum(pi_s / pi_s.sum(), 1e-6)
    rev["em_prior_adjustment"] = {}
    for n_, P_ in (("ct_graph_hybrid", 0.5 * g_test + 0.5 * c_test), ("graph_ensemble", g_test), ("context_branch", c_test)):
        P_ = np.clip(P_[keep].astype(np.float64), 1e-12, 1)
        pt = pi_s.copy()
        for it_ in range(200):
            w_ = P_ * (pt / pi_s)
            w_ /= w_.sum(1, keepdims=True)
            new = w_.mean(0)
            if np.abs(new - pt).max() < 1e-6:
                pt = new
                break
            pt = new
        pred_em = w_.argmax(1)
        cm_em = np.bincount(yk * K + pred_em, minlength=K * K).reshape(K, K)
        rev["em_prior_adjustment"][n_] = {"iterations": it_ + 1, "macro_mcc": cm_summary(cm_em)["macro_mcc"],
                                          "test_far": float((pred_em[~a_true] != BEN_IDX).mean()),
                                          "test_detection_rate": float((pred_em[a_true] != BEN_IDX).mean()),
                                          "estimated_target_prior": pt.tolist(), "source_prior": pi_s.tolist()}
    json.dump(rev, open(os.path.join(OUT, "revision.json"), "w"), indent=1, default=float)
    log("REVISION graph_free", json.dumps({o: {l: v[l]["delta"]["macro_mcc"] for l in units} for o, v in rev["graph_free"].items()}))
    log("REVISION operating points", json.dumps(rev["operating_points"]["ct_graph_hybrid"]))

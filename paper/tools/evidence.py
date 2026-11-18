"""Read-only access to the result files behind every number in the manuscript.

Nothing is recomputed from predictions here. Values come from the post-hoc outputs (results/final/*/posthoc/posthoc.json),
the training histories (results/v3_runs/*), the latency and collector-cost runs (results/latency,
results/relsys) and the cleaned-label audits (results/final/cleanids). The only derived quantity is the
Benjamini-Hochberg adjustment over the 16 dataset x baseline block-level p-values, computed from the stored raw p-values.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FINAL = RESULTS / "final"

# key, short label, long label, headline post-hoc, power post-hoc, collector-cost run, latency key
SETTINGS = [
    ("ton", "ToN", "NF-ToN-IoT-v3", "ctg-ton-allm787-posthoc-full", "ctg-ton-power787-posthoc-full", "ton", "ton"),
    ("cic", "CIC", "NF-CSE-CIC-IDS2018-v3", "ctg-cic-allm-posthoc-full", "ctg-cic-power-posthoc-full", "cic", "cic"),
    ("bs1", "BS1$\\rightarrow$BS2", "5G-NIDD, BS1 $\\rightarrow$ BS2", "ctg-5gs-bs1-allm-posthoc-full", "ctg-5gs-bs1-power-posthoc-full", "5gs-bs1", "bs1"),
    ("bs2", "BS2$\\rightarrow$BS1", "5G-NIDD, BS2 $\\rightarrow$ BS1", "ctg-5gs-bs2-allm-posthoc-full", "ctg-5gs-bs2-power-posthoc-full", "5gs-bs2", "bs2"),
]
PLAIN = {"ton": "ToN", "cic": "CIC", "bs1": "BS1→BS2", "bs2": "BS2→BS1"}
BASELINES = ["LightGBM", "XGBoost", "DNN", "NEGSC"]
MODELS = [("ct_graph_hybrid", "CT-Graph (proposed)"), ("graph_ensemble", "Graph branch"), ("context_branch", "Context branch"),
          ("LightGBM", "LightGBM"), ("XGBoost", "XGBoost"), ("DNN", "DNN"), ("NEGSC", "NEGSC"), ("EULER", "EULER")]
CLEAN = {
    "ton": [("Standard", "ctg-ton-allm787-posthoc-full"), ("C1", "ctg-ton-allmc1787-posthoc-full"),
            ("C2", "ctg-ton-allmc2787-posthoc-full"), ("C1$\\cup$C2", "ctg-ton-allmc1c2787-posthoc-full")],
    "bs2": [("Standard", "ctg-5gs-bs2-allm-posthoc-full"), ("C1", "ctg-5gs-bs2-allmc1-posthoc-full"),
            ("C2", "ctg-5gs-bs2-allmc2-posthoc-full"), ("C1$\\cup$C2", "ctg-5gs-bs2-allmc1c2-posthoc-full")],
}
# First-stage (frozen) configuration: v4b graph branch + flow-only CT-GBDT v4, before the second-stage adoptions.
FIRST_STAGE = {"ton": "ctg-ton-final787-posthoc-full", "cic": "ctg-cic-final-posthoc-full",
               "bs1": "ctg-5gs-bs1-final-posthoc-full", "bs2": "ctg-5gs-bs2-final-posthoc-full"}
# Graph-branch design sequence on ToN validation (seed 42; best epoch, better of the two decision rules).
ARCHITECTURE = [("v3", "ton-v3s42"), ("v3, no replay", "ton-v3nrs42"), ("v3, no first-contact", "ton-v3ncs42"),
                ("v3, depth 0", "ton-v3enc0s42"), ("v4a", "ton-v4as42"), ("v4b (adopted)", "ton-v4bs42"),
                ("v5a", "ton-v5as42"), ("v5b", "ton-v5bs42"), ("v5c", "ton-v5cs42"), ("v5d", "ton-v5ds42")]


@lru_cache(maxsize=None)
def posthoc(slug):
    return json.loads((FINAL / slug / "posthoc" / "posthoc.json").read_text())


def headline(key):
    return posthoc(next(s for s in SETTINGS if s[0] == key)[3])


def power(key):
    return posthoc(next(s for s in SETTINGS if s[0] == key)[4])["power"]


def metric(key, model, name, binary=False):
    m = headline(key)["all_metrics"].get(model)
    if m is None:
        return None
    return (m.get("binary") or {}).get(name) if binary else m.get(name)


def comparison(key, baseline, subset="all_rows"):
    c = headline(key)["comparisons"][baseline]
    blk, seg = c[f"{subset}_block"], c[f"{subset}_segment"]
    return {"delta": blk["delta"]["macro_mcc"], "ci": blk["ci95"]["macro_mcc"],
            "p_block": blk["p_randomisation_two_sided"]["macro_mcc"], "p_segment": seg["p_randomisation_two_sided"]["macro_mcc"],
            "units_block": blk["units"], "units_segment": seg["units"]}


def binary_comparison(key, baseline="EULER"):
    e = headline(key)["binary_comparisons"][baseline]
    blk, seg = e["all_rows_block"], e["all_rows_segment"]
    return {"delta": blk["delta"]["mcc"], "ci": blk["ci95"]["mcc"], "p_block": blk["p_randomisation_two_sided"]["mcc"],
            "p_segment": seg["p_randomisation_two_sided"]["mcc"]}


def benjamini_hochberg(pvalues):
    keys = list(pvalues)
    p = np.array([pvalues[k] for k in keys], dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, p[idx] * m / rank)
        adjusted[idx] = running
    return dict(zip(keys, adjusted))


def multiplicity():
    """Holm (within dataset, stored) and BH (across all 16 comparisons, from the stored raw p-values)."""
    out, raw_b, raw_s = {}, {}, {}
    for key, *_ in SETTINGS:
        P = power(key)
        for b in BASELINES:
            out[(key, b)] = {"holm_block": P["holm_block"]["adjusted"][b], "holm_segment": P["holm_segment"]["adjusted"][b]}
            raw_b[(key, b)] = P["holm_block"]["raw"][b]
            raw_s[(key, b)] = P["holm_segment"]["raw"][b]
    bh_b, bh_s = benjamini_hochberg(raw_b), benjamini_hochberg(raw_s)
    for k in out:
        out[k].update(bh_block=bh_b[k], bh_segment=bh_s[k])
    return out


def power_rows():
    rows = []
    for key, *_ in SETTINGS:
        P = power(key)
        for b in BASELINES:
            lv = P["comparisons"][b]
            rows.append({"key": key, "baseline": b, "delta": lv["block"]["observed_delta"],
                         **{f"{u}_{f}": lv[u][src] for u in ("block", "segment")
                            for f, src in (("se", "bootstrap_se"), ("mde", "mde_80"), ("sim", "simulated_power_signflip_at_observed"),
                                           ("minp", "min_attainable_p_two_sided"))},
                         "block_power02": lv["block"]["design_analysis"]["0.02"]["power"],
                         "segment_power02": lv["segment"]["design_analysis"]["0.02"]["power"],
                         "block_typem01": lv["block"]["design_analysis"]["0.01"]["exaggeration"],
                         "block_types01": lv["block"]["design_analysis"]["0.01"]["type_s"]})
    return rows


def per_class(key):
    return headline(key)["all_metrics"]


def confusion(key, model="ct_graph_hybrid"):
    m = headline(key)["all_metrics"][model]
    return np.asarray(m["confusion_matrix"], dtype=float), list(m["confusion_labels"])


def architecture():
    rows = []
    for label, run in ARCHITECTURE:
        path = next((RESULTS / "v3_runs" / run).glob("nids-*/best.training.json"))
        rows.append((label, json.loads(path.read_text())["best_validation_macro_mcc"]))
    return rows


def fusion(key):
    t = headline(key)["hybrid"]["test"]
    am = headline(key)["all_metrics"]
    h = headline(key)["hybrid"]
    return {"equal": t["ct_graph_hybrid_equal_weight"]["macro_mcc"], "graph": am["graph_ensemble"]["macro_mcc"],
            "context": am["context_branch"]["macro_mcc"], "pooled": t["ct_graph_hybrid_pooled_selection"]["macro_mcc"],
            "pooled_w": h["pooled_selection"]["weight_graph"], "cvar": t["ct_graph_hybrid_tuned"]["macro_mcc"], "cvar_w": h["weight_graph"]}


def cleaned(key):
    rows = []
    for label, slug in CLEAN[key]:
        d = posthoc(slug)
        am = d["all_metrics"]
        kept = am["ct_graph_hybrid"]["rows"]
        comp = {b: (d["comparisons"][b]["all_rows_block"]["delta"]["macro_mcc"],
                    d["comparisons"][b]["all_rows_block"]["p_randomisation_two_sided"]["macro_mcc"],
                    d["comparisons"][b]["all_rows_segment"]["p_randomisation_two_sided"]["macro_mcc"]) for b in BASELINES}
        rows.append({"set": label, "rows": kept, "scores": {m: am[m]["macro_mcc"] for m, _ in MODELS if m in am and am[m].get("macro_mcc") is not None},
                     "comparisons": comp})
    return rows


def latency():
    return json.loads((RESULTS / "latency" / "latency_table.json").read_text())


def relsys(key):
    run = next(s for s in SETTINGS if s[0] == key)[5]
    return json.loads((RESULTS / "relsys" / run / "relsys.json").read_text())


def first_stage(key):
    return posthoc(FIRST_STAGE[key])["hybrid"]["test"]["ct_graph_hybrid"]["macro_mcc"]


def split_sizes(key):
    """Rows per split from the post-hoc headline (test) and the preparation audits (train/val/calibration)."""
    return headline(key)["all_metrics"]["ct_graph_hybrid"]["rows"]


DEEP = {"ton": "ton", "cic": "cic", "bs1": "5gs-bs1", "bs2": "5gs-bs2"}
ANALYSIS = {"ton": ("ton_analysis", None), "cic": ("cic_analysis", None),
            "bs1": ("5g_analysis", "preserved"), "bs2": ("5g_analysis", "preserved")}


def audit(key):
    """Dataset audit: duplicate / label-conflicting vectors (whole release), training-side Bayes bound, train-to-validation
    shift (results/deep, train/validation/calibration only) and test rows whose vector occurs in training."""
    folder, sub = ANALYSIS[key]
    a = json.loads((RESULTS / folder / "analysis.json").read_text())
    dup = (a[sub] if sub else a)["duplicates"]
    d = json.loads((RESULTS / "deep" / DEEP[key] / "deep" / "deep.json").read_text())
    A, C = d["A_bayes_bound"], d["C_shift"]
    comp = headline(key)["comparisons"]["LightGBM"]
    rows = headline(key)["all_metrics"]["ct_graph_hybrid"]["rows"]
    return {"dup": dup["rows_with_duplicate_feature_vector"], "conflict": dup["rows_with_label_conflicting_feature_vector"],
            "ceiling": dup.get("max_accuracy_given_exact_feature_conflicts", dup.get("max_accuracy_given_exact_conflicts")),
            "train_conflict": A["rows_on_label_conflicting_vectors"], "bayes": A["per_flow_bayes_error_lower_bound_train"],
            "val_seen": A["validation_rows_with_train_vector"], "kl": C["label_prior_kl_bits_val_vs_train"],
            "adv_auc": C["adversarial_auc_all_flow_features"], "psi": C["features_psi_over_0.25"],
            "test_seen": 1 - comp["novel_rows"] / rows, "per_class_conflict_train": A["per_class_fraction_on_conflicting_vectors"]}


def flat_failures(min_gain=0.3, max_flat=0.5, min_ctgraph=0.7):
    """Classes that every multi-class baseline misses (best F1 below max_flat) but CT-Graph detects (F1 at least
    min_ctgraph and at least min_gain above the best baseline)."""
    out = []
    for key, *_ in SETTINGS:
        am = headline(key)["all_metrics"]
        for c, h in am["ct_graph_hybrid"]["per_class"].items():
            base = {b: am[b]["per_class"][c]["f1"] for b in BASELINES if c in am[b]["per_class"]}
            best = max(base.values())
            if best < max_flat and h["f1"] >= min_ctgraph and h["f1"] - best >= min_gain:
                out.append({"key": key, "class": c, "support": h["support"], "baselines": base,
                            "graph": am["graph_ensemble"]["per_class"][c]["f1"], "context": am["context_branch"]["per_class"][c]["f1"],
                            "ctgraph": h["f1"]})
    return out


SINGLE = {"ton": "ctg-ton-single787-posthoc-full", "cic": "ctg-cic-single-posthoc-full",
          "bs1": "ctg-5gs-bs1-single-posthoc-full", "bs2": "ctg-5gs-bs2-single-posthoc-full"}
LATSEED = {"ton": "ton", "cic": "cic", "bs1": "5gs-bs1", "bs2": "5gs-bs2"}


def stronger_gbdt(key):
    return max(("LightGBM", "XGBoost"), key=lambda b: metric(key, b, "macro_mcc"))


def single_seed(key):
    """Single-seed operating points of CT-Graph. Accuracy: post-hoc SINGLE_SEED run (frozen predictions, decision rule
    chosen on validation). Cost: measured per-seed graph test pass (2 x T4) + the context-branch cost of the latency table
    split in proportion to per-seed prediction times measured in a separate CPU run + smoothing/fusion/relational extras."""
    d = posthoc(SINGLE[key])["single_seed"]
    lat_key = next(s[6] for s in SETTINGS if s[0] == key)
    T = latency()[lat_key]
    seeds = json.loads((RESULTS / "latency" / "graph_seeds.json").read_text())[lat_key]
    L = json.loads((RESULTS / "latseed" / LATSEED[key] / "latency" / "latency.json").read_text())["timing"]
    per_full = [L[f"CT-GBDT seed {i}"]["full_us_per_flow"] for i in range(3)]
    per_micro = [L[f"CT-GBDT seed {i}"]["micro_ms"]["p50"] for i in range(3)]
    c3_full, c3_micro = T["CT-GBDT"]["full_us_per_flow"], T["CT-GBDT"]["micro_ms"]["p50"]
    rows, gb = T["rows"], stronger_gbdt(key)
    out = []
    for i, seed in enumerate(("42", "43", "44")):
        g_us = 1e6 * seeds[seed]["test_pass_elapsed_s"] / rows
        g_mb = 1e3 * seeds[seed]["microbatch_latency_s"]["50"]
        for tag, ctx_us, ctx_mb in (("ctx3", c3_full, c3_micro),
                                    ("ctx1", c3_full * per_full[i] / sum(per_full), c3_micro * per_micro[i] / sum(per_micro))):
            e = d[f"ct_graph_g{i}_{tag}"]
            out.append({"seed": int(seed), "context": tag, "rule": e["rule"], "macro_mcc": e["test"]["macro_mcc"],
                        "d_full": e["vs"]["CT-Graph (3+3 seeds)"]["block"]["delta"]["macro_mcc"],
                        "p_full": e["vs"]["CT-Graph (3+3 seeds)"]["block"]["p_randomisation_two_sided"]["macro_mcc"],
                        "d_gbdt": e["vs"][gb]["block"]["delta"]["macro_mcc"],
                        "p_gbdt": e["vs"][gb]["block"]["p_randomisation_two_sided"]["macro_mcc"],
                        "p_gbdt_seg": e["vs"][gb]["segment"]["p_randomisation_two_sided"]["macro_mcc"],
                        "us_flow": g_us + ctx_us + T["extra_us"], "micro_ms": g_mb + ctx_mb + T["extra_us"] * T["B"] / 1e3})
    return out



def ton_errors():
    """Test-error diagnosis of the ToN headline model (results/tonerr; uses test labels, informs no choice)."""
    d = json.loads((RESULTS / "tonerr" / "tonerr" / "tonerr.json").read_text())
    return d


def bs2_errors():
    """Test-error diagnosis of the frozen BS2->BS1 headline model (results/bs2err_v4b; test labels, no choice)."""
    return json.loads((RESULTS / "bs2err_v4b" / "bs2err" / "bs2err.json").read_text())

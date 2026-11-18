# CT-Graph: reproducibility package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24347748.svg)](https://doi.org/10.5281/zenodo.24347748)

**Paper:** *CT-Graph: Collector-Causal Temporal Graph Context for Detecting Multi-Flow Attacks
Missed by Per-Flow Intrusion Detectors* (revised manuscript)

**Authors:** Haitham Elwahsh, Tarek Khalifa, Omar Ibrahim Alirr, Engy El-shafeiy

This repository holds the code, frozen split definitions, run configurations and result files
of the paper.

The package contains what the reviewers asked to be released. It does not contain the raw
datasets (public, linked below) or per-record predictions.

| Request | Where |
|---|---|
| Frozen split definitions (Reviewer 2, comment 8; Reviewer 3) | `splits/` |
| Configuration files (Reviewer 2, comment 8) | `configs/` |
| Evaluation scripts in a reproducible form (Reviewer 2, comment 8) | `experiments/`, `paper/` |
| Code, instead of "available on request only" (Reviewer 3) | `run_pipeline.py`, `experiments/` |

## Layout

```
run_pipeline.py            data preparation, collector-causal graph, graph branch
requirements.txt
experiments/
  build_runs.py            defines every reported run and its settings
  temporal_lib.py          shared reader for the prepared splits
  *_body.py                44 analysis scripts (context branch, fusion, tests, baselines, audits)
splits/<dataset>/          frozen split manifests, split plans, feature statistics
configs/
  run_variants.json        settings and inputs of all 802 runs defined by the builder
  checkpoint_configs/      training configuration stored in each reported graph checkpoint
checkpoints/<dataset>/      the 12 reported graph-branch checkpoints (3 seeds per setting)
results/                   all 826 result files of the study (JSON, JSONL, CSV)
paper/
  tools/make_tables.py     regenerates every number and table of the paper
  tools/evidence.py        reads results/ for make_tables.py
```

## Contents

| Path | What it is |
|---|---|
| `run_pipeline.py` | Data preparation (Apache Spark), construction of the collector-causal graph, and training and evaluation of the graph branch |
| `experiments/build_runs.py` | Defines every reported run and the command-line settings it passes to `run_pipeline.py`; these settings override the script's defaults |
| `experiments/*_body.py` | Relational features (`relprep_body.py`); context branch (`context_gbdt_body.py`, `ctgbdt*_body.py`); fusion, decision rule, operating points and statistical tests (`posthoc_body.py`); per-flow baselines (`baselines_body.py`); matched and strict graph-free baselines (`matched_body.py`, `strict_body.py`); queueing, campaign and split audits (`queue_body.py`, `campaign_body.py`, `splitaudit_body.py`) |
| `splits/<dataset>/manifest.json` | Frozen split: segment boundaries, per-shard SHA-256 digests, timing profile and its digest, class counts per split |
| `splits/<dataset>/split_audit.json` | Split plan: block length, selection rule, split fractions and segments (for `nids-cic`, prepared with an earlier version that stored the plan only in its manifest, the same fields are copied from the manifest; its segments equal those of `nids-cic-rel`) |
| `splits/<dataset>/preprocess.json` | Feature statistics and vocabularies fitted on the training split only; its SHA-256 equals the `preprocess_sha256` recorded in the manifest |
| `configs/run_variants.json` | Settings and inputs of every run |
| `configs/checkpoint_configs/` | Configuration and parameter count read from 11 of the 12 released checkpoints (all but ToN seed 42): `5gs/` for both 5G-NIDD directions, `tc/` for ToN and CIC |
| `checkpoints/<dataset>/seed<k>/best.pt` | The frozen graph-branch checkpoints of the reported model, three seeds per setting; `checkpoints/SHA256SUMS` lists their digests |
| `results/` | Every result file of the study, with paths as in the original run tree: the files behind each number and table, the per-seed runs, and the development and diagnostic runs. Copies of the split manifests that each run carried are omitted; they are identical to `splits/` |
| `paper/tools/make_tables.py` | Writes `paper/tables/numbers.tex` (every number quoted in the paper) and the table files |

Prepared datasets: `nids-ton` (NF-ToN-IoT-v3); `nids-cic` and `nids-cic-rel`
(NF-CSE-CIC-IDS2018-v3, without and with relational inputs); `nids-5gs-bs1` and `nids-5gs-bs2`
(5G-NIDD, trained on base station 1 or 2, without TTL and hop-count fields). Every run on a
prepared dataset used a byte-identical manifest. Candidate preparations that the adoption
rule rejected (relational graph inputs on ToN and BS2) are defined in `configs/run_variants.json`
but are not part of any reported result and are not included.

## Reproducing the numbers and tables

```bash
pip install -r requirements.txt
python paper/tools/make_tables.py   # creates paper/tables/
```

From the files in `results/`, this reproduces the revised manuscript exactly:
1,543 numbers and 24 tables.

## Re-evaluating the released checkpoints

1. Prepare the dataset from the public release with its preparation run (`experiments/build_runs.py`,
   e.g. the `ton-prepare-full` variant) and check that the resulting `manifest.json` equals
   `splits/<dataset>/manifest.json`; preparation is deterministic.
2. Copy `checkpoints/<dataset>/seed<k>/best.pt` into the prepared folder.
3. Run `python run_pipeline.py --mode evaluate --output_dir <prepared folder>` (set `CTG_CPU=1`
   to evaluate on CPU). The test results are written under `<prepared folder>/evaluations/`.

Each checkpoint stores the configuration it was trained with (see `configs/checkpoint_configs/`).

## Reproducing the experiments

Every experiment is a self-contained batch run. `python experiments/build_runs.py <variant>`
writes the script for one run, with its code and settings, and a `run.json` listing the runs
whose outputs it reads. Inputs are mounted under `/data/input` and outputs are written to
`/data/output`. `configs/run_variants.json` lists all variants and their settings.

The raw datasets are public and are not redistributed here:

- NF-ToN-IoT-v3: <https://doi.org/10.48610/44d7c5e>
- NF-CSE-CIC-IDS2018-v3: <https://doi.org/10.48610/ece9b83>
- 5G-NIDD: <https://doi.org/10.21227/xtep-hv36>

Preparation is deterministic: the manifests in `splits/` give the per-shard SHA-256 digests
that a re-preparation can be checked against.

## Citation

If you use this code or these results, please cite the paper:

```bibtex
@unpublished{elwahsh2026ctgraph,
  author = {Elwahsh, Haitham and Khalifa, Tarek and Alirr, Omar Ibrahim and El-shafeiy, Engy},
  title  = {{CT-Graph}: Collector-Causal Temporal Graph Context for Detecting Multi-Flow Attacks
            Missed by Per-Flow Intrusion Detectors},
  year   = {2026},
  note   = {Manuscript under review}
}
```

This package is archived at Zenodo: <https://doi.org/10.5281/zenodo.24347748>.
The same information is in `CITATION.cff`.

## Licence

MIT, see `LICENSE`.

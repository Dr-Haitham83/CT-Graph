#!/usr/bin/env python3
"""Build batch runs that embed the current run_pipeline.py.

  python experiments/build_runs.py prepare-debug   # CPU: spread 2M-row subset, full prepare path
  python experiments/build_runs.py prepare-full    # CPU: complete NF-ToN-IoT-v3 preparation
  python experiments/build_runs.py train-debug     # GPU 2xT4: short train/evaluate on prepare-debug output
  python experiments/build_runs.py train-full      # GPU 2xT4: train/evaluate on prepare-full output

Each writes experiments/<name>/{run.json,<name>.py}: a self-contained script and the list of runs whose
outputs it reads (mounted under /data/input); results are written to /data/output.
"""
import base64
import hashlib
import json
import os
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent / "run_pipeline.py"
# CTG_OWNER selects the account; other accounts prepare their own copy of the frozen splits
# (deterministic preparation; manifests are compared before any cross-account comparison).
OWNER = os.environ.get("CTG_OWNER", "runner")

COMMON = r'''
import base64, glob, json, os, shutil, subprocess, sys, time, zlib
T0 = time.time()
CODE_DIR = "/data/output/code"
os.makedirs(CODE_DIR, exist_ok=True)
PIPELINE = os.path.join(CODE_DIR, "run_pipeline.py")
with open(PIPELINE, "wb") as f:
    f.write(zlib.decompress(base64.b64decode(PAYLOAD)))


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def sh(cmd, check=True, env=None):
    log("$", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    if check and result.returncode:
        raise SystemExit(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def pip(*packages):
    sh([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", *packages])


def show_failures(root):
    for path in sorted(glob.glob(os.path.join(root, "failure-*.json"))):
        with open(path) as f:
            failure = json.load(f)
        print("FAILURE", path, failure.get("error_type"), failure.get("error"))
        print(failure.get("traceback", "")[-6000:])
    for path in sorted(glob.glob(os.path.join(root, "*.log"))):
        print("==== tail", path)
        with open(path, errors="replace") as f:
            lines = [l for l in f.readlines() if "UserWarning" not in l and "return func" not in l]
        print("".join(lines[-80:]))
'''

PREPARE = r'''
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

pip("ruptures", "torch-geometric", "torchmetrics", "pydantic", "filelock", "psutil")
try:
    import pyspark  # noqa: F401
except ImportError:
    pip("pyspark>=3.5,<5")
sh(["java", "-version"])
scratch = "/data/tmp" if os.path.isdir("/data/tmp") else "/tmp"
os.makedirs(scratch, exist_ok=True)


def fetch_fairdata(dataset_id, member, sha256):
    """Official release from Fairdata (Etsin), verified against its published SHA-256."""
    import hashlib
    import requests
    target = os.path.join(scratch, os.path.basename(member))
    attempts = []
    for endpoint, payload in (("https://etsin.fairdata.fi/api/v3/download/authorize", {"cr_id": dataset_id, "file": member}),
                              ("https://download.fairdata.fi/authorize", {"dataset": dataset_id, "file": member}),
                              ("https://download.fairdata.fi/api/v1/authorize", {"dataset": dataset_id, "file": member})):
        try:
            r = requests.post(endpoint, json=payload, timeout=60)
            attempts.append((endpoint, r.status_code, r.text[:200]))
            if r.status_code != 200:
                continue
            body = r.json()
            url = body.get("url") or (f"{endpoint.rsplit('/', 1)[0].replace('/api/v1', '')}/download?token={body['token']}"
                                      f"&dataset={dataset_id}&file={member}" if body.get("token") else None)
            if not url:
                continue
            h = hashlib.sha256()
            with requests.get(url, stream=True, timeout=600) as download, open(target, "wb") as out:
                download.raise_for_status()
                for chunk in download.iter_content(1 << 20):
                    out.write(chunk)
                    h.update(chunk)
            if h.hexdigest() == sha256:
                log("fetched", member, "sha256 verified")
                return target
            attempts.append((endpoint, "sha256 mismatch", h.hexdigest()))
        except Exception as exc:
            attempts.append((endpoint, repr(exc)[:200]))
    raise SystemExit(f"Could not fetch {member} from Fairdata: {attempts}")


if FAIRDATA:
    parquet = fetch_fairdata(*FAIRDATA)
    # Place the release as <root>/5G-NIDD/<file> so run_pipeline reads <root>/dataset_analysis.json.
    layout_root = os.path.join(scratch, "fiveg")
    os.makedirs(os.path.join(layout_root, DATASET_NAME), exist_ok=True)
    moved = os.path.join(layout_root, DATASET_NAME, os.path.basename(parquet))
    shutil.move(parquet, moved)
    parquet = moved
    if SITE_LAYOUT:
        # Capture boundaries from Argus Rank resets; the site boundary from the publisher's separate
        # BTS_1 file (row count), checked to coincide with a capture boundary. Labels are not read.
        import zipfile
        import numpy as np
        import pandas as pd
        with zipfile.ZipFile(parquet) as z:
            member = [m for m in z.namelist() if m.lower().endswith(".csv")][0]
            rank = pd.read_csv(z.open(member), usecols=["Rank"])["Rank"].to_numpy()
        starts = np.flatnonzero(rank == 1)
        rows_per_capture = np.diff(np.r_[starts, len(rank)]).tolist()
        site_file = fetch_fairdata(FAIRDATA[0], SITE_LAYOUT["file"], SITE_LAYOUT["sha256"])
        with zipfile.ZipFile(site_file) as z:
            member = [m for m in z.namelist() if m.lower().endswith(".csv")][0]
            first_site_rows = sum(1 for _ in z.open(member)) - 1
        boundary = np.cumsum(rows_per_capture).tolist()
        if first_site_rows not in boundary:
            raise SystemExit(f"Site file rows {first_site_rows} do not match a capture boundary {boundary}")
        layout = [{"rows": int(n), "site": "BS1" if end <= first_site_rows else "BS2"} for n, end in zip(rows_per_capture, boundary)]
        with open(os.path.join(layout_root, "dataset_analysis.json"), "w") as f:
            json.dump({"datasets": {DATASET_NAME: {"capture_files": layout, "status_interval_s": 5.0,
                                                    "provenance": {"captures": "Rank resets", "site_boundary_rows": first_site_rows,
                                                                   "site_source": SITE_LAYOUT["file"]}}}}, f, indent=1)
        log("capture layout", len(layout), "captures; BS1 rows", first_site_rows, [(c["rows"], c["site"]) for c in layout])
else:
    source = sorted(glob.glob(f"/data/input/**/{CSV_NAME}", recursive=True))[0]
    parquet = os.path.join(scratch, DATASET_NAME, DATASET_NAME + ".parquet")
    os.makedirs(os.path.dirname(parquet), exist_ok=True)
# All-string columns reproduce the pipeline's CSV decoding exactly while giving Spark parallel
# row-group tasks (a single CSV is one ingestion task). Row order is preserved.
with (open(source, "rb") if not FAIRDATA else open(os.devnull, "rb")) as f:
    header = f.readline().decode("utf-8-sig").strip().split(",")
reader = [] if FAIRDATA else pacsv.open_csv(source, read_options=pacsv.ReadOptions(block_size=64 << 20),
                        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in header},
                                                             strings_can_be_null=True))
writer, rows, kept = None, 0, 0
windows = [(start, start + WINDOW_ROWS) for start in WINDOW_STARTS] if WINDOW_ROWS else None
for batch in reader:
    table = pa.Table.from_batches([batch])
    if windows is not None:
        pieces = []
        for a, b in windows:
            lo, hi = max(a, rows), min(b, rows + len(table))
            if lo < hi:
                pieces.append(table.slice(lo - rows, hi - lo))
        table = pa.concat_tables(pieces) if pieces else None
    rows += len(batch)
    if table is not None and len(table):
        if writer is None:
            writer = pq.ParquetWriter(parquet, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=262144)
        kept += len(table)
if writer is not None:
    writer.close()
log(f"input: {kept:,} of {rows:,} converted source rows -> {parquet} ({os.path.getsize(parquet)/2**20:.0f} MiB)")
import numpy as np
if CONTRACT:
    contract_path = os.path.join(scratch, "exporter_contract.json")
    with open(contract_path, "w") as f:
        json.dump(CONTRACT, f, indent=1)
    EXTRA = [*EXTRA, "--timeouts_json", contract_path]
for OUTPUT_NAME, RUN_EXTRA in RUNS:
    out = f"/data/output/{OUTPUT_NAME}"
    env = dict(os.environ, PYSPARK_PYTHON=sys.executable, PYSPARK_DRIVER_PYTHON=sys.executable)
    code = sh([sys.executable, PIPELINE, "--mode", "prepare", "--dataset", DATASET_NAME, "--data_path", parquet,
               "--output_dir", out, "--spark_master", "local[4]", "--spark_partitions", "64",
               "--spark_driver_memory", "10g", *EXTRA, *RUN_EXTRA], check=False, env=env)
    show_failures(out)
    if code:
        raise SystemExit(code)
    manifest = json.load(open(os.path.join(out, "manifest.json")))
    audit = json.load(open(os.path.join(out, "split_audit.json")))
    plan = audit["plan"]
    print(json.dumps({k: plan.get(k) for k in ("scope", "block_s", "selection_rule", "embargo_s")}, indent=1))
    for c in plan.get("candidates", []):
        print("candidate", c["block_s"], "segments", c["segments"], "coverage", round(c["min_share_over_target"], 3),
              "purge", round(c["estimated_purged_fraction"], 4))
    print(json.dumps(audit["class_counts"], indent=1))
    print("sizes", manifest["sizes"], "unknown test classes", manifest["unknown_test_classes"],
          "missing from holdouts", manifest["classes_missing_from_holdouts"])
    # Integrity: every shard is one stream, availability-sorted, row counts match the manifest.
    for split, shards in manifest["shards"].items():
        streams = {}
        for shard in shards:
            t = pq.read_table(shard["path"], columns=["stream", "hi", "label"])
            assert t.num_rows == shard["rows"], shard
            assert t["stream"].unique().to_pylist() == [shard["stream"]], shard["path"]
            hi = t["hi"].to_numpy()
            assert np.all(np.diff(hi) >= 0), f"unsorted shard {shard['path']}"
            streams.setdefault(shard["stream"], 0)
            streams[shard["stream"]] += shard["rows"]
        print(split, "streams", len(streams), "rows", sum(streams.values()), "largest stream", max(streams.values()))
    print(json.dumps({k: plan.get(k) for k in ("scope", "cut_method", "embargo_s", "note")}, indent=1))
    shutil.rmtree(os.path.join(out, "timing_warmup_reserved"), ignore_errors=True)
log("prepare kernel done")
'''

REEVAL = r'''
# EVALUATION-ONLY RE-RUN of a frozen checkpoint for the major-revision sensitivity analyses. Nothing is trained or selected:
#   CTG_ADMISSION_MARGIN_S = g'  : graph admission hi_j <= hi_i - g' and memory watermark with g' instead of the manifest g
#   CTG_MERGE_STREAMS = 1        : one monitor timeline per split, no state reset at segment boundaries (single GPU)
pip("ruptures", "torch-geometric", "torchmetrics", "pydantic", "filelock", "psutil")
import torch
ckpt = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/best.pt", recursive=True))
prepared = [os.path.dirname(m) for m in sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
            if os.path.isdir(os.path.join(os.path.dirname(m), "prepared"))]
if not ckpt or not prepared:
    raise SystemExit(f"checkpoint or prepared data for {PREPARED_NAME} not found")
source = os.path.dirname(ckpt[0])
out = f"/data/output/{PREPARED_NAME}"
os.makedirs(out, exist_ok=True)
for name in os.listdir(source):
    path = os.path.join(source, name)
    if os.path.isfile(path) and name not in ("latest_evaluation.json",):
        shutil.copy2(path, out)
os.symlink(os.path.join(prepared[0], "prepared"), os.path.join(out, "prepared"))
env = dict(os.environ)
if MARGIN is not None:
    env["CTG_ADMISSION_MARGIN_S"] = str(MARGIN)
if MERGE:
    env["CTG_MERGE_STREAMS"] = "1"
if globals().get("EVASION"):
    env["CTG_EVASION"] = EVASION  # test-time evasion of the attack records (see run_pipeline.evasion_spec)
if globals().get("DROP_SCALES"):
    env["CTG_DROP_MEMORY_SCALES"] = DROP_SCALES  # inference-time memory ablation
if globals().get("CPU"):
    env["CTG_CPU"] = "1"  # evaluation on CPU; the checkpoint's calibration is kept (see run_pipeline.cpu_mode)
    env["OMP_NUM_THREADS"] = str(os.cpu_count() or 4)
log("re-evaluation", {"margin": MARGIN, "merge": MERGE, "gpus": GPUS, "cpu": bool(globals().get("CPU")), "checkpoint": ckpt[0]})
try:
    code = sh([sys.executable, PIPELINE, "--mode", "evaluate", "--output_dir", out, "--gpus", str(GPUS), "--precision", "fp16",
               "--tf32", "off"], check=False, env=env)
    show_failures(out)
finally:
    os.unlink(os.path.join(out, "prepared"))
if code:
    raise SystemExit(code)
latest = json.load(open(os.path.join(out, "latest_evaluation.json")))["path"]
result = json.load(open(os.path.join(latest, "results.json")))
print(json.dumps({k: result["test"].get(k) for k in ("macro_mcc", "macro_f1", "accuracy", "detection_rate", "false_alarm_rate")}, indent=1))
'''

TRAIN = r'''
pip("ruptures", "torch-geometric", "torchmetrics", "pydantic", "filelock", "psutil")
import torch
log("cuda", torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
if not matches:
    raise SystemExit(f"prepared directory {PREPARED_NAME} not found under /data/input")
source = os.path.dirname(matches[0])
out = f"/data/output/{PREPARED_NAME}"
os.makedirs(out, exist_ok=True)
for name in ("manifest.json", "preprocess.json", "split_audit.json", "donors.pt", "exporter_timeouts.frozen.json"):
    if os.path.exists(os.path.join(source, name)):
        shutil.copy2(os.path.join(source, name), out)
if not os.path.exists(os.path.join(out, "prepared")):
    os.symlink(os.path.join(source, "prepared"), os.path.join(out, "prepared"))
common = ["--output_dir", out, "--gpus", "2", "--precision", "fp16", "--tf32", "off"]
try:
    code = sh([sys.executable, PIPELINE, "--mode", "train", *common, *TRAIN_ARGS], check=False)
    show_failures(out)
    if not code:
        code = sh([sys.executable, PIPELINE, "--mode", "evaluate", *common], check=False)
        show_failures(out)
finally:
    os.unlink(os.path.join(out, "prepared"))  # never re-export the mounted input
if code:
    raise SystemExit(code)
latest = json.load(open(os.path.join(out, "latest_evaluation.json")))["path"]
result = json.load(open(os.path.join(latest, "results.json")))
test = result["test"]
print(json.dumps({k: test.get(k) for k in ("macro_mcc", "mcc", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy",
                                           "detection_rate", "false_alarm_rate", "binary_pr_auc_binned",
                                           "binary_roc_auc_binned", "ece", "decision_rule", "decision_rules",
                                           "operating_point", "known_class_records", "unknown_class_records")}, indent=1))
for name, row in zip(result["classes"], test["per_class"]):
    print(f"{name:>12} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} n={row['support']}")
# Segment-cluster bootstrap: test rows within one segment share context and are not independent.
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def summary(cm):
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
    return {"macro_mcc": float(mcc_k[present].mean()), "macro_f1": float(f1_k[present].mean()),
            "accuracy": float(c / s), "mcc": float((c * s - (p * t).sum()) / den_all) if den_all > 0 else 0.0,
            "per_class_f1": f1_k.tolist(), "per_class_mcc": mcc_k.tolist()}


rule = result["decision_rule"]
table = pa.concat_tables([pq.read_table(path, columns=["stream", "label", f"{rule}_prediction"])
                          for path in sorted(glob.glob(os.path.join(latest, "predictions-rank*.parquet")))])
y = table["label"].to_numpy()
pred = table[f"{rule}_prediction"].to_numpy()
streams, index = np.unique(table["stream"].to_numpy(zero_copy_only=False), return_inverse=True)
k = len(result["classes"])
keep = y >= 0
cms = np.bincount(index[keep] * k * k + y[keep] * k + pred[keep], minlength=len(streams) * k * k).reshape(len(streams), k, k)
point = summary(cms.sum(0))
rng = np.random.default_rng(42)
draws = [summary(cms[rng.integers(0, len(streams), len(streams))].sum(0)) for _ in range(2000)]
interval = {m: [float(np.percentile([d[m] for d in draws], 2.5)), float(np.percentile([d[m] for d in draws], 97.5))]
            for m in ("macro_mcc", "macro_f1", "accuracy", "mcc")}
per_class = {name: {"f1": point["per_class_f1"][i], "mcc": point["per_class_mcc"][i],
                    "f1_ci95": [float(np.percentile([d["per_class_f1"][i] for d in draws], q)) for q in (2.5, 97.5)]}
             for i, name in enumerate(result["classes"])}
report = {"decision_rule": rule, "segments": int(len(streams)), "rows": int(keep.sum()), "resamples": 2000,
          "method": "percentile bootstrap over test segments (graph-state episodes)", "point": {m: point[m] for m in interval},
          "ci95": interval, "per_class": per_class}
json.dump(report, open(os.path.join(latest, "bootstrap_ci.json"), "w"), indent=1)
print(json.dumps({"point": report["point"], "ci95": interval, "segments": report["segments"]}, indent=1))
log("train kernel done")
'''

LANES = ["--lr", "0.001", "--train_lanes", "8"]
CONTEXT = r'''
pip("ruptures", "torch-geometric", "torchmetrics", "pydantic", "filelock", "psutil")
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
source = os.path.dirname(matches[0])
out = f"/data/output/{PREPARED_NAME}"
os.makedirs(out, exist_ok=True)
for name in ("manifest.json", "preprocess.json", "split_audit.json", "exporter_timeouts.frozen.json"):
    if os.path.exists(os.path.join(source, name)):
        shutil.copy2(os.path.join(source, name), out)
if not os.path.exists(os.path.join(out, "prepared")):
    os.symlink(os.path.join(source, "prepared"), os.path.join(out, "prepared"))
try:
    code = sh([sys.executable, PIPELINE, "--mode", "export_context", "--output_dir", out, "--gpus", "2",
               "--precision", "fp16", "--tf32", "off", *CONTEXT_ARGS], check=False)
    show_failures(out)
finally:
    os.unlink(os.path.join(out, "prepared"))
if code:
    raise SystemExit(code)
for split in ("train", "val", "calibration", "test"):
    print(split, sorted(os.listdir(os.path.join(out, "context", split))))
log("context kernel done")
'''

MODELS = r'''
pip("ruptures", "torch-geometric", "torchmetrics", "pydantic", "filelock", "psutil", "httpx")
import numpy as np
import torch
log("cuda", torch.cuda.is_available(), torch.cuda.device_count(), "cpus", os.cpu_count())
# Q1 NetFlow baselines (DNN, NEGSC, EULER) re-implemented in run_pipeline.py --mode baseline_models.
# JOBS: (prepared directory name, models, CUDA device or None). Jobs run in parallel, one device each.
processes, outputs = [], []
for prepared_name, models, device in JOBS:
    matches = sorted(glob.glob(f"/data/input/**/{prepared_name}/manifest.json", recursive=True))
    source = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
    out = f"/data/output/{prepared_name}"
    os.makedirs(out, exist_ok=True)
    for name in ("manifest.json", "preprocess.json", "split_audit.json", "exporter_timeouts.frozen.json"):
        if os.path.exists(os.path.join(source, name)):
            shutil.copy2(os.path.join(source, name), out)
    if not os.path.exists(os.path.join(out, "prepared")):
        os.symlink(os.path.join(source, "prepared"), os.path.join(out, "prepared"))
    env = dict(os.environ, BASELINE_BUDGET_HOURS=str(BUDGET_HOURS), BASELINE_SMOKE="1" if SMOKE else "0",
               BASELINE_THREADS=str(max(1, (os.cpu_count() or 1) // len(JOBS))),
               CUDA_VISIBLE_DEVICES="" if device is None else str(device))
    command = [sys.executable, PIPELINE, "--mode", "baseline_models", "--output_dir", out, "--baselines", ",".join(models),
               *MODEL_ARGS]
    log("$", " ".join(command), "device", device)
    processes.append(subprocess.Popen(command, env=env))
    outputs.append((out, prepared_name))
codes = [p.wait() for p in processes]
for out, _ in outputs:
    show_failures(out)
    os.unlink(os.path.join(out, "prepared"))  # never re-export the mounted input
    shutil.rmtree(os.path.join(out, "papers"), ignore_errors=True)  # verified PDFs are not redistributed
if any(codes):
    raise SystemExit(f"model baselines failed: {codes}")
for out, prepared_name in outputs:
    folder = os.path.join(out, "baselines_label_control" if "--label_permutation_seed" in MODEL_ARGS else "baselines")
    report = json.load(open(os.path.join(folder, "model_baselines.json")))
    print(prepared_name, json.dumps({"self_checks": report["self_checks"], "excluded": report["excluded"],
                                     "smoke_run": report.get("smoke_run")}, indent=1))
    for bid, m in report["models"].items():
        t = m["test"]
        if "binary" in t and "macro_mcc" not in t:
            log(prepared_name, bid, "delta", m.get("delta_s"), "binary", {k: t["binary"].get(k) for k in
                ("mcc", "detection_rate", "false_alarm_rate", "roc_auc", "pr_auc")}, "CI", t["ci95_segment_bootstrap"])
            print(json.dumps({"delta_selection": m.get("delta_selection"), "cutoff": m.get("cutoff"),
                              "training": {k: v for k, v in m.get("training", {}).items() if k != "history"}}, indent=1))
            continue
        log(prepared_name, bid, m["decision_rule"], {k: round(t[k], 4) for k in ("macro_mcc", "mcc", "accuracy", "macro_f1")},
            "binary MCC", round(t["binary"]["mcc"], 4), "CI", t["ci95_segment_bootstrap"]["macro_mcc"])
        print(json.dumps({"validation_macro_mcc": m["validation_macro_mcc"],
                          "training": {k: v for k, v in m.get("training", {}).items() if k != "history"}}, indent=1))
        for name, row in t["per_class"].items():
            print(f"{name:>24} P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} n={row['support']}")
    # The post-hoc takes the unseen-feature-vector flags from the first baseline file: they must be identical.
    reference = sorted(glob.glob("/data/input/**/baselines/LightGBM_test_predictions.npz", recursive=True))
    if reference and folder.endswith("baselines"):
        for bid in report["models"]:
            d = np.load(os.path.join(folder, f"{bid}_test_predictions.npz"), allow_pickle=False)
            o = np.argsort(d["source_id"])
            for path in reference:
                r = np.load(path, allow_pickle=False)
                ro = np.argsort(r["source_id"])
                if len(ro) != len(o) or not np.array_equal(d["source_id"][o], r["source_id"][ro]):
                    continue
                log(prepared_name, bid, "same test rows as", path, "labels", np.array_equal(d["label"][o], r["label"][ro]),
                    "novel flags", np.array_equal(d["novel"][o], r["novel"][ro]))
log("model baselines kernel done")
'''

DATASETS = {
    "ton": dict(name="NF-ToN-IoT-v3", source="seyhed/nf-ton-iot-v3", csv="NF-ToN-IoT-v3.csv", fairdata=None,
                site_layout=None, extra=[], runs=[("", "nids-ton", [])],
                train=["--epochs", "6", "--patience", "3", "--batch_size", "4096", *LANES]),
    # The prefix-derived replay guard for CIC is 520 s (a long same-key gap, not an export delay), which
    # would purge most of the short CIC campaigns. The documented UQ nProbe export contract bounds the
    # delay instead: records are exported within two minutes (active 120 s, idle 60 s, scan <= 1 s).
    "cic": dict(name="NF-CSE-CIC-IDS2018-v3", source="ndayisabae/nf-cicids2018-v3", csv="NF-CICIDS2018-v3.csv",
                fairdata=None, site_layout=None,
                contract={"mode": "exporter_contract", "exporter_id": "uq-nprobe-nf-v3",
                          "source": ("UQ NetFlow v3 release: flows exported in intervals of at most two minutes "
                                     "(nProbe active timeout 120 s, idle timeout 60 s, scan <= 1 s)"),
                          "delay_hi_s": 121.0},
                # Short, long-connection DoS campaigns: purge on flow activity (state is reset per segment).
                extra=["--split_purge_on", "activity",
                       "--split_block_candidates_s", "120,180,240,300,360,480,600,900,1200,1800"],
                runs=[("", "nids-cic", [])],
                train=["--epochs", "6", "--patience", "3", "--batch_size", "4096", *LANES]),
    # Official fields-preserved 5G-NIDD (row-identical to the public Combined.csv, plus addresses and
    # capture clocks). Attacks are short bursts inside each capture, so the leak-free closed-set
    # protocol is cross-site: train/val/calibration on one base station, test on the other.
    "5g": dict(name="5G-NIDD", source="humera11/5g-nidd-dataset", csv=None,
               fairdata=["9d13ef28-2ca7-44b0-9950-225359afac65", "/BTS1_BTS2_fields_preserved.zip",
                         "625f4d178c8a8224018cdedaba09ae397c804853c8db0ee1108773fbcf8f2a9d"],
               site_layout={"file": "/BTS_1.zip", "sha256": "01e6c4669e644c7f8b54bfb0e53c4d8c2232c547cc7fdab1a63539c8ca9589c0"},
               extra=["--argus_clock", "session", "--split_block_candidates_s", "10,15,20,30,45,60,90,120,180,300"],
               runs=[("-bs1", "nids-5g-bs1", ["--split_scope", "site", "--split_site_source", "BS1"]),
                     ("-bs2", "nids-5g-bs2", ["--split_scope", "site", "--split_site_source", "BS2"])],
               train=["--epochs", "12", "--patience", "4", "--batch_size", "2048", *LANES]),
}
# Host-fingerprint ablation: the same cross-site protocol without TTL/hop-count fields.
DATASETS["5gs"] = dict(DATASETS["5g"], extra=[*DATASETS["5g"]["extra"], "--exclude_features", "sTtl,dTtl,sHops,dHops"],
                       runs=[("-bs1", "nids-5gs-bs1", ["--split_scope", "site", "--split_site_source", "BS1"]),
                             ("-bs2", "nids-5gs-bs2", ["--split_scope", "site", "--split_site_source", "BS2"])])
TITLES = {"ton": "ToN", "cic": "CIC", "5g": "5G", "5gs": "5GS"}
VARIANTS = {
    "prepare-debug": dict(dataset="ton", kind="prepare", title="CTG ToN Prepare Debug", gpu=False, sources=[],
                          settings={"WINDOW_ROWS": 250000, "WINDOW_STARTS": [0, 3400000, 6800000, 10200000, 13600000,
                                                                              17000000, 20400000, 23800000, 27200000],
                                    "RUNS": [("nids-ton-debug", [])], "EXTRA": []}),
    "train-debug": dict(dataset="ton", kind="train", title="CTG ToN Train Debug", gpu=True,
                        sources=[f"{OWNER}/ctg-ton-prepare-debug"],
                        settings={"PREPARED_NAME": "nids-ton-debug",
                                  "TRAIN_ARGS": ["--epochs", "2", "--patience", "2", "--batch_size", "2048", *LANES]}),
}
for key, ds in DATASETS.items():
    tag = TITLES[key]
    VARIANTS[f"{key}-prepare-full"] = dict(dataset=key, kind="prepare", title=f"CTG {tag} Prepare Full", gpu=False, sources=[],
                                           settings={"WINDOW_ROWS": 0, "WINDOW_STARTS": [], "EXTRA": ds["extra"],
                                                     "RUNS": [(out, extra) for _, out, extra in ds["runs"]]})
    for suffix, out, _ in ds["runs"]:
        label, stem = f"{tag} {suffix.strip('-').upper()}".strip(), f"{key}{suffix}"
        VARIANTS[f"{stem}-baselines-full"] = dict(dataset=key, kind="baselines", title=f"CTG {label} Baselines Full",
                                                  gpu=False, sources=[f"{OWNER}/ctg-{key}-prepare-full"],
                                                  settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-train-full"] = dict(dataset=key, kind="train", title=f"CTG {label} Train Full", gpu=True,
                                              sources=[f"{OWNER}/ctg-{key}-prepare-full"],
                                              settings={"PREPARED_NAME": out, "TRAIN_ARGS": ds["train"]})
        VARIANTS[f"{stem}-posthoc-full"] = dict(dataset=key, kind="posthoc", title=f"CTG {label} Posthoc Full", gpu=False,
                                                sources=[f"{OWNER}/ctg-{stem}-train-full", f"{OWNER}/ctg-{stem}-baselines-full"],
                                                settings={})
# Label-randomisation controls on 5G BS1->BS2 (training labels randomised; test must fall to chance).
VARIANTS["5g-bs1-control-train-full"] = dict(dataset="5g", kind="train", title="CTG 5G BS1 Control Train Full", gpu=True,
                                              sources=[f"{OWNER}/ctg-5g-prepare-full"],
                                              settings={"PREPARED_NAME": "nids-5g-bs1",
                                                        "TRAIN_ARGS": [*DATASETS["5g"]["train"], "--label_permutation_seed", "7"]})
VARIANTS["5g-bs1-control-baselines-full"] = dict(dataset="5g", kind="baselines", title="CTG 5G BS1 Control Baselines Full",
                                                  gpu=False, sources=[f"{OWNER}/ctg-5g-prepare-full"],
                                                  settings={"PREPARED_NAME": "nids-5g-bs1", "LABEL_CONTROL": True})
# Improved model (v2 encodings, the run_pipeline defaults) with independent seeds for statistical reliability;
# the ensemble post-hoc averages seed probabilities and picks its decision rule on validation predictions.
V2_SEEDS = (42, 43, 44)
for key, ds in DATASETS.items():
    for suffix, out, _ in ds["runs"]:
        stem, label = f"{key}{suffix}", f"{TITLES[key]} {suffix.strip('-').upper()}".strip()
        for seed in V2_SEEDS:
            VARIANTS[f"{stem}-v2s{seed}-train-full"] = dict(
                dataset=key, kind="train", title=f"CTG {label} V2S{seed} Train Full", gpu=True,
                sources=[f"{OWNER}/ctg-{key}-prepare-full"],
                settings={"PREPARED_NAME": out, "TRAIN_ARGS": [*ds["train"], "--seed", str(seed)]})
        VARIANTS[f"{stem}-v2-posthoc-full"] = dict(
            dataset=key, kind="posthoc", title=f"CTG {label} V2 Posthoc Full", gpu=False,
            sources=[f"{OWNER}/ctg-{stem}-v2s{seed}-train-full" for seed in V2_SEEDS] + [f"{OWNER}/ctg-{stem}-baselines-full"],
            settings={})
# CT-Graph hybrid: model-free causal context export (GPU) and the context branch (CPU, 3 seeds).
for key, ds in DATASETS.items():
    for suffix, out, _ in ds["runs"]:
        stem, label = f"{key}{suffix}", f"{TITLES[key]} {suffix.strip('-').upper()}".strip()
        VARIANTS[f"{stem}-context-full"] = dict(
            dataset=key, kind="context", title=f"CTG {label} Context Full", gpu=True,
            sources=[f"{OWNER}/ctg-{key}-prepare-full"],
            settings={"PREPARED_NAME": out, "CONTEXT_ARGS": ["--batch_size", "2048", "--train_lanes", "1"]})
        VARIANTS[f"{stem}-ctgbdt-full"] = dict(
            dataset=key, kind="context_gbdt", title=f"CTG {label} Ctgbdt Full", gpu=False,
            sources=[f"{OWNER}/ctg-{key}-prepare-full", f"{OWNER}/ctg-{stem}-context-full"],
            settings={"PREPARED_NAME": out, "SEEDS": [42, 43, 44]})
        VARIANTS[f"{stem}-v2-posthoc-full"]["sources"].append(f"{OWNER}/ctg-{stem}-ctgbdt-full")
# Q1 NetFlow baselines (run_pipeline.py --mode baseline_models) on the frozen splits, CPU sessions.
# The large NF datasets get one session for the DNN and one for the two graph models; 5G-NIDD runs all three.
Q1_MODELS = ["DNN", "NEGSC", "EULER"]
for key, ds in DATASETS.items():
    for suffix, out, _ in ds["runs"]:
        stem, label = f"{key}{suffix}", f"{TITLES[key]} {suffix.strip('-').upper()}".strip()
        sources = [f"{OWNER}/ctg-{key}-prepare-full", f"{OWNER}/ctg-{stem}-baselines-full"]
        jobs = ({"dnn": ["DNN"], "gnnb": ["NEGSC", "EULER"]} if key in ("ton", "cic") else {"q1b": Q1_MODELS})
        for tag, models in jobs.items():
            VARIANTS[f"{stem}-{tag}-full"] = dict(
                dataset=key, kind="models", title=f"CTG {label} {tag.capitalize()} Full", gpu=False, sources=sources,
                settings={"JOBS": [(out, models, None)], "MODEL_ARGS": [], "BUDGET_HOURS": 10.0, "SMOKE": False})
        # Final comparison: v2 graph seeds + CT-GBDT + tabular baselines + the Q1 NetFlow baselines.
        VARIANTS[f"{stem}-v2q1-posthoc-full"] = dict(
            VARIANTS[f"{stem}-v2-posthoc-full"], title=f"CTG {label} V2q1 Posthoc Full",
            sources=[*VARIANTS[f"{stem}-v2-posthoc-full"]["sources"], *[f"{OWNER}/ctg-{stem}-{tag}-full" for tag in jobs]])
# Temporal-lab recommendations (train/validation/calibration only; filled from results/temporal/<stem>).
TEMPORAL_LAB = {"ton": ["flow", "T_segz"],  # ToN lab: flow+T_segz (val +0.014, cal +0.006); 5G labs chose REL / flow
                "cic": ["flow", "T_count", "T_novelty", "T_flags", "T_volume", "T_timing", "T_segz"]}  # CIC lab: flow+T_all
# Validation-only fusion design study (never opens a test file); graph seeds = the runs that exist.
STUDY_SEEDS = {"ton": [42], "cic": [42]}
for key, ds in DATASETS.items():
    for suffix, out, _ in ds["runs"]:
        stem, label = f"{key}{suffix}", f"{TITLES[key]} {suffix.strip('-').upper()}".strip()
        VARIANTS[f"{stem}-study-full"] = dict(
            dataset=key, kind="study", title=f"CTG {label} Study Full", gpu=False,
            sources=[f"{OWNER}/ctg-{key}-prepare-full", f"{OWNER}/ctg-{stem}-ctgbdt-full",
                     *[f"{OWNER}/ctg-{stem}-v2s{seed}-train-full" for seed in STUDY_SEEDS.get(key, V2_SEEDS)]],
            settings={"PREPARED_NAME": out})
for key, ds in DATASETS.items():
    for suffix, out, _ in ds["runs"]:
        stem, label = f"{key}{suffix}", f"{TITLES[key]} {suffix.strip('-').upper()}".strip()
        VARIANTS[f"{stem}-feat-full"] = dict(dataset=key, kind="features", title=f"CTG {label} Feat Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-temporal-full"] = dict(dataset=key, kind="temporal", title=f"CTG {label} Temporal Full", gpu=False,
                                                 sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-ctgbdt6-full"] = dict(
            dataset=key, kind="ctgbdt6", title=f"CTG {label} Ctgbdt6 Full", gpu=False, sources=[f"{OWNER}/ctg-{key}-prepare-full"],
            settings={"PREPARED_NAME": out, "SEEDS": [42, 43, 44], "LAB_PARTS": TEMPORAL_LAB.get(stem, [])})
        VARIANTS[f"{stem}-collective-full"] = dict(dataset=key, kind="collective", title=f"CTG {label} Collective Full",
                                                   gpu=False, sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-screen-full"] = dict(dataset=key, kind="screen", title=f"CTG {label} Screen Full", gpu=False,
                                               sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-robust-full"] = dict(dataset=key, kind="robust", title=f"CTG {label} Robust Full", gpu=False,
                                               sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-mvdiag-full"] = dict(dataset=key, kind="mvdiag", title=f"CTG {label} Mvdiag Full", gpu=False,
                                               sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-robust2-full"] = dict(dataset=key, kind="robust2", title=f"CTG {label} Robust2 Full", gpu=False,
                                                sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-collectivem-full"] = dict(dataset=key, kind="collectivem", title=f"CTG {label} Collectivem Full",
                                                    gpu=False, sources=[f"{OWNER}/ctg-{key}-prepare-full"],
                                                    settings={"PREPARED_NAME": out, "MERGED": True})
        VARIANTS[f"{stem}-lab2-full"] = dict(dataset=key, kind="lab2", title=f"CTG {label} Lab2 Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-lab3-full"] = dict(dataset=key, kind="lab3", title=f"CTG {label} Lab3 Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-ctgbdt9-full"] = dict(dataset=key, kind="ctgbdt9", title=f"CTG {label} Ctgbdt9 Full", gpu=False,
                                                sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out, "SEEDS": [42, 43, 44]})
        VARIANTS[f"{stem}-final9-posthoc-full"] = dict(
            dataset=key, kind="posthoc", title=f"CTG {label} Final9 Posthoc Full", gpu=False,
            sources=[*[f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full" for seed in (42, 43, 44)],
                     f"{OWNER}/ctg-{stem}-ctgbdt9-full", f"{OWNER}/ctg-{stem}-baselines-full",
                     *[f"{OWNER}/ctg-{stem}-{tag}-full" for tag in (("dnn", "gnnb") if key in ("ton", "cic") else ("q1b",))]],
            settings={})
        VARIANTS[f"{stem}-lab4-full"] = dict(dataset=key, kind="lab4", title=f"CTG {label} Lab4 Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-lab5-full"] = dict(dataset=key, kind="lab5", title=f"CTG {label} Lab5 Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-root-full"] = dict(dataset=key, kind="rootcause", title=f"CTG {label} Root Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-deep-full"] = dict(dataset=key, kind="deep", title=f"CTG {label} Deep Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
        VARIANTS[f"{stem}-hypo-full"] = dict(dataset=key, kind="hypothesis", title=f"CTG {label} Hypo Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out})
VARIANTS["unittest"] = dict(dataset="ton", kind="unittest", title="CTG ToN Unittest", gpu=False, sources=[], settings={})
VARIANTS["train1-debug"] = dict(dataset="ton", kind="train", title="CTG ToN Train1 Debug", gpu=True,
                                sources=[f"{OWNER}/ctg-ton-prepare-debug"],
                                settings={"PREPARED_NAME": "nids-ton-debug",
                                          "TRAIN_ARGS": ["--epochs", "2", "--patience", "2", "--batch_size", "2048", *LANES,
                                                         "--gpus", "1"]})
VARIANTS["models-debug"] = dict(dataset="ton", kind="models", title="CTG ToN Models Debug", gpu=False,
                                sources=[f"{OWNER}/ctg-ton-prepare-debug"],
                                settings={"JOBS": [("nids-ton-debug", Q1_MODELS, None)], "MODEL_ARGS": [],
                                          "BUDGET_HOURS": 1.0, "SMOKE": True})
VARIANTS["context-debug"] = dict(dataset="ton", kind="context", title="CTG ToN Context Debug", gpu=True,
                                 sources=[f"{OWNER}/ctg-ton-prepare-debug"],
                                 settings={"PREPARED_NAME": "nids-ton-debug", "CONTEXT_ARGS": ["--batch_size", "2048", "--train_lanes", "1"]})
VARIANTS["ctgbdt-debug"] = dict(dataset="ton", kind="context_gbdt", title="CTG ToN Ctgbdt Debug", gpu=False,
                                sources=[f"{OWNER}/ctg-ton-prepare-debug", f"{OWNER}/ctg-ton-context-debug"],
                                settings={"PREPARED_NAME": "nids-ton-debug", "SEEDS": [42]})
# Longer schedules where validation had not converged at the last epoch (patience never triggered).
for key in ("ton", "cic"):
    for seed in (42, 43):
        VARIANTS[f"{key}-v2e12s{seed}-train-full"] = dict(
            dataset=key, kind="train", title=f"CTG {TITLES[key]} V2E12S{seed} Train Full", gpu=True,
            sources=[f"{OWNER}/ctg-{key}-prepare-full"],
            settings={"PREPARED_NAME": DATASETS[key]["runs"][0][1],
                      "TRAIN_ARGS": ["--epochs", "12", "--patience", "3", "--batch_size", "4096", *LANES, "--seed", str(seed)]})
# v3 architecture (cardinality-aware endpoint memory + residual flow encoder + jumping-knowledge head; the
# run_pipeline defaults) and its ablations, same 12-epoch schedule as the v2e12 reference run.
V4 = ["--encoder_depth", "0", "--no-jump_knowledge", "--patience", "6"]
V3_ABLATIONS = {"v3": [], "v3nc": ["--no-cardinality"], "v3nr": ["--replay_per_class", "0"],
                "v3enc0": ["--encoder_depth", "0", "--no-jump_knowledge"],
                "v4a": V4, "v4b": [*V4, "--context_norm"],
                "v5a": [*V4, "--context_norm", "--numeric_embedding", "ple"],
                "v5b": [*V4, "--context_norm", "--numeric_embedding", "ple", "--ema_decay", "0.9998"],
                "v5c": [*V4, "--context_norm", "--numeric_embedding", "ple", "--replay_per_class", "32", "--replay_items", "8"],
                "v5d": [*V4, "--context_norm", "--numeric_embedding", "ple", "--K", "12", "--Ks", "6", "--Ku", "6"]}
for key in DATASETS:
    for suffix, out, _ in DATASETS[key]["runs"]:
        stem, label = f"{key}{suffix}", f"{TITLES[key]} {suffix.strip('-').upper()}".strip()
        base = ["--epochs", "12", "--patience", "3", *DATASETS[key]["train"][4:]] if key in ("ton", "cic") else DATASETS[key]["train"]
        for tag, flags in V3_ABLATIONS.items():
            for seed in (42, 43, 44):
                VARIANTS[f"{stem}-{tag}s{seed}-train-full"] = dict(
                    dataset=key, kind="train", title=f"CTG {label} {tag.upper()}S{seed} Train Full", gpu=True,
                    sources=[f"{OWNER}/ctg-{key}-prepare-full"],
                    settings={"PREPARED_NAME": out, "TRAIN_ARGS": [*base, *flags, "--seed", str(seed)]})
# v6r graph branch (CIC): the frozen v4b configuration trained on relational-augmented prepared data.
for seed in (42, 43, 44):
    VARIANTS[f"cic-v6rs{seed}-train-full"] = dict(
        dataset="cic", kind="train", title=f"CTG CIC V6RS{seed} Train Full", gpu=True,
        sources=[f"{OWNER}/ctg-cic-relprep-full"],
        settings={"PREPARED_NAME": "nids-cic-rel",
                  "TRAIN_ARGS": [*VARIANTS[f"cic-v4bs{seed}-train-full"]["settings"]["TRAIN_ARGS"]]})
VARIANTS["cic-decide-full"] = dict(
    dataset="cic", kind="decide", title="CTG CIC Decide Full", gpu=False,
    sources=[*[f"{OWNER}/ctg-cic-v4bs{seed}-train-full" for seed in (42, 43, 44)],
             f"{OWNER}/ctg-cic-ctgbdt7-full", f"{OWNER}/ctg-cic-ctgbdt5-full"],
    datasets=[f"{OWNER}/ctg-cic-v6r8-val"], settings={})
# v6r8 final on runner: v6r8 graph runs (dataset, full run layout) + the CIC headline context branch (CT-GBDT v5).
VARIANTS["cic-final13-posthoc-full"] = dict(
    dataset="cic", kind="posthoc", title="CTG CIC Final13 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-cic-ctgbdt5-full", f"{OWNER}/ctg-cic-baselines-full", f"{OWNER}/ctg-cic-dnn-full", f"{OWNER}/ctg-cic-gnnb-full"],
    datasets=[f"{OWNER}/ctg-cic-v6r8s{seed}-graph" for seed in (42, 43, 44)], settings={})
# ToN v6r8: the same relational augmentation and v4b configuration with batch 8192.
VARIANTS["ton-relprep-full"] = dict(dataset="ton", kind="relprep", title="CTG ToN Relprep Full", gpu=False,
                                    sources=[f"{OWNER}/ctg-ton-prepare-full"], settings={"PREPARED_NAME": "nids-ton"})
for seed in (42, 43, 44):
    args = list(VARIANTS[f"ton-v4bs{seed}-train-full"]["settings"]["TRAIN_ARGS"])
    args[args.index("--batch_size") + 1] = "8192"
    VARIANTS[f"ton-v6r8s{seed}-train-full"] = dict(
        dataset="ton", kind="train", title=f"CTG ToN V6R8S{seed} Train Full", gpu=True,
        sources=[f"{OWNER}/ctg-ton-relprep-full"], settings={"PREPARED_NAME": "nids-ton-rel", "TRAIN_ARGS": args})
VARIANTS["ton-decide787-full"] = dict(
    dataset="ton", kind="decide", title="CTG ToN Decide787 Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full", f"{OWNER}/ctg-ton-v6r8s44-train-full",
             f"{OWNER}/ctg-ton-smooth787-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-v6r8s42-graph", f"{OWNER}/ctg-ton-v6r8s43-graph"],
    settings={"CONTEXT_KEYS": ("smooth787",)})
VARIANTS["ton-final14787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Final14787 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-v6r8s44-train-full", f"{OWNER}/ctg-ton-smooth787-full"],
    datasets=[f"{OWNER}/ctg-ton-v6r8s42-graph", f"{OWNER}/ctg-ton-v6r8s43-graph", f"{OWNER}/ctg-ton-baseline-predictions"],
    settings={})
# 5G BS2 graph with relational context computed on the MERGED monitor timeline (the boundary-robust form), paired with a
# v4b re-run (v4bc) under the same pipeline so that both write calibration predictions (BS2 validation is saturated).
VARIANTS["5gs-bs2-relprepm-full"] = dict(dataset="5gs", kind="relprep", title="CTG 5GS BS2 Relprepm Full", gpu=False,
                                         sources=[f"{OWNER}/ctg-5gs-prepare-full"],
                                         settings={"PREPARED_NAME": "nids-5gs-bs2", "MERGED": True})
for seed in (42, 43, 44):
    base_args = list(VARIANTS[f"5gs-bs2-v4bs{seed}-train-full"]["settings"]["TRAIN_ARGS"])
    VARIANTS[f"5gs-bs2-v4bcs{seed}-train-full"] = dict(
        dataset="5gs", kind="train", title=f"CTG 5GS BS2 V4BCS{seed} Train Full", gpu=True,
        sources=[f"{OWNER}/ctg-5gs-prepare-full"], settings={"PREPARED_NAME": "nids-5gs-bs2", "TRAIN_ARGS": base_args})
    VARIANTS[f"5gs-bs2-v6rms{seed}-train-full"] = dict(
        dataset="5gs", kind="train", title=f"CTG 5GS BS2 V6RMS{seed} Train Full", gpu=True,
        sources=[f"{OWNER}/ctg-5gs-bs2-relprepm-full"], settings={"PREPARED_NAME": "nids-5gs-bs2-relm", "TRAIN_ARGS": base_args})
VARIANTS["5gs-bs2-decide-full"] = dict(
    dataset="5gs", kind="decide", title="CTG 5GS BS2 Decide Full", gpu=False,
    sources=[*[f"{OWNER}/ctg-5gs-bs2-{t}s{seed}-train-full" for t in ("v4bc", "v6rm") for seed in (42, 43, 44)]],
    datasets=[f"{OWNER}/ctg-5gs-bs2-ctgbdt4-predictions"],
    settings={"FAMILIES": ("v6rm", "v4bc"), "REF": "v4bc", "RULE": "cal", "CONTEXT_KEYS": ("ctgbdt4",)})
# BS2 test-error analysis (user request 2026-10-01, after the v4bc / v6rm graph runs; diagnosis only).
VARIANTS["5gs-bs2-err-full"] = dict(
    dataset="5gs", kind="bs2err", title="CTG 5GS BS2 Err Full", gpu=False,
    sources=[f"{OWNER}/ctg-5gs-prepare-full", *[f"{OWNER}/ctg-5gs-bs2-{t}s{seed}-train-full" for t in ("v4bc", "v6rm") for seed in (42, 43, 44)]],
    datasets=[f"{OWNER}/ctg-5gs-bs2-ctgbdt4-predictions"], settings={"PREPARED_NAME": "nids-5gs-bs2"})
# The same diagnosis on the FROZEN headline configuration (v4b graph seeds + CT-GBDT v4), manuscript review round 1.
VARIANTS["5gs-bs2-errv4b-full"] = dict(
    dataset="5gs", kind="bs2err", title="CTG 5GS BS2 Errv4b Full", gpu=False,
    sources=[f"{OWNER}/ctg-5gs-prepare-full", *[f"{OWNER}/ctg-5gs-bs2-v4bs{seed}-train-full" for seed in (42, 43, 44)],
             f"{OWNER}/ctg-5gs-bs2-ctgbdt4-full"],
    settings={"PREPARED_NAME": "nids-5gs-bs2", "FAMILIES": ("v4b",)})
VARIANTS["5gs-bs2-final15-posthoc-full"] = dict(
    dataset="5gs", kind="posthoc", title="CTG 5GS BS2 Final15 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-5gs-bs2-v6rms{seed}-train-full" for seed in (42, 43, 44)],
    datasets=[f"{OWNER}/ctg-5gs-bs2-ctgbdt4-predictions", f"{OWNER}/ctg-5gs-bs2-baseline-predictions"], settings={})
VARIANTS["cic-relprep-full"] = dict(dataset="cic", kind="relprep", title="CTG CIC Relprep Full", gpu=False,
                                    sources=[f"{OWNER}/ctg-cic-prepare-full"], settings={"PREPARED_NAME": "nids-cic"})
# Final v4b pipeline: context export with the frozen v4b context settings, the CT-GBDT context branch on it, and
# the final post-hoc (v4b graph seeds + CT-GBDT + all baselines). Cross-account inputs arrive as private datasets.
for key in DATASETS:
    for suffix, out, _ in DATASETS[key]["runs"]:
        stem, label = f"{key}{suffix}", f"{TITLES[key]} {suffix.strip('-').upper()}".strip()
        VARIANTS[f"{stem}-ctx4-full"] = dict(
            dataset=key, kind="context", title=f"CTG {label} Ctx4 Full", gpu=True,
            sources=[f"{OWNER}/ctg-{key}-prepare-full"],
            settings={"PREPARED_NAME": out, "CONTEXT_ARGS": ["--batch_size", "2048", "--train_lanes", "1", "--context_norm"]})
        VARIANTS[f"{stem}-ctgbdt4-full"] = dict(
            dataset=key, kind="context_gbdt", title=f"CTG {label} Ctgbdt4 Full", gpu=False,
            sources=[f"{OWNER}/ctg-{key}-prepare-full", f"{OWNER}/ctg-{stem}-ctx4-full"],
            settings={"PREPARED_NAME": out, "SEEDS": [42, 43, 44]})
        VARIANTS[f"{stem}-ctgbdt5-full"] = dict(
            dataset=key, kind="ctgbdt5", title=f"CTG {label} Ctgbdt5 Full", gpu=False,
            sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out, "SEEDS": [42, 43, 44]})
        VARIANTS[f"{stem}-final5-posthoc-full"] = dict(
            dataset=key, kind="posthoc", title=f"CTG {label} Final5 Posthoc Full", gpu=False,
            sources=[*[f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full" for seed in (42, 43, 44)],
                     f"{OWNER}/ctg-{stem}-ctgbdt5-full", f"{OWNER}/ctg-{stem}-baselines-full",
                     *[f"{OWNER}/ctg-{stem}-{tag}-full" for tag in (("dnn", "gnnb") if key in ("ton", "cic") else ("q1b",))]],
            settings={})
        VARIANTS[f"{stem}-ctgbdt7-full"] = dict(
            dataset=key, kind="ctgbdt7", title=f"CTG {label} Ctgbdt7 Full", gpu=False,
            sources=[f"{OWNER}/ctg-{key}-prepare-full"], settings={"PREPARED_NAME": out, "SEEDS": [42, 43, 44]})
        VARIANTS[f"{stem}-final7-posthoc-full"] = dict(
            dataset=key, kind="posthoc", title=f"CTG {label} Final7 Posthoc Full", gpu=False,
            sources=[*[f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full" for seed in (42, 43, 44)],
                     f"{OWNER}/ctg-{stem}-ctgbdt7-full", f"{OWNER}/ctg-{stem}-baselines-full",
                     *[f"{OWNER}/ctg-{stem}-{tag}-full" for tag in (("dnn", "gnnb") if key in ("ton", "cic") else ("q1b",))]],
            settings={})
        VARIANTS[f"{stem}-ctgbdt8-full"] = dict(
            dataset=key, kind="ctgbdt8", title=f"CTG {label} Ctgbdt8 Full", gpu=False, sources=[f"{OWNER}/ctg-{key}-prepare-full"],
            settings={"PREPARED_NAME": out, "SEEDS": [42, 43, 44] if key.startswith("5g") else [42],
                      "VIEWS": ["flow+REL", "flow+T_segz", "all"]})
        VARIANTS[f"{stem}-final8-posthoc-full"] = dict(
            dataset=key, kind="posthoc", title=f"CTG {label} Final8 Posthoc Full", gpu=False,
            sources=[*[f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full" for seed in (42, 43, 44)],
                     f"{OWNER}/ctg-{stem}-ctgbdt8-full", f"{OWNER}/ctg-{stem}-baselines-full",
                     *[f"{OWNER}/ctg-{stem}-{tag}-full" for tag in (("dnn", "gnnb") if key in ("ton", "cic") else ("q1b",))]],
            settings={})
        VARIANTS[f"{stem}-final-posthoc-full"] = dict(
            dataset=key, kind="posthoc", title=f"CTG {label} Final Posthoc Full", gpu=False,
            sources=[*[f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full" for seed in (42, 43, 44)],
                     f"{OWNER}/ctg-{stem}-ctgbdt4-full", f"{OWNER}/ctg-{stem}-baselines-full",
                     *[f"{OWNER}/ctg-{stem}-{tag}-full" for tag in (("dnn", "gnnb") if key in ("ton", "cic") else ("q1b",))]],
            settings={})
# ToN final post-hoc on runner: seeds 43/44 are runner kernels; seed 42 (runner), the CT-GBDT branch (runner) and
# the baseline predictions (runner) arrive as private datasets with their original directory layout.
VARIANTS["ton-final787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Final787 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-baseline-predictions", f"{OWNER}/ctg-ton-ctgbdt4-predictions"],
    settings={})
# Second-stage ToN comparisons on runner: identical inputs to final787 except the context branch.
VARIANTS["ton-final6787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Final6787 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full", f"{OWNER}/ctg-ton-ctgbdt6-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-baseline-predictions"], settings={})
VARIANTS["ton-final7787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Final7787 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-baseline-predictions", f"{OWNER}/ctg-ton-ctgbdt7-predictions"],
    settings={})
# ToN causal temporal smoothing of the frozen CT-GBDT v4 branch (lab 4) and its final comparison (runner).
VARIANTS["ton-smooth787-full"] = dict(
    dataset="ton", kind="smooth", title="CTG ToN Smooth787 Full", gpu=False, sources=[f"{OWNER}/ctg-ton-prepare-full"],
    datasets=[f"{OWNER}/ctg-ton-ctgbdt4-predictions"], settings={"PREPARED_NAME": "nids-ton"})
VARIANTS["ton-final10787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Final10787 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full", f"{OWNER}/ctg-ton-smooth787-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-baseline-predictions"], settings={})
# 5G BS1: pair-keyed causal smoothing of the frozen CT-GBDT v4 branch (lab 5) and its final comparison.
VARIANTS["5gs-bs1-smooth-full"] = dict(
    dataset="5gs", kind="smooth", title="CTG 5GS BS1 Smooth Full", gpu=False,
    sources=[f"{OWNER}/ctg-5gs-prepare-full", f"{OWNER}/ctg-5gs-bs1-ctgbdt4-full"],
    settings={"PREPARED_NAME": "nids-5gs-bs1", "KEY": "pair"})
VARIANTS["5gs-bs1-final10-posthoc-full"] = dict(
    dataset="5gs", kind="posthoc", title="CTG 5GS BS1 Final10 Posthoc Full", gpu=False,
    sources=[*[f"{OWNER}/ctg-5gs-bs1-v4bs{seed}-train-full" for seed in (42, 43, 44)], f"{OWNER}/ctg-5gs-bs1-smooth-full",
             f"{OWNER}/ctg-5gs-bs1-baselines-full", f"{OWNER}/ctg-5gs-bs1-q1b-full"], settings={})
# Graph-branch smoothing (validation-only adoption) and the corresponding final comparisons.
VARIANTS["ton-gsmooth787-full"] = dict(
    dataset="ton", kind="gsmooth", title="CTG ToN Gsmooth787 Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full",
             f"{OWNER}/ctg-ton-smooth787-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph"], settings={"PREPARED_NAME": "nids-ton", "KEY": "src"})
VARIANTS["ton-err787-full"] = dict(
    dataset="ton", kind="tonerr", title="CTG ToN Err787 Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full",
             f"{OWNER}/ctg-ton-smooth787-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph"], settings={"PREPARED_NAME": "nids-ton"})
VARIANTS["ton-scanlab787-full"] = dict(
    dataset="ton", kind="scanlab", title="CTG ToN Scanlab787 Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full",
             f"{OWNER}/ctg-ton-smooth787-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph"], settings={"PREPARED_NAME": "nids-ton"})
# Label-collision ablation ("cleaned benchmark"): drop lists + the headline post-hoc on the standard and cleaned test sets.
VARIANTS["ton-cleanids787-full"] = dict(dataset="ton", kind="cleanids", title="CTG ToN Cleanids787 Full", gpu=False,
                                        sources=[f"{OWNER}/ctg-ton-prepare-full"], settings={"PREPARED_NAME": "nids-ton"})
TON_HEADLINE = dict(sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full", f"{OWNER}/ctg-ton-smooth787-full"],
                    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-baseline-predictions"])
VARIANTS["ton-std787-posthoc-full"] = dict(dataset="ton", kind="posthoc", title="CTG ToN Std787 Posthoc Full", gpu=False,
                                           settings={}, **TON_HEADLINE)
for rule in ("c1", "c2", "c1c2"):
    VARIANTS[f"ton-clean{rule}787-posthoc-full"] = dict(
        dataset="ton", kind="posthoc", title=f"CTG ToN Clean{rule}787 Posthoc Full", gpu=False,
        sources=[*TON_HEADLINE["sources"], f"{OWNER}/ctg-ton-cleanids787-full"], datasets=TON_HEADLINE["datasets"],
        settings={"CLEAN_RULE": rule.upper()})
VARIANTS["5gs-bs2-cleanids-full"] = dict(dataset="5gs", kind="cleanids", title="CTG 5GS BS2 Cleanids Full", gpu=False,
                                         sources=[f"{OWNER}/ctg-5gs-prepare-full"], settings={"PREPARED_NAME": "nids-5gs-bs2"})
BS2_HEADLINE = [*[f"{OWNER}/ctg-5gs-bs2-v4bs{seed}-train-full" for seed in (42, 43, 44)], f"{OWNER}/ctg-5gs-bs2-ctgbdt4-full",
                f"{OWNER}/ctg-5gs-bs2-baselines-full", f"{OWNER}/ctg-5gs-bs2-q1b-full"]
VARIANTS["5gs-bs2-std-posthoc-full"] = dict(dataset="5gs", kind="posthoc", title="CTG 5GS BS2 Std Posthoc Full", gpu=False,
                                            sources=BS2_HEADLINE, settings={})
for rule in ("c1", "c2", "c1c2"):
    VARIANTS[f"5gs-bs2-clean{rule}-posthoc-full"] = dict(
        dataset="5gs", kind="posthoc", title=f"CTG 5GS BS2 Clean{rule} Posthoc Full", gpu=False,
        sources=[*BS2_HEADLINE, f"{OWNER}/ctg-5gs-bs2-cleanids-full"], settings={"CLEAN_RULE": rule.upper()})
# Cleaned-LABEL retrain (train side of the label-collision ablation): train / validation / calibration labels nulled under
# each rule (cleanprep; test untouched), the CPU models refitted on them (LightGBM + XGBoost baselines, the frozen
# flow-only CT-GBDT v4 branch with its validation gate against the standard branch), then the post-hoc on the cleaned
# test (CLEAN_RULE) and on the standard test. The graph branch is retrained on GPU only if the gate passes.
for rule in ("c1", "c2", "c1c2"):
    R = rule.upper()
    for stem, key_, prepared, prep_src, std_ct, headline in (
            ("5gs-bs2", "5gs", "nids-5gs-bs2", f"{OWNER}/ctg-5gs-prepare-full", f"{OWNER}/ctg-5gs-bs2-ctgbdt4-full", None),
            ("ton", "ton", "nids-ton", f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-ctgbdt4-full", None)):
        label = "5GS BS2" if stem == "5gs-bs2" else "ToN"
        VARIANTS[f"{stem}-cleanprep{rule}-full"] = dict(
            dataset=key_, kind="cleanprep", title=f"CTG {label} Cleanprep{rule} Full", gpu=False, sources=[prep_src],
            settings={"PREPARED_NAME": prepared, "CLEAN": R, "FIT_ONLY": False})
        VARIANTS[f"{stem}-cbase{rule}-full"] = dict(
            dataset=key_, kind="baselines", title=f"CTG {label} Cbase{rule} Full", gpu=False,
            sources=[f"{OWNER}/ctg-{stem}-cleanprep{rule}-full"], settings={"PREPARED_NAME": f"{prepared}-{rule}"})
        VARIANTS[f"{stem}-cct{rule}-full"] = dict(
            dataset=key_, kind="context_gbdt", title=f"CTG {label} Cct{rule} Full", gpu=False,
            sources=[f"{OWNER}/ctg-{stem}-cleanprep{rule}-full", std_ct],
            settings={"PREPARED_NAME": f"{prepared}-{rule}", "SEEDS": [42, 43, 44], "FLOW_ONLY": True, "COMPARE_TO_STANDARD": True})
    # ToN: the frozen source smoothing re-applied to the cleaned-label branch (standard prepared data: full traffic)
    VARIANTS[f"ton-csmooth{rule}-full"] = dict(
        dataset="ton", kind="smooth", title=f"CTG ToN Csmooth{rule} Full", gpu=False,
        sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-cct{rule}-full"],
        settings={"PREPARED_NAME": "nids-ton", "FROZEN_ADOPTED": True})
    BS2_CTRAIN = [*[f"{OWNER}/ctg-5gs-bs2-v4bs{seed}-train-full" for seed in (42, 43, 44)], f"{OWNER}/ctg-5gs-bs2-cct{rule}-full",
                  f"{OWNER}/ctg-5gs-bs2-cbase{rule}-full"]
    VARIANTS[f"5gs-bs2-ctrain{rule}-posthoc-full"] = dict(
        dataset="5gs", kind="posthoc", title=f"CTG 5GS BS2 Ctrain{rule} Posthoc Full", gpu=False,
        sources=[*BS2_CTRAIN, f"{OWNER}/ctg-5gs-bs2-cleanids-full"], settings={"CLEAN_RULE": R})
    VARIANTS[f"5gs-bs2-ctrain{rule}std-posthoc-full"] = dict(
        dataset="5gs", kind="posthoc", title=f"CTG 5GS BS2 Ctrain{rule}std Posthoc Full", gpu=False, sources=BS2_CTRAIN, settings={})
    # ToN post-hoc on runner (graph seeds 43/44 live there); the cleaned-label branch + baselines arrive as a dataset
    TON_CTRAIN = dict(sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full"],
                      datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-ctrain{rule}-branch"])
    VARIANTS[f"ton-ctrain{rule}787-posthoc-full"] = dict(
        dataset="ton", kind="posthoc", title=f"CTG ToN Ctrain{rule}787 Posthoc Full", gpu=False,
        sources=[*TON_CTRAIN["sources"], f"{OWNER}/ctg-ton-cleanids787-full"], datasets=TON_CTRAIN["datasets"],
        settings={"CLEAN_RULE": R})
    VARIANTS[f"ton-ctrain{rule}std787-posthoc-full"] = dict(
        dataset="ton", kind="posthoc", title=f"CTG ToN Ctrain{rule}std787 Posthoc Full", gpu=False, settings={}, **TON_CTRAIN)
# LEAKAGE-FREE cleaned-label retrain ("cf"): C1 collisions from train / validation / calibration labels only (the "c"
# variants above also used TEST labels, which on 5G BS2 nulled 100 % of the udpflood training labels; kept for the record).
for rule in ("c1", "c2", "c1c2"):
    R = rule.upper()
    for stem, key_, prepared, prep_src, std_ct in (
            ("5gs-bs2", "5gs", "nids-5gs-bs2", f"{OWNER}/ctg-5gs-prepare-full", f"{OWNER}/ctg-5gs-bs2-ctgbdt4-full"),
            ("ton", "ton", "nids-ton", f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-ctgbdt4-full")):
        label = "5GS BS2" if stem == "5gs-bs2" else "ToN"
        VARIANTS[f"{stem}-cfprep{rule}-full"] = dict(
            dataset=key_, kind="cleanprep", title=f"CTG {label} Cfprep{rule} Full", gpu=False, sources=[prep_src],
            settings={"PREPARED_NAME": prepared, "CLEAN": R, "FIT_ONLY": True})
        VARIANTS[f"{stem}-cfbase{rule}-full"] = dict(
            dataset=key_, kind="baselines", title=f"CTG {label} Cfbase{rule} Full", gpu=False,
            sources=[f"{OWNER}/ctg-{stem}-cfprep{rule}-full"], settings={"PREPARED_NAME": f"{prepared}-f{rule}"})
        VARIANTS[f"{stem}-cfct{rule}-full"] = dict(
            dataset=key_, kind="context_gbdt", title=f"CTG {label} Cfct{rule} Full", gpu=False,
            sources=[f"{OWNER}/ctg-{stem}-cfprep{rule}-full", std_ct],
            settings={"PREPARED_NAME": f"{prepared}-f{rule}", "SEEDS": [42, 43, 44], "FLOW_ONLY": True, "COMPARE_TO_STANDARD": True})
    VARIANTS[f"ton-cfsmooth{rule}-full"] = dict(
        dataset="ton", kind="smooth", title=f"CTG ToN Cfsmooth{rule} Full", gpu=False,
        sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-cfct{rule}-full"],
        settings={"PREPARED_NAME": "nids-ton", "FROZEN_ADOPTED": True})
    BS2_CF = [*[f"{OWNER}/ctg-5gs-bs2-v4bs{seed}-train-full" for seed in (42, 43, 44)], f"{OWNER}/ctg-5gs-bs2-cfct{rule}-full",
              f"{OWNER}/ctg-5gs-bs2-cfbase{rule}-full"]
    VARIANTS[f"5gs-bs2-cftrain{rule}-posthoc-full"] = dict(
        dataset="5gs", kind="posthoc", title=f"CTG 5GS BS2 Cftrain{rule} Posthoc Full", gpu=False,
        sources=[*BS2_CF, f"{OWNER}/ctg-5gs-bs2-cleanids-full"], settings={"CLEAN_RULE": R})
    VARIANTS[f"5gs-bs2-cftrain{rule}std-posthoc-full"] = dict(
        dataset="5gs", kind="posthoc", title=f"CTG 5GS BS2 Cftrain{rule}std Posthoc Full", gpu=False, sources=BS2_CF, settings={})
    TON_CF = dict(sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full"],
                  datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-cf{rule}-branch"])
    VARIANTS[f"ton-cftrain{rule}787-posthoc-full"] = dict(
        dataset="ton", kind="posthoc", title=f"CTG ToN Cftrain{rule}787 Posthoc Full", gpu=False,
        sources=[*TON_CF["sources"], f"{OWNER}/ctg-ton-cleanids787-full"], datasets=TON_CF["datasets"], settings={"CLEAN_RULE": R})
    VARIANTS[f"ton-cftrain{rule}std787-posthoc-full"] = dict(
        dataset="ton", kind="posthoc", title=f"CTG ToN Cftrain{rule}std787 Posthoc Full", gpu=False, settings={}, **TON_CF)
# ToN C1 cleaned-label GRAPH retrain (the C1 gate passed on ToN, +0.014 on cleaned validation; the pre-registered
# C1∪C2 gate failed, so this run is outside the pre-registration and reported as exploratory). Frozen v4b arguments.
for seed in (42, 43, 44):
    VARIANTS[f"ton-cfv4bc1s{seed}-train-full"] = dict(
        dataset="ton", kind="train", title=f"CTG ToN Cfv4bc1s{seed} Train Full", gpu=True,
        sources=[f"{OWNER}/ctg-ton-cfprepc1-full"],
        settings={"PREPARED_NAME": "nids-ton-fc1",
                  "TRAIN_ARGS": [*VARIANTS[f"ton-v4bs{seed}-train-full"]["settings"]["TRAIN_ARGS"]]})
# ToN C1 cleaned-label hybrid (exploratory): cleaned-label v4b graph seeds (s42 / s43 from runner as datasets, s44 on
# runner) + cleaned-label smoothed CT-GBDT + cleaned-label baselines, on the C1-cleaned test and on the standard test.
TON_CFG = dict(sources=[f"{OWNER}/ctg-ton-cfv4bc1s44-train-full"],
               datasets=[f"{OWNER}/ctg-ton-cfv4bc1s42-graph", f"{OWNER}/ctg-ton-cfv4bc1s43-graph", f"{OWNER}/ctg-ton-cfc1-branch"])
VARIANTS["ton-cfgc1787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Cfgc1787 Posthoc Full", gpu=False,
    sources=[*TON_CFG["sources"], f"{OWNER}/ctg-ton-cleanids787-full"], datasets=TON_CFG["datasets"], settings={"CLEAN_RULE": "C1"})
VARIANTS["ton-cfgc1std787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Cfgc1std787 Posthoc Full", gpu=False, settings={}, **TON_CFG)
# Inference latency (CPU side: GBDT baselines, CT-GBDT context branch, smoothing, relational features, fusion) at the
# FROZEN tree counts; the graph branch is timed in its frozen test pass (results.json). MICRO = graph micro-batch rows
# = batch_size // (2 GPUs * 8 lanes).
for stem, key_, prepared, micro, lgbm, xgbr, ct, feats, smooth_key in (
        ("ton", "ton", "nids-ton", 256, 108, 329, [108, 101, 90], "flow", "src"),
        ("cic", "cic", "nids-cic", 512, 92, 200, [300, 300, 300], "flow+REL", None),
        ("5gs-bs1", "5gs", "nids-5gs-bs1", 128, 53, 135, [53, 57, 56], "flow", "pair"),
        ("5gs-bs2", "5gs", "nids-5gs-bs2", 128, 53, 129, [53, 59, 59], "flow", None)):
    label = {"ton": "ToN", "cic": "CIC", "5gs-bs1": "5GS BS1", "5gs-bs2": "5GS BS2"}[stem]
    VARIANTS[f"{stem}-latency-full"] = dict(
        dataset=key_, kind="latency", title=f"CTG {label} Latency Full", gpu=False, sources=[f"{OWNER}/ctg-{key_}-prepare-full"],
        settings={"PREPARED_NAME": prepared, "MICRO": micro, "LGBM_TREES": lgbm, "XGB_ROUNDS": xgbr, "CT_TREES": ct,
                  "CT_FEATURES": feats, "SMOOTH_KEY": smooth_key})
    # Same benchmark plus the cost of each CT-GBDT seed alone (single-seed ablation, manuscript review round 1).
    VARIANTS[f"{stem}-latseed-full"] = dict(VARIANTS[f"{stem}-latency-full"], title=f"CTG {label} Latseed Full",
                                           settings={**VARIANTS[f"{stem}-latency-full"]["settings"], "PER_SEED": True})
# Streaming relational-context extraction at the flow collector: exactness vs the batch library, ns / flow, live state,
# peak flow-record rate and headroom (test split; label-free).
for stem, key_, prepared, label in (("ton", "ton", "nids-ton", "ToN"), ("cic", "cic", "nids-cic", "CIC"),
                                    ("5gs-bs1", "5gs", "nids-5gs-bs1", "5GS BS1"), ("5gs-bs2", "5gs", "nids-5gs-bs2", "5GS BS2")):
    VARIANTS[f"{stem}-relsys-full"] = dict(dataset=key_, kind="relsys", title=f"CTG {label} Relsys Full", gpu=False,
                                           sources=[f"{OWNER}/ctg-{key_}-prepare-full"], settings={"PREPARED_NAME": prepared})
# Matched graph-free baseline (major revision): GBDTs with the same 59 relational features and causal smoothing.
for stem, key_, prepared, label in (("ton", "ton", "nids-ton", "ToN"), ("cic", "cic", "nids-cic", "CIC"),
                                    ("5gs-bs1", "5gs", "nids-5gs-bs1", "5GS BS1"), ("5gs-bs2", "5gs", "nids-5gs-bs2", "5GS BS2")):
    VARIANTS[f"{stem}-matched-full"] = dict(dataset=key_, kind="matched", title=f"CTG {label} Matched Full", gpu=False,
                                            sources=[f"{OWNER}/ctg-{key_}-prepare-full", f"{OWNER}/ctg-{stem}-baselines-full"],
                                            settings={"PREPARED_NAME": prepared})
# Seed variability of the matched LightGBM / XGBoost with relational features on CIC (revision): three further seed sets.
for rep_, seeds in (("r2", [45, 46, 47]), ("r3", [48, 49, 50]), ("r4", [51, 52, 53])):
    VARIANTS[f"cic-matched{rep_}-full"] = dict(VARIANTS["cic-matched-full"], title=f"CTG CIC Matched{rep_} Full",
                                               settings={**VARIANTS["cic-matched-full"]["settings"], "SEEDS_OVERRIDE": seeds})
# Strict graph-free ablation on 5G-NIDD (major revision): context-branch protocol with relational features absent,
# per segment, or on one merged timeline; smoothing fixed to the adoption-rule decision (BS1: pair key, BS2: none).
for stem, prepared, label, smooth_key in (("5gs-bs1", "nids-5gs-bs1", "5GS BS1", "pair"), ("5gs-bs2", "nids-5gs-bs2", "5GS BS2", None)):
    for mode in ("none", "seg", "merged"):
        VARIANTS[f"{stem}-strict{mode}-full"] = dict(dataset="5gs", kind="strict", title=f"CTG {label} Strict{mode} Full", gpu=False,
                                                     sources=[f"{OWNER}/ctg-5gs-prepare-full"],
                                                     settings={"PREPARED_NAME": prepared, "REL_MODE": mode, "SMOOTH_KEY": smooth_key})
# Held-out context-conditioned bound (major revision): same ranking as the root-cause kernel, lookups scored on calibration.
for stem in ("ton", "cic", "5gs-bs1", "5gs-bs2"):
    if f"{stem}-root-full" in VARIANTS:
        VARIANTS[f"{stem}-rootho-full"] = dict(VARIANTS[f"{stem}-root-full"], title=VARIANTS[f"{stem}-root-full"]["title"].replace("Root", "Rootho"),
                                               settings={**VARIANTS[f"{stem}-root-full"]["settings"], "SKIP_CONFIRM": True})
# Campaign-overlap audit and queueing under the recorded arrival traces (major revision).
QUEUE_MU = {"ton": [1630.8, 1041.0], "cic": [1519.4, 1225.6], "bs1": [1003.1, 702.7], "bs2": [1020.4, 754.8]}
for stem, key_, prepared, label, lk in (("ton", "ton", "nids-ton", "ToN", "ton"), ("cic", "cic", "nids-cic", "CIC", "cic"),
                                        ("5gs-bs1", "5gs", "nids-5gs-bs1", "5GS BS1", "bs1"), ("5gs-bs2", "5gs", "nids-5gs-bs2", "5GS BS2", "bs2")):
    VARIANTS[f"{stem}-campaign-full"] = dict(dataset=key_, kind="campaign", title=f"CTG {label} Campaign Full", gpu=False,
                                               sources=[f"{OWNER}/ctg-{key_}-prepare-full"], settings={"PREPARED_NAME": prepared})
    VARIANTS[f"{stem}-queue-full"] = dict(dataset=key_, kind="queue", title=f"CTG {label} Queue Full", gpu=False,
                                            sources=[f"{OWNER}/ctg-{key_}-prepare-full"],
                                            settings={"PREPARED_NAME": prepared, "MU_AMORTISED": QUEUE_MU[lk][0], "MU_BATCH": QUEUE_MU[lk][1]})
# Queueing of the 1 + 3 configuration (one graph seed, three context seeds; revision): rates from its measured latency
# (amortised 1e6 / us per flow; batch = micro-batch of 128 / median micro-batch delay), Table tab:single.
QUEUE_MU13 = {"bs1": [2899.2, 128 / 0.0624], "bs2": [2952.1, 128 / 0.0588]}
for stem, prepared, label, lk in (("5gs-bs1", "nids-5gs-bs1", "5GS BS1", "bs1"), ("5gs-bs2", "nids-5gs-bs2", "5GS BS2", "bs2")):
    VARIANTS[f"{stem}-queue13-full"] = dict(dataset="5gs", kind="queue", title=f"CTG {label} Queue13 Full", gpu=False,
                                              sources=["{}/ctg-5gs-prepare-full".format(OWNER)],
                                              settings={"PREPARED_NAME": prepared, "MU_AMORTISED": QUEUE_MU13[lk][0], "MU_BATCH": QUEUE_MU13[lk][1]})
# Split-vote audit (revision, R1.2): label-based block vote against a label-free chronological cut.
for stem, prepared, label in (("ton", "nids-ton", "ToN"), ("cic", "nids-cic", "CIC")):
    VARIANTS[f"{stem}-splitaudit-full"] = dict(dataset=stem, kind="splitaudit", title=f"CTG {label} Splitaudit Full", gpu=False,
                                               sources=[f"{OWNER}/ctg-{stem}-prepare-full"], settings={"PREPARED_NAME": prepared})
# Evaluation-only sensitivity re-runs of frozen graph checkpoints (major revision).
for stem, key_, prepared, label in (("5gs-bs1", "5gs", "nids-5gs-bs1", "5GS BS1"), ("5gs-bs2", "5gs", "nids-5gs-bs2", "5GS BS2")):
    for tag, margin in (("g5", 5.1), ("g30", 30.0)):
        for seed in (42, 43, 44):
            VARIANTS[f"{stem}-re{tag}s{seed}-full"] = dict(
                dataset=key_, kind="reeval", title=f"CTG {label} Re{tag}s{seed} Full", gpu=True,
                sources=[f"{OWNER}/ctg-{key_}-prepare-full", f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full"],
                settings={"PREPARED_NAME": prepared, "MARGIN": margin, "MERGE": False, "GPUS": 2})
# CPU versions (CPU sessions): a reference run at the manifest bound (MARGIN None) plus g' = 5.1 s and 30 s.
for stem, key_, prepared, label in (("5gs-bs1", "5gs", "nids-5gs-bs1", "5GS BS1"), ("5gs-bs2", "5gs", "nids-5gs-bs2", "5GS BS2")):
    for tag, margin in (("ref", None), ("g5", 5.1), ("g30", 30.0)):
        for seed in (42, 43, 44):
            VARIANTS[f"{stem}-cpu{tag}s{seed}-full"] = dict(
                dataset=key_, kind="reeval", title=f"CTG {label} Cpu{tag}s{seed} Full", gpu=False,
                sources=[f"{OWNER}/ctg-{key_}-prepare-full", f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full"],
                settings={"PREPARED_NAME": prepared, "MARGIN": margin, "MERGE": False, "GPUS": 1, "CPU": True})
# ToN and CIC CPU re-evaluations of the headline graph checkpoints (on the account that holds them, CTG_OWNER=runner):
# reference at the manifest bound, two other bounds, and one merged monitor timeline without segment resets.
for stem, key_, prepared, label, prep_src, train, seeds, bounds in (
        ("ton", "ton", "nids-ton", "ToN", "ctg-ton-prepare-full", "ctg-ton-v4bs{}-train-full", (43, 44), (("g30", 30.0), ("g121", 121.0))),
        ("cic", "cic", "nids-cic-rel", "CIC", "ctg-cic-relprep-full", "ctg-cic-v6r8s{}-train-full", (42, 43, 44), (("g60", 60.0), ("g300", 300.0)))):
    for tag, margin, merge in (("ref", None, False), ("merge", None, True)) + tuple((t, m, False) for t, m in bounds):
        for seed in seeds:
            VARIANTS[f"{stem}-cpu{tag}s{seed}-full"] = dict(
                dataset=key_, kind="reeval", title=f"CTG {label} Cpu{tag}s{seed} Full", gpu=False,
                sources=[f"{OWNER}/{prep_src}", f"{OWNER}/{train.format(seed)}"],
                settings={"PREPARED_NAME": prepared, "MARGIN": margin, "MERGE": merge, "GPUS": 1, "CPU": True})
# The same evasion applied to the strict graph-free LightGBM with relational features (fair comparison, revision).
for stem, prepared, label, smooth_key in (("5gs-bs1", "nids-5gs-bs1", "5GS BS1", "pair"), ("5gs-bs2", "nids-5gs-bs2", "5GS BS2", None)):
    for tag, ev in (("rot64", "rotate:64"), ("rotall", "rotate:1000000000"), ("thin10", "thin:10")):
        VARIANTS[f"{stem}-strict{tag}-full"] = dict(dataset="5gs", kind="strict", title=f"CTG {label} Strict{tag} Full", gpu=False,
                                                    sources=[f"{OWNER}/ctg-5gs-prepare-full"],
                                                    settings={"PREPARED_NAME": prepared, "REL_MODE": "seg", "SMOOTH_KEY": smooth_key,
                                                              "EVASION": ev})
# Seed repeats of the attacked strict LightGBM (revision): every attack, two further seed sets.
for stem, prepared, label, smooth_key in (("5gs-bs1", "nids-5gs-bs1", "5GS BS1", "pair"), ("5gs-bs2", "nids-5gs-bs2", "5GS BS2", None)):
    for tag, ev in (("rot64", "rotate:64"), ("rotall", "rotate:1000000000"), ("thin10", "thin:10")):
        for rep_, seeds in (("r2", [45, 46, 47]), ("r3", [48, 49, 50])):
            VARIANTS[f"{stem}-strict{tag}{rep_}-full"] = dict(
                dataset="5gs", kind="strict", title=f"CTG {label} Strict{tag}{rep_} Full", gpu=False,
                sources=[f"{OWNER}/ctg-5gs-prepare-full"],
                settings={"PREPARED_NAME": prepared, "REL_MODE": "seg", "SMOOTH_KEY": smooth_key, "EVASION": ev, "SEEDS_OVERRIDE": seeds})
# Context branch of BS1 (per-flow LightGBM with pair smoothing, = strict "none") recomputed under each evasion, saved in
# the context-branch format, and the fused model under evasion (revision). BS2 has no smoothing, so its context branch
# is unaffected by source rotation and identical on the records kept by thinning: the original context branch is used.
for tag, ev in (("rot64", "rotate:64"), ("rotall", "rotate:1000000000"), ("thin10", "thin:10")):
    VARIANTS[f"5gs-bs1-strictctx{tag}-full"] = dict(dataset="5gs", kind="strict", title=f"CTG 5GS BS1 Strictctx{tag} Full", gpu=False,
                                                    sources=[f"{OWNER}/ctg-5gs-prepare-full"],
                                                    settings={"PREPARED_NAME": "nids-5gs-bs1", "REL_MODE": "none", "SMOOTH_KEY": "pair",
                                                              "EVASION": ev, "SAVE_CONTEXT": True})
# Configuration variant (analysis, not the reported model): the fused model with the relational features in its context
# branch, as adopted on CIC, under the same evasions. Context = strict LightGBM with relational features (per segment).
for stem, prepared, label, smooth_key in (("5gs-bs1", "nids-5gs-bs1", "5GS BS1", "pair"), ("5gs-bs2", "nids-5gs-bs2", "5GS BS2", None)):
    for tag, ev in (("ref", None), ("rot64", "rotate:64"), ("rotall", "rotate:1000000000"), ("thin10", "thin:10")):
        VARIANTS[f"{stem}-strictrelctx{tag}-full"] = dict(
            dataset="5gs", kind="strict", title=f"CTG {label} Strictrelctx{tag} Full", gpu=False, sources=[f"{OWNER}/ctg-5gs-prepare-full"],
            settings={"PREPARED_NAME": prepared, "REL_MODE": "seg", "SMOOTH_KEY": smooth_key, "EVASION": ev, "SAVE_CONTEXT": True})
# Seed variability of the strict LightGBM with relational features (revision): three further seed sets, same protocol.
for stem, prepared, label, smooth_key in (("5gs-bs1", "nids-5gs-bs1", "5GS BS1", "pair"), ("5gs-bs2", "nids-5gs-bs2", "5GS BS2", None)):
    for rep_, seeds in (("r2", [45, 46, 47]), ("r3", [48, 49, 50]), ("r4", [51, 52, 53])):
        VARIANTS[f"{stem}-strictseg{rep_}-full"] = dict(
            dataset="5gs", kind="strict", title=f"CTG {label} Strictseg{rep_} Full", gpu=False, sources=[f"{OWNER}/ctg-5gs-prepare-full"],
            settings={"PREPARED_NAME": prepared, "REL_MODE": "seg", "SMOOTH_KEY": smooth_key, "SEEDS_OVERRIDE": seeds})
# Robustness of the frozen 5G-NIDD graph checkpoints (revision, CPU): adaptive evasion of the attack records in the test
# site (source rotation over 64 addresses or a fresh address per record; thinning to 1 in 10 attack records) and an
# inference-time ablation of the 7-day memory scale. Reference: the cpuref runs above.
ROBUST = {"rot64": {"EVASION": "rotate:64"}, "rotall": {"EVASION": "rotate:1000000000"}, "thin10": {"EVASION": "thin:10"},
          "nomem7d": {"DROP_SCALES": "604800"}, "nomem60": {"DROP_SCALES": "60"}, "nomem1h": {"DROP_SCALES": "3600"},
          "nomem1d": {"DROP_SCALES": "86400"}}
for stem, key_, prepared, label in (("5gs-bs1", "5gs", "nids-5gs-bs1", "5GS BS1"), ("5gs-bs2", "5gs", "nids-5gs-bs2", "5GS BS2")):
    for tag, extra in ROBUST.items():
        for seed in (42, 43, 44):
            VARIANTS[f"{stem}-cpu{tag}s{seed}-full"] = dict(
                dataset=key_, kind="reeval", title=f"CTG {label} Cpu{tag}s{seed} Full", gpu=False,
                sources=[f"{OWNER}/ctg-{key_}-prepare-full", f"{OWNER}/ctg-{stem}-v4bs{seed}-train-full"],
                settings={"PREPARED_NAME": prepared, "MARGIN": None, "MERGE": False, "GPUS": 1, "CPU": True, **extra})
# ToN seed 42 (checkpoint on a second account): paired reference and 7-day-memory ablation, same account and code.
VARIANTS["ton-cpurefs42-full"] = dict(
    dataset="ton", kind="reeval", title="CTG ToN Cpurefs42 Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-v4bs42-train-full"],
    settings={"PREPARED_NAME": "nids-ton", "MARGIN": None, "MERGE": False, "GPUS": 1, "CPU": True})
for seed in (42, 43, 44):
    VARIANTS[f"ton-cpunomem7ds{seed}-full"] = dict(
        dataset="ton", kind="reeval", title=f"CTG ToN Cpunomem7ds{seed} Full", gpu=False,
        sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-v4bs{seed}-train-full"],
        settings={"PREPARED_NAME": "nids-ton", "MARGIN": None, "MERGE": False, "GPUS": 1, "CPU": True, "DROP_SCALES": "604800"})
# Ordering-leakage audit of temporal flow graphs: start / end ordering vs causal admission (all splits; Computers & Security
# revision). Counts definite look-ahead edges (lo_j > hi_i) and non-certifiable edges (hi_j > lo_i).
for stem, key_, prepared, label in (("ton", "ton", "nids-ton", "ToN"), ("cic", "cic", "nids-cic", "CIC"),
                                    ("5gs-bs1", "5gs", "nids-5gs-bs1", "5GS BS1"), ("5gs-bs2", "5gs", "nids-5gs-bs2", "5GS BS2")):
    VARIANTS[f"{stem}-edgeaudit-full"] = dict(dataset=key_, kind="edgeaudit", title=f"CTG {label} Edgeaudit Full", gpu=False,
                                              sources=[f"{OWNER}/ctg-{key_}-prepare-full"], settings={"PREPARED_NAME": prepared})
# Statistical power / design analysis of the frozen HEADLINE comparisons (same inputs as the headline post-hocs + POWER).
for new, old, title in (("ton-power787-posthoc-full", "ton-final10787-posthoc-full", "CTG ToN Power787 Posthoc Full"),
                        ("cic-power-posthoc-full", "cic-final13-posthoc-full", "CTG CIC Power Posthoc Full"),
                        ("5gs-bs1-power-posthoc-full", "5gs-bs1-final10-posthoc-full", "CTG 5GS BS1 Power Posthoc Full"),
                        ("5gs-bs2-power-posthoc-full", "5gs-bs2-std-posthoc-full", "CTG 5GS BS2 Power Posthoc Full")):
    if old in VARIANTS:
        VARIANTS[new] = dict(VARIANTS[old], title=title, settings={**VARIANTS[old]["settings"], "POWER": True})
# ALL_METRICS re-runs (same inputs, full metric set for every model) of every setting reported in the README.
ALLM = [("ton-allm787", "ton-final10787", "ToN Allm787"), ("cic-allm", "cic-final13", "CIC Allm"),
        ("5gs-bs1-allm", "5gs-bs1-final10", "5GS BS1 Allm"), ("5gs-bs2-allm", "5gs-bs2-std", "5GS BS2 Allm"),
        ("ton-allmcf787", "ton-cftrainc1787", "ToN Allmcf787"), ("ton-allmcfstd787", "ton-cftrainc1std787", "ToN Allmcfstd787"),
        ("ton-allmcg787", "ton-cfgc1787", "ToN Allmcg787"), ("ton-allmcgstd787", "ton-cfgc1std787", "ToN Allmcgstd787"),
        ("5gs-bs2-allmcf", "5gs-bs2-cftrainc1", "5GS BS2 Allmcf"), ("5gs-bs2-allmcfstd", "5gs-bs2-cftrainc1std", "5GS BS2 Allmcfstd")]
ALLM += [(f"ton-allm{r}787", f"ton-clean{r}787", f"ToN Allm{r}787") for r in ("c1", "c2", "c1c2")]
ALLM += [(f"5gs-bs2-allm{r}", f"5gs-bs2-clean{r}", f"5GS BS2 Allm{r}") for r in ("c1", "c2", "c1c2")]
for new, old, title in ALLM:
    if f"{old}-posthoc-full" in VARIANTS:
        VARIANTS[f"{new}-posthoc-full"] = dict(VARIANTS[f"{old}-posthoc-full"], title=f"CTG {title} Posthoc Full",
                                                settings={**VARIANTS[f"{old}-posthoc-full"]["settings"], "ALL_METRICS": True})
# Single-seed ablation of the headline CT-Graph (manuscript review round 1): same inputs as the ALL_METRICS headline
# post-hocs plus SINGLE_SEED (each graph seed fused with the three-seed and with the matching one-seed context branch).
for new, old, title in (("ton-single787", "ton-allm787", "ToN Single787"), ("cic-single", "cic-allm", "CIC Single"),
                        ("5gs-bs1-single", "5gs-bs1-allm", "5GS BS1 Single"), ("5gs-bs2-single", "5gs-bs2-allm", "5GS BS2 Single")):
    if f"{old}-posthoc-full" in VARIANTS:
        VARIANTS[f"{new}-posthoc-full"] = dict(VARIANTS[f"{old}-posthoc-full"], title=f"CTG {title} Posthoc Full",
                                                settings={**VARIANTS[f"{old}-posthoc-full"]["settings"], "SINGLE_SEED": True})
# Major-revision analyses on the frozen headline predictions (graph-free paired tests, effect distributions, source-frozen
# low-FAR operating points, score distributions, class counts with Wilson intervals). Same inputs as the ALL_METRICS post-hocs.
for new, old, title in (("ton-rev787", "ton-allm787", "ToN Rev787"), ("cic-rev", "cic-allm", "CIC Rev"),
                        ("5gs-bs1-rev", "5gs-bs1-allm", "5GS BS1 Rev"), ("5gs-bs2-rev", "5gs-bs2-allm", "5GS BS2 Rev")):
    if f"{old}-posthoc-full" in VARIANTS:
        VARIANTS[f"{new}-posthoc-full"] = dict(VARIANTS[f"{old}-posthoc-full"], title=f"CTG {title} Posthoc Full",
                                                settings={**VARIANTS[f"{old}-posthoc-full"]["settings"], "REVISION": True})
# Second round: the same revision post-hoc with the matched graph-free baselines (GBDT + 59 relational features) as extra baselines.
for new, old, stem, title in (("ton-rev2787", "ton-rev787", "ton", "ToN Rev2787"), ("cic-rev2", "cic-rev", "cic", "CIC Rev2"),
                              ("5gs-bs1-rev2", "5gs-bs1-rev", "5gs-bs1", "5GS BS1 Rev2"), ("5gs-bs2-rev2", "5gs-bs2-rev", "5gs-bs2", "5GS BS2 Rev2")):
    if f"{old}-posthoc-full" in VARIANTS:
        VARIANTS[f"{new}-posthoc-full"] = dict(VARIANTS[f"{old}-posthoc-full"], title=f"CTG {title} Posthoc Full",
                                                sources=[*VARIANTS[f"{old}-posthoc-full"]["sources"], f"{OWNER}/ctg-{stem}-matched-full"])
# ToN fusion under the CPU re-evaluations (seeds 43 and 44, the two with checkpoints): same inputs as the ToN revision
# post-hoc except that the GPU graph runs (and the seed-42 GPU predictions) are replaced by the CPU re-runs.
if "ton-rev787-posthoc-full" in VARIANTS:
    _r = VARIANTS["ton-rev787-posthoc-full"]
    for tag in ("ref", "merge", "g30", "g121"):
        VARIANTS[f"ton-cpu{tag}-posthoc-full"] = dict(
            _r, title=f"CTG ToN Cpu{tag} Posthoc Full",
            sources=[f"{OWNER}/ctg-ton-cpu{tag}s{seed}-full" for seed in (43, 44)] + [x for x in _r["sources"] if "-train-full" not in x],
            datasets=[x for x in _r.get("datasets", []) if not x.endswith("-graph")])
for stem in ("5gs-bs1", "5gs-bs2"):
    rev_ = VARIANTS.get(f"{stem}-rev-posthoc-full")
    if rev_:
        for tag in ("rot64", "rotall", "thin10"):
            ctx = ([f"{OWNER}/ctg-5gs-bs1-strictctx{tag}-full"] if stem == "5gs-bs1" else
                   [x for x in rev_["sources"] if "ctgbdt" in x or "smooth" in x])
            VARIANTS[f"{stem}-cpu{tag}-posthoc-full"] = dict(
                rev_, title=f"CTG {stem.upper().replace('5GS-', '5GS ')} Cpu{tag} Posthoc Full",
                sources=[f"{OWNER}/ctg-{stem}-cpu{tag}s{seed}-full" for seed in (42, 43, 44)] + ctx
                        + [x for x in rev_["sources"] if "-train-full" not in x and "ctgbdt" not in x and "smooth" not in x])
# CIC fusion under the CPU re-evaluations: the graph runs were evaluated on runner (which holds the checkpoints) and
# uploaded to runner as private datasets; context branch and baselines as in the CIC revision post-hoc.
if "cic-rev-posthoc-full" in VARIANTS:
    _c = VARIANTS["cic-rev-posthoc-full"]
    for tag in ("ref", "merge", "g60", "g300"):
        VARIANTS[f"cic-cpu{tag}-posthoc-full"] = dict(
            _c, title=f"CTG CIC Cpu{tag} Posthoc Full",
            datasets=[x for x in _c.get("datasets", []) if not x.endswith("-graph")] + [f"{OWNER}/ctg-cic-cpu{tag}-graph"])
for stem in ("5gs-bs1", "5gs-bs2"):
    rev_ = VARIANTS.get(f"{stem}-rev-posthoc-full")
    if rev_:
        for tag in ("ref", "rot64", "rotall", "thin10"):
            VARIANTS[f"{stem}-cpurel{tag}-posthoc-full"] = dict(
                rev_, title=f"CTG {stem.upper().replace('5GS-', '5GS ')} Cpurel{tag} Posthoc Full",
                sources=[f"{OWNER}/ctg-{stem}-cpu{tag}s{seed}-full" for seed in (42, 43, 44)] + [f"{OWNER}/ctg-{stem}-strictrelctx{tag}-full"]
                        + [x for x in rev_["sources"] if "-train-full" not in x and "ctgbdt" not in x and "smooth" not in x])
# Delay-bound sensitivity of the fused model on 5G-NIDD: the revision post-hoc with the graph runs replaced by the CPU
# re-evaluations (reference at the manifest bound, g' = 5.1 s, g' = 30 s). The context branch does not depend on g.
for stem in ("5gs-bs1", "5gs-bs2"):
    rev_ = VARIANTS.get(f"{stem}-rev-posthoc-full")
    if rev_:
        for tag in ("ref", "g5", "g30"):
            VARIANTS[f"{stem}-cpu{tag}-posthoc-full"] = dict(
                rev_, title=f"CTG {stem.upper().replace('5GS-', '5GS ')} Cpu{tag} Posthoc Full",
                sources=[f"{OWNER}/ctg-{stem}-cpu{tag}s{seed}-full" for seed in (42, 43, 44)]
                        + [src for src in rev_["sources"] if "-train-full" not in src])
VARIANTS["ton-final12787-posthoc-full"] = dict(
    dataset="ton", kind="posthoc", title="CTG ToN Final12787 Posthoc Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-gsmooth787-full", f"{OWNER}/ctg-ton-smooth787-full"],
    datasets=[f"{OWNER}/ctg-ton-baseline-predictions"], settings={})
for site, ctx in (("bs1", "5gs-bs1-smooth"), ("bs2", "5gs-bs2-ctgbdt4")):
    VARIANTS[f"5gs-{site}-gsmooth-full"] = dict(
        dataset="5gs", kind="gsmooth", title=f"CTG 5GS {site.upper()} Gsmooth Full", gpu=False,
        sources=[f"{OWNER}/ctg-5gs-prepare-full", *[f"{OWNER}/ctg-5gs-{site}-v4bs{seed}-train-full" for seed in (42, 43, 44)],
                 f"{OWNER}/ctg-{ctx}-full"],
        settings={"PREPARED_NAME": f"nids-5gs-{site}", "KEY": "pair"})
    VARIANTS[f"5gs-{site}-final12-posthoc-full"] = dict(
        dataset="5gs", kind="posthoc", title=f"CTG 5GS {site.upper()} Final12 Posthoc Full", gpu=False,
        sources=[f"{OWNER}/ctg-5gs-{site}-gsmooth-full", f"{OWNER}/ctg-{ctx}-full", f"{OWNER}/ctg-5gs-{site}-baselines-full",
                 f"{OWNER}/ctg-5gs-{site}-q1b-full"], settings={})
# Formal degradation analysis (validation only) on the v4b graph seeds + the context-GBDT branch of each account.
VARIANTS["ton-formal-full"] = dict(
    dataset="ton", kind="formal", title="CTG ToN Formal Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-prepare-full", f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-ctgbdt4-predictions"], settings={"PREPARED_NAME": "nids-ton"})
VARIANTS["cic-formal-full"] = dict(
    dataset="cic", kind="formal", title="CTG CIC Formal Full", gpu=False,
    sources=[f"{OWNER}/ctg-cic-prepare-full", f"{OWNER}/ctg-cic-v4bs42-train-full", f"{OWNER}/ctg-cic-v4bs43-train-full",
             f"{OWNER}/ctg-cic-ctgbdt-full"], settings={"PREPARED_NAME": "nids-cic"})
for site, seeds in (("bs1", (42, 43, 44)), ("bs2", (42, 43))):
    VARIANTS[f"5gs-{site}-formal-full"] = dict(
        dataset="5gs", kind="formal", title=f"CTG 5GS {site.upper()} Formal Full", gpu=False,
        sources=[f"{OWNER}/ctg-5gs-prepare-full", *[f"{OWNER}/ctg-5gs-{site}-v4bs{sd}-train-full" for sd in seeds],
                 f"{OWNER}/ctg-5gs-{site}-ctgbdt-full"], settings={"PREPARED_NAME": f"nids-5gs-{site}"})
for site in ("bs1", "bs2"):
    VARIANTS[f"5gs-{site}-loco-full"] = dict(dataset="5gs", kind="loco", title=f"CTG 5GS {site.upper()} Loco Full", gpu=False,
                                             sources=[f"{OWNER}/ctg-5gs-prepare-full"], settings={"PREPARED_NAME": f"nids-5gs-{site}"})
VARIANTS["ton-gapdiag787-full"] = dict(
    dataset="ton", kind="gapdiag", title="CTG ToN Gapdiag787 Full", gpu=False,
    sources=[f"{OWNER}/ctg-ton-v4bs43-train-full", f"{OWNER}/ctg-ton-v4bs44-train-full"],
    datasets=[f"{OWNER}/ctg-ton-v4bs42-graph", f"{OWNER}/ctg-ton-ctgbdt4-predictions"], settings={})
# Backwards-compatible names for the ToN kernels already published.
for stage in ("prepare-full", "baselines-full", "train-full", "posthoc-full"):
    VARIANTS[stage] = VARIANTS[f"ton-{stage}"]


# Target-site calibration sample-size curve on BS2->BS1 (revision): benign-sample fractions and FAR targets, 20 random draws.
if "5gs-bs2-rev2-posthoc-full" in VARIANTS:
    _b = VARIANTS["5gs-bs2-rev2-posthoc-full"]
    VARIANTS["5gs-bs2-rev2cal-posthoc-full"] = dict(_b, title=_b["title"].replace(" Posthoc", "cal Posthoc"),
        settings={**_b.get("settings", {}), "RECAL_FRACS": [0.001, 0.005, 0.01, 0.02, 0.05], "RECAL_FARS": [0.0001, 0.001, 0.005, 0.01]})
# Rule diagnostic (revision): equal-weight fusion scored under both decision rules, for the reported BS2 run and its CPU re-run.
for base in ("5gs-bs2-cpuref-posthoc-full", "5gs-bs2-rev2-posthoc-full"):
    if base in VARIANTS:
        _b = VARIANTS[base]
        VARIANTS[base.replace("-posthoc-full", "diag-posthoc-full")] = dict(
            _b, title=_b["title"].replace(" Posthoc", "diag Posthoc"), settings={**_b.get("settings", {}), "RULE_DIAG": True})


def slugify(title):
    return "-".join(title.lower().split())


def build(name):
    spec = VARIANTS[name]
    code = PIPELINE.read_bytes()
    payload = base64.b64encode(zlib.compress(code, 9)).decode()
    ds = DATASETS[spec["dataset"]]
    settings = {"DATASET_NAME": ds["name"], "CSV_NAME": ds["csv"], "FAIRDATA": ds["fairdata"],
                "SITE_LAYOUT": ds["site_layout"], "CONTRACT": ds.get("contract"), **spec["settings"]}
    settings = "".join(f"{k} = {v!r}\n" for k, v in settings.items())
    body = {"prepare": PREPARE, "train": TRAIN, "reeval": REEVAL, "context": CONTEXT,
            "context_gbdt": (HERE / "context_gbdt_body.py").read_text(),
            "baselines": (HERE / "baselines_body.py").read_text(), "models": MODELS,
            "posthoc": (HERE / "posthoc_body.py").read_text(), "study": (HERE / "study_body.py").read_text(),
            "hypothesis": (HERE / "hypothesis_body.py").read_text(),
            "unittest": (HERE / "unittest_body.py").read_text(), "deep": (HERE / "deep_body.py").read_text(),
            "formal": (HERE / "formal_body.py").read_text(), "loco": (HERE / "loco_body.py").read_text(),
            "rootcause": (HERE / "rootcause_body.py").read_text(),
            "gapdiag": (HERE / "gapdiag_body.py").read_text(),
            "features": (HERE / "features_body.py").read_text(),
            "ctgbdt5": (HERE / "ctgbdt5_body.py").read_text(),
            "temporal": (HERE / "temporal_lib.py").read_text() + (HERE / "temporal_body.py").read_text(),
            "ctgbdt6": (HERE / "temporal_lib.py").read_text() + (HERE / "ctgbdt6_body.py").read_text(),
            "collective": (HERE / "temporal_lib.py").read_text() + (HERE / "collective_body.py").read_text(),
            "collectivem": (HERE / "temporal_lib.py").read_text() + (HERE / "collective_body.py").read_text(),
            "ctgbdt7": (HERE / "temporal_lib.py").read_text() + (HERE / "ctgbdt7_body.py").read_text(),
            "screen": (HERE / "temporal_lib.py").read_text() + (HERE / "screen_body.py").read_text(),
            "ctgbdt8": (HERE / "temporal_lib.py").read_text() + (HERE / "ctgbdt8_body.py").read_text(),
            "robust": (HERE / "temporal_lib.py").read_text() + (HERE / "robust_body.py").read_text(),
            "mvdiag": (HERE / "temporal_lib.py").read_text() + (HERE / "mvdiag_body.py").read_text(),
            "robust2": (HERE / "temporal_lib.py").read_text() + (HERE / "robust2_body.py").read_text(),
            "lab2": (HERE / "temporal_lib.py").read_text() + (HERE / "lab2_body.py").read_text(),
            "lab3": (HERE / "temporal_lib.py").read_text() + (HERE / "lab3_body.py").read_text(),
            "ctgbdt9": (HERE / "temporal_lib.py").read_text() + (HERE / "ctgbdt9_body.py").read_text(),
            "lab4": (HERE / "temporal_lib.py").read_text() + (HERE / "lab4_body.py").read_text(),
            "smooth": (HERE / "temporal_lib.py").read_text() + (HERE / "smooth_body.py").read_text(),
            "lab5": (HERE / "temporal_lib.py").read_text() + (HERE / "lab5_body.py").read_text(),
            "relprep": (HERE / "temporal_lib.py").read_text() + (HERE / "relprep_body.py").read_text(),
            "decide": (HERE / "decide_body.py").read_text(),
            "gsmooth": (HERE / "temporal_lib.py").read_text() + (HERE / "gsmooth_body.py").read_text(),
            "tonerr": (HERE / "temporal_lib.py").read_text() + (HERE / "tonerr_body.py").read_text(),
            "scanlab": (HERE / "temporal_lib.py").read_text() + (HERE / "scanlab_body.py").read_text(),
            "cleanids": (HERE / "temporal_lib.py").read_text() + (HERE / "cleanids_body.py").read_text(),
            "cleanprep": (HERE / "temporal_lib.py").read_text() + (HERE / "cleanprep_body.py").read_text(),
            "bs2err": (HERE / "temporal_lib.py").read_text() + (HERE / "bs2err_body.py").read_text(),
            "latency": (HERE / "temporal_lib.py").read_text() + (HERE / "latency_body.py").read_text(),
            "relsys": (HERE / "temporal_lib.py").read_text() + (HERE / "relsys_body.py").read_text(),
            "edgeaudit": (HERE / "temporal_lib.py").read_text() + (HERE / "edgeaudit_body.py").read_text(),
            "matched": (HERE / "temporal_lib.py").read_text() + (HERE / "matched_body.py").read_text(),
            "strict": (HERE / "temporal_lib.py").read_text() + (HERE / "strict_body.py").read_text(),
            "campaign": (HERE / "temporal_lib.py").read_text() + (HERE / "campaign_body.py").read_text(),
            "queue": (HERE / "temporal_lib.py").read_text() + (HERE / "queue_body.py").read_text(),
            "splitaudit": (HERE / "temporal_lib.py").read_text() + (HERE / "splitaudit_body.py").read_text()}[spec["kind"]]
    script = (f'"""{spec["title"]}: generated by experiments/build_runs.py from run_pipeline.py '
              f'sha256={hashlib.sha256(code).hexdigest()}."""\n'
              f"PAYLOAD = {payload!r}\n{settings}{COMMON}{body}")
    slug = "ctg-" + (name if name.split("-")[0] in DATASETS or name.split("-")[0][:2] in DATASETS else "ton-" + name)
    if slugify(spec["title"]) != slug:
        raise ValueError(f"run title {spec['title']!r} does not resolve to slug {slug!r}")
    folder = HERE / slug if OWNER == "runner" else HERE / "accounts" / OWNER / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{slug}.py").write_text(script)
    metadata = {"id": f"{OWNER}/{slug}", "title": spec["title"], "code_file": f"{slug}.py", "language": "python",
                "run_type": "script", "private": True, "gpu": spec["gpu"], "tpu": False,
                "internet": True,
                "data_inputs": ([ds["source"]] if spec["kind"] == "prepare" and not ds["fairdata"] else []) + spec.get("datasets", []),
                "other_inputs": [], "inputs": spec["sources"], "model_inputs": []}
    if spec["gpu"]:
        metadata["machine_shape"] = "NvidiaTeslaT4"
    (folder / "run.json").write_text(json.dumps(metadata, indent=2))
    print(folder, len(script))


if __name__ == "__main__":
    # Optional "--seeds 42,43" restricts v2 post-hoc graph sources to seeds that actually ran.
    args = sys.argv[1:]
    if "--seeds" in args:
        i = args.index("--seeds")
        seeds = [int(v) for v in args[i + 1].split(",")]
        del args[i:i + 2]
        for name in args:
            spec = VARIANTS[name]
            if spec["kind"] == "posthoc":
                spec["sources"] = [src for src in spec["sources"]
                                   if "-v2s" not in src or any(src.endswith(f"-v2s{seed}-train-full") for seed in seeds)]
    for name in args or VARIANTS:
        build(name)

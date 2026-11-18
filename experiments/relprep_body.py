
# RELATIONAL-AUGMENTED PREPARED DATA for the graph branch (v6r). Every shard of every split is rewritten with the causal
# relational context (the root-cause / CT-GBDT v5 features: per source / destination / pair windows, label-free, each
# flow using only earlier flows of its own segment) appended to its `numeric` vector. Rows, order and every other column
# are unchanged. The names are appended to schema.derived (feature_names = numeric + derived stays aligned); means,
# scales and observed fractions of the new columns are fitted on the TRAINING split only, exactly as in preparation.
# manifest.preprocess_sha256 and the shard digests are updated; donors.pt is not copied (with robust features on, donors
# are re-featurised from training shards).
import hashlib
import shutil

started = time.time()
MERGED = globals().get("MERGED", False)  # relational context on ONE merged monitor timeline (boundary-robust variant)
OUT_NAME = PREPARED_NAME + ("-relm" if MERGED else "-rel")
OUT_ROOT = f"/data/output/{OUT_NAME}"
os.makedirs(OUT_ROOT, exist_ok=True)
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


new_manifest = json.loads(json.dumps(manifest))
stats = None
for split in ("train", "val", "calibration", "test"):
    f = load(split)
    if MERGED:
        f["hi"], f["stream"], how = merged_clock(f)
        log("merged timeline:", how)
    R = relational(f).to_numpy(np.float32)
    n_all = len(f)
    del f
    if split == "train":
        finite = np.isfinite(R)
        seen = finite.mean(0)
        vals = np.where(finite, R, 0).astype(np.float64)
        cnt = np.maximum(finite.sum(0), 1)
        mean = vals.sum(0) / cnt
        var = np.maximum((vals * vals).sum(0) / cnt - mean * mean, 0)
        stats = {"means": mean, "scale": np.where(var > 1e-12, np.sqrt(var), 1.0), "seen": seen}
    os.makedirs(os.path.join(OUT_ROOT, "prepared", split), exist_ok=True)
    offset = 0
    for i, shard in enumerate(manifest["shards"][split]):
        table = pq.read_table(shard_path(split, shard))
        n = len(table)
        numeric = table["numeric"].combine_chunks()
        offs = numeric.offsets.to_numpy()
        old = numeric.values.slice(int(offs[0]), int(offs[-1] - offs[0])).to_numpy(zero_copy_only=False).reshape(n, width)
        new = np.concatenate([old.astype(np.float32), R[offset:offset + n]], 1)
        w2 = new.shape[1]
        arr = pa.ListArray.from_arrays(pa.array(np.arange(0, n * w2 + 1, w2, dtype=np.int32)), pa.array(new.ravel(), pa.float32()))
        table = table.set_column(table.schema.get_field_index("numeric"), pa.field("numeric", pa.list_(pa.float32())), arr)
        out_path = os.path.join(OUT_ROOT, "prepared", split, os.path.basename(shard["path"]))
        # Keep the ORIGINAL row-group boundaries: the training loader reads and caches whole row groups, so default
        # (~1M-row) groups made v6r ~9x slower per epoch than v4b.
        src_meta = pq.ParquetFile(shard_path(split, shard)).metadata
        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        start = 0
        for g in range(src_meta.num_row_groups):
            rows_g = src_meta.row_group(g).num_rows
            writer.write_table(table.slice(start, rows_g), row_group_size=max(rows_g, 1))
            start += rows_g
        writer.close()
        assert start == n, (split, i, start, n)
        entry = new_manifest["shards"][split][i]
        entry["path"] = out_path
        if "sha256" in entry:
            entry["sha256"] = sha256(out_path)
        offset += n
        del table, numeric, old, new, arr
    assert offset == n_all, (split, offset, n_all)
    log("rewrote", split, n_all, "rows,", len(manifest["shards"][split]), "shards", round(time.time() - started), "s")
    del R

REL = list(relational_names)
new_prep = json.loads(json.dumps(prep))
new_prep["means"] = list(prep["means"]) + stats["means"].tolist()
new_prep["scale"] = list(prep["scale"]) + stats["scale"].tolist()
new_prep["observed_fraction"] = list(prep["observed_fraction"]) + stats["seen"].tolist()
new_prep["feature_names"] = list(prep["feature_names"]) + REL
new_prep["schema"]["derived"] = list(prep["schema"].get("derived", [])) + REL
new_prep["relational_augmentation"] = {"features": REL, "source_prepared": PREPARED_NAME, "merged_timeline": MERGED,
                                       "definition": "causal per-segment windows (root-cause library); train-fitted scaling"}
assert len(new_prep["means"]) == len(new_prep["feature_names"]) == width + len(REL)
assert new_prep["feature_names"] == new_prep["schema"]["numeric"] + new_prep["schema"]["derived"]
with open(os.path.join(OUT_ROOT, "preprocess.json"), "w") as fh:
    json.dump(new_prep, fh, indent=1)
new_manifest["preprocess_sha256"] = sha256(os.path.join(OUT_ROOT, "preprocess.json"))
new_manifest["relational_augmentation"] = {"features": len(REL), "source_prepared": PREPARED_NAME}
with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as fh:
    json.dump(new_manifest, fh, indent=1)
for name in ("split_audit.json", "exporter_timeouts.frozen.json"):
    if os.path.exists(os.path.join(root, name)):
        shutil.copy2(os.path.join(root, name), OUT_ROOT)
log("relational-augmented prepared data written to", OUT_ROOT, "| features", width, "->", width + len(REL),
    "| elapsed", round(time.time() - started), "s")


# CLEANED-LABEL PREPARED DATA (label-collision ablation, train side). The rules are the pre-registered cleaned-benchmark
# rules of cleanids_body.py, applied to the FITTING / SELECTION splits (train, validation, calibration):
#   C1  a flow whose per-flow input vector occurs anywhere in the dataset with the Benign label AND with an attack label
#   C2  a Benign-labelled flow whose source host emits attack-labelled flows in the same split
#   C1C2 the union
# Only LABELS change: an affected flow keeps its row, features, order and stream, and its label is set to null, so it
# stays traffic (graph / relational / smoothing context) but is never fitted on and never scored (every loader keeps
# label >= 0 rows; the graph loss masks y < 0). The TEST split is copied unchanged: models trained here still predict the
# full standard test set, and the cleaned test set is applied in the post-hoc (CLEAN_RULE) from the cleanids drop lists.
# preprocess.class_counts (the logit-adjustment prior) is recomputed from the cleaned training labels; feature scaling is
# label-free and unchanged. manifest shard digests / label counts and preprocess_sha256 are updated.
import hashlib
import shutil

RULE = CLEAN.upper()
# FIT_ONLY (leakage-free, the default for any model comparison): C1 collisions are computed from the labels of the
# fitting / selection splits only (train, validation, calibration). Without it the collision set also uses TEST labels
# (benchmark-curation definition): on 5G BS2 that nulls 100 % of the udpflood training labels (every one is identical to
# some BS1 test Benign flow), i.e. test labels would decide which training labels survive.
FIT_ONLY = globals().get("FIT_ONLY", True)
COLLISION_SPLITS = ("train", "val", "calibration") if FIT_ONLY else ("train", "val", "calibration", "test")
OUT_NAME = f"{PREPARED_NAME}-{'f' if FIT_ONLY else ''}{RULE.lower()}"
OUT_ROOT = f"/data/output/{OUT_NAME}"
os.makedirs(OUT_ROOT, exist_ok=True)
started = time.time()
BEN = classes.index("Benign")
SPLITS = ("train", "val", "calibration", "test")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shard_arrays(split, shard):
    # identical hashing to cleanids_body.py (numeric vector + category codes, NaN -> sentinel)
    t = pq.read_table(shard_path(split, shard), columns=["numeric", "categories", "label", "s0", "s1"])
    x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
    cats = []
    for j, vocab in enumerate(prep["vocabs"]):
        values = pc.list_element(t["categories"], j)
        index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
        cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
    X = np.concatenate([x, np.column_stack(cats)], 1) if cats else x
    h = pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(X, nan=-7.77e30)), index=False).to_numpy()
    y = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
    return h, y, key(t["s0"].to_numpy(), t["s1"].to_numpy())


# pass 1: hashes / labels / sources of every split
A = {}
for split in SPLITS:
    parts = [shard_arrays(split, s) for s in manifest["shards"][split]]
    A[split] = {"h": np.concatenate([p[0] for p in parts]), "y": np.concatenate([p[1] for p in parts]),
                "src": np.concatenate([p[2] for p in parts]), "sizes": [len(p[1]) for p in parts]}
    del parts
    log("hashed", split, len(A[split]["y"]), round(time.time() - started), "s")
H = np.concatenate([A[s]["h"][A[s]["y"] >= 0] for s in COLLISION_SPLITS])
Y = np.concatenate([A[s]["y"][A[s]["y"] >= 0] for s in COLLISION_SPLITS])
g = pd.DataFrame({"h": H, "benign": Y == BEN, "attack": Y != BEN}).groupby("h").agg(benign=("benign", "any"), attack=("attack", "any"))
colliding = g.index.to_numpy()[(g.benign & g.attack).to_numpy()]
del H, Y, g
stats = {"rule": RULE, "fit_only": FIT_ONLY, "collision_splits": list(COLLISION_SPLITS), "colliding_vectors": int(len(colliding)), "splits": {}}
drops = {}
for split in SPLITS:
    y, lab = A[split]["y"], A[split]["y"] >= 0
    c1 = lab & np.isin(A[split]["h"], colliding)
    c2 = (y == BEN) & np.isin(A[split]["src"], np.unique(A[split]["src"][lab & (y != BEN)]))
    drop = {"C1": c1, "C2": c2, "C1C2": c1 | c2}[RULE]
    drops[split] = drop
    stats["splits"][split] = {"labelled": int(lab.sum()), "nulled": int(drop.sum()),
                              "by_class": {classes[k]: [int((drop & (y == k)).sum()), int((y == k).sum())] for k in range(K) if (y == k).any()}}
    log(split, "rule", RULE, "would null", int(drop.sum()), "of", int(lab.sum()), "labelled rows")
# the test split is NOT modified; its would-be count must equal the cleanids drop list for this rule
stats["test_split_unchanged"] = True

# pass 2: rewrite train / validation / calibration with nulled labels (original row groups preserved); copy test
new_manifest = json.loads(json.dumps(manifest))
for split in SPLITS:
    os.makedirs(os.path.join(OUT_ROOT, "prepared", split), exist_ok=True)
    offset = 0
    for i, shard in enumerate(manifest["shards"][split]):
        src_path = shard_path(split, shard)
        out_path = os.path.join(OUT_ROOT, "prepared", split, os.path.basename(shard["path"]))
        n = A[split]["sizes"][i]
        entry = new_manifest["shards"][split][i]
        entry["path"] = out_path
        if split == "test":
            shutil.copy2(src_path, out_path)
        else:
            table = pq.read_table(src_path)
            assert len(table) == n, (split, i)
            d = drops[split][offset:offset + n]
            col = table["label"].combine_chunks()
            labels = pc.if_else(pa.array(d), pa.scalar(None, type=col.type), col)
            table = table.set_column(table.schema.get_field_index("label"), table.schema.field("label"), labels)
            meta_ = pq.ParquetFile(src_path).metadata
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            start = 0
            for gi in range(meta_.num_row_groups):
                rows_g = meta_.row_group(gi).num_rows
                writer.write_table(table.slice(start, rows_g), row_group_size=max(rows_g, 1))
                start += rows_g
            writer.close()
            assert start == n, (split, i, start, n)
            if "labels" in entry:
                vals = pd.Series(labels.to_pylist()).fillna("<unlabelled>").value_counts()
                entry["labels"] = {str(k): int(v) for k, v in vals.items()}
            del table, col, labels
        if "sha256" in entry:
            entry["sha256"] = sha256(out_path)
        offset += n
    log("wrote", split, offset, "rows", round(time.time() - started), "s")

new_prep = json.loads(json.dumps(prep))
ytr = A["train"]["y"][~drops["train"]]
new_prep["class_counts"] = [int((ytr == k).sum()) for k in range(K)]
new_prep["label_cleaning"] = {"rule": RULE, "fit_only": FIT_ONLY, "source_prepared": PREPARED_NAME, "original_class_counts": prep["class_counts"],
                              "splits_cleaned": ["train", "val", "calibration"]}
with open(os.path.join(OUT_ROOT, "preprocess.json"), "w") as fh:
    json.dump(new_prep, fh, indent=1)
new_manifest["preprocess_sha256"] = sha256(os.path.join(OUT_ROOT, "preprocess.json"))
new_manifest["label_cleaning"] = stats
with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as fh:
    json.dump(new_manifest, fh, indent=1)
for name in ("split_audit.json", "exporter_timeouts.frozen.json"):
    if os.path.exists(os.path.join(root, name)):
        shutil.copy2(os.path.join(root, name), OUT_ROOT)
json.dump(stats, open(os.path.join(OUT_ROOT, "label_cleaning.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
log("cleaned-label prepared data written to", OUT_ROOT, "| class_counts", prep["class_counts"], "->", new_prep["class_counts"])

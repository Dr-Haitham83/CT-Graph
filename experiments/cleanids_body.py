
# CLEANED-BENCHMARK DROP LISTS (label-collision ablation; the standard benchmark is always reported first).
# Rules fixed before any cleaned result is computed:
#   C1  exact-feature deduplication: drop every labelled TEST flow whose per-flow input vector (the flow features every
#       model receives) occurs anywhere in the dataset (train / validation / calibration / test) with the Benign label
#       AND with an attack label (an attack-vs-benign label collision)
#   C2  disentangled mixed-label hosts: drop Benign-labelled TEST flows whose source host also emits attack-labelled
#       flows in the test split (their Benign labels are unreliable while the host is attacking)
#   C1C2 the union of both
# Output: clean_test_drop.npz (source ids per rule) + clean_stats.json. Shards are streamed to bound memory.
OUT = "/data/output/cleanids"
os.makedirs(OUT, exist_ok=True)
started = time.time()
BEN = classes.index("Benign")


def shard_frames(split, with_meta=False):
    for shard in manifest["shards"][split]:
        cols = ["numeric", "categories", "label"] + (["source_id", "s0", "s1"] if with_meta else [])
        t = pq.read_table(shard_path(split, shard), columns=cols)
        x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
        cats = []
        for j, vocab in enumerate(prep["vocabs"]):
            values = pc.list_element(t["categories"], j)
            index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
            cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
        X = np.concatenate([x, np.column_stack(cats)], 1) if cats else x
        h = pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(X, nan=-7.77e30)), index=False).to_numpy()
        y = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        meta = {"ids": np.asarray(t["source_id"].to_pylist(), dtype=object).astype(str),
                "src": key(t["s0"].to_numpy(), t["s1"].to_numpy())} if with_meta else None
        yield h, y, meta


H, Y = [], []
test_h, test_y, test_ids, test_src = [], [], [], []
for split in ("train", "val", "calibration", "test"):
    for h, y, meta in shard_frames(split, with_meta=(split == "test")):
        lab = y >= 0
        H.append(h[lab])
        Y.append(y[lab])
        if split == "test":
            test_h.append(h[lab]); test_y.append(y[lab]); test_ids.append(meta["ids"][lab]); test_src.append(meta["src"][lab])
    log("hashed", split, round(time.time() - started), "s")
H, Y = np.concatenate(H), np.concatenate(Y)
g = pd.DataFrame({"h": H, "benign": Y == BEN, "attack": Y != BEN}).groupby("h").agg(benign=("benign", "any"), attack=("attack", "any"))
colliding = g.index.to_numpy()[(g.benign & g.attack).to_numpy()]
del H, Y, g
test_h, test_y = np.concatenate(test_h), np.concatenate(test_y)
test_ids, test_src = np.concatenate(test_ids), np.concatenate(test_src)
c1 = np.isin(test_h, colliding)
attack_hosts = np.unique(test_src[test_y != BEN])
c2 = (test_y == BEN) & np.isin(test_src, attack_hosts)
rules = {"C1": c1, "C2": c2, "C1C2": c1 | c2}
np.savez_compressed(os.path.join(OUT, "clean_test_drop.npz"), **{r: test_ids[m] for r, m in rules.items()})
stats = {"colliding_vectors": int(len(colliding)), "test_rows": int(len(test_y)), "rules": {}}
for r, m in rules.items():
    stats["rules"][r] = {"dropped": int(m.sum()), "dropped_fraction": float(m.mean()),
                         "by_class": {classes[k]: {"dropped": int((m & (test_y == k)).sum()), "of": int((test_y == k).sum()),
                                                   "fraction": round(float((m & (test_y == k)).sum() / max((test_y == k).sum(), 1)), 4)}
                                      for k in range(K) if (test_y == k).any()}}
json.dump(stats, open(os.path.join(OUT, "clean_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
log("clean drop lists written")

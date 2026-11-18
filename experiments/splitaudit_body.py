# flake8: noqa -- executed after the COMMON prelude and temporal_lib.py (manifest, shard_path).
# SPLIT-VOTE AUDIT (revision, R1.2): how far the label-based campaign-forward block vote moves the split away from a
# label-free chronological cut at the same record fractions, and how many test attack families would have no training
# records under that label-free cut. Reads only availability times and labels of the prepared splits; no model.
OUT = "/data/output/splitaudit"
os.makedirs(OUT, exist_ok=True)
frames = []
for split in ("train", "validation", "calibration", "test"):
    for shard in manifest["shards"].get(split, []):
        t = pq.read_table(shard_path(split, shard), columns=["hi", "label"])
        frames.append(pd.DataFrame({"hi": t["hi"].to_numpy(zero_copy_only=False),
                                    "label": t["label"].to_pandas().fillna("Benign").astype(str), "split": split}))
f = pd.concat(frames, ignore_index=True).sort_values("hi", kind="stable").reset_index(drop=True)
n = len(f)
fr = np.array([0.60, 0.15, 0.10, 0.15]); edges = np.floor(np.cumsum(fr)[:-1] * n).astype(int)
names = np.array(["train", "validation", "calibration", "test"])
f["global"] = names[np.searchsorted(edges, np.arange(n), side="right")]
att = f["label"].str.lower() != "benign"
same = f["split"] == f["global"]
rep = {"prepared": PREPARED_NAME, "records": int(n), "attack_records": int(att.sum()),
       "share_same_split": float(same.mean()), "share_same_split_attack": float(same[att].mean()),
       "share_same_split_benign": float(same[~att].mean()),
       "test_records_also_test_under_global": float((f.loc[f.split == "test", "global"] == "test").mean())}
fam = {}
for lab, g in f[att].groupby("label"):
    fam[lab] = {"records": int(len(g)),
                "campaign_forward": {s: int((g.split == s).sum()) for s in names},
                "global_cut": {s: int((g["global"] == s).sum()) for s in names}}
test_fams = [k for k, v in fam.items() if v["campaign_forward"]["test"] > 0]
rep["test_families"] = len(test_fams)
rep["test_families_without_training_under_global"] = sorted(k for k in test_fams if fam[k]["global_cut"]["train"] == 0)
rep["test_families_absent_from_test_under_global"] = sorted(k for k in test_fams if fam[k]["global_cut"]["test"] == 0)
rep["families"] = fam
log(json.dumps({k: v for k, v in rep.items() if k != "families"}))
json.dump(rep, open(os.path.join(OUT, "splitaudit.json"), "w"), indent=1, default=float)

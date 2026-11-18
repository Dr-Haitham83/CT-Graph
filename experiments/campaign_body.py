# flake8: noqa -- executed after the COMMON prelude and temporal_lib.py (manifest, classes, shard_path, label_set).
# CAMPAIGN OVERLAP AUDIT (major revision). Labels and availability times only; no model, no feature.
# For every attack class: (1) forward order per family: share of test records that precede the last training record of
# the same class; (2) activity episodes = maximal runs of records of the class separated by gaps > G (G = 600 s, 3600 s)
# on the absolute clock of the release (capture-relative releases: within a capture); an episode is "shared" if it has
# records in more than one split, "test-only" if all its records are in the test split; (3) per split, the number of
# segments (graph-state episodes) holding the class. Purged boundary records are not in any split and are ignored.
OUT = "/data/output/campaign"
os.makedirs(OUT, exist_ok=True)
GAPS = (600.0, 3600.0)
rows = []
for split in ("train", "val", "calibration", "test"):
    for shard in manifest["shards"].get(split, []):
        t = pq.read_table(shard_path(split, shard), columns=["hi", "label", "stream"])
        y = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        rows.append(pd.DataFrame({"t": t["hi"].to_numpy(zero_copy_only=False), "y": y, "stream": t["stream"].to_numpy(zero_copy_only=False),
                                  "split": split}))
f = pd.concat(rows, ignore_index=True)
f = f[f["y"] >= 0]
relative = manifest.get("time_basis") == "capture_relative"
# capture-relative clocks restart per capture: episodes and order are evaluated within the capture (stream prefix)
f["capture"] = f["stream"].str.split("/segment-").str[0] if relative else "all"
report = {"prepared": PREPARED_NAME, "time_basis": manifest.get("time_basis"), "classes": {}}
benign = [c.lower() for c in classes].index("benign")
for k, c in enumerate(classes):
    g = f[f["y"] == k].sort_values(["capture", "t"])
    if not len(g):
        continue
    entry = {"records": {s: int((g["split"] == s).sum()) for s in ("train", "val", "calibration", "test")},
             "segments": {s: int(g.loc[g["split"] == s, "stream"].nunique()) for s in ("train", "val", "calibration", "test")}}
    tr_last = g[g["split"] == "train"].groupby("capture")["t"].max()
    te = g[g["split"] == "test"]
    if len(te) and len(tr_last):
        before = te["t"].to_numpy() < te["capture"].map(tr_last).fillna(-np.inf).to_numpy()
        entry["test_before_last_train_share"] = float(before.mean())
    for gap in GAPS:
        new = (g["capture"] != g["capture"].shift()) | (g["t"].diff() > gap)
        ep = new.cumsum()
        span = g.groupby(ep)["split"].agg(lambda s: tuple(sorted(set(s))))
        test_rows = g[g["split"] == "test"]
        ep_test = ep[g["split"] == "test"]
        shared_test = ep_test.map(lambda e: len(span[e]) > 1)
        entry[f"episodes_gap_{int(gap)}s"] = {
            "episodes": int(len(span)), "multi_split_episodes": int(sum(len(v) > 1 for v in span)),
            "test_only_episodes": int(sum(v == ("test",) for v in span)),
            "test_records_in_shared_episodes": int(shared_test.sum()),
            "test_records_in_test_only_episodes": int((~shared_test).sum()) if len(test_rows) else 0,
            "test_records_shared_share": float(shared_test.mean()) if len(test_rows) else None}
    report["classes"][c] = entry
    log(c, json.dumps(entry)[:400])
json.dump(report, open(os.path.join(OUT, "campaign.json"), "w"), indent=1, default=float)
log("done")

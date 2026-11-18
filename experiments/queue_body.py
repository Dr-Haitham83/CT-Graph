# flake8: noqa -- executed after the COMMON prelude and temporal_lib.py (manifest, shard_path).
# QUEUEING UNDER THE RECORDED ARRIVAL TRACES (major revision). Records reach the detector at their availability time hi.
# A single detector instance serves them first-come first-served at a constant rate MU (records/s). Per test stream
# (an independent capture or split segment), with a_t the records arriving in second t, the backlog evolves as
#   Q_t = max(0, Q_{t-1} + a_t - MU)
# and a record arriving in second t waits about Q_t / MU seconds before it is scored (fluid approximation; the micro-batch
# delay comes on top). Two service rates are evaluated per setting (settings MU_AMORTISED and MU_BATCH): the amortised
# throughput of the whole test split, and the micro-batch size divided by the median micro-batch delay.
OUT = "/data/output/queue"
os.makedirs(OUT, exist_ok=True)
times = []
for shard in manifest["shards"]["test"]:
    t = pq.read_table(shard_path("test", shard), columns=["hi", "stream"])
    times.append(pd.DataFrame({"t": t["hi"].to_numpy(zero_copy_only=False), "s": t["stream"].to_numpy(zero_copy_only=False)}))
f = pd.concat(times, ignore_index=True)
report = {"prepared": PREPARED_NAME, "records": int(len(f)), "streams": int(f["s"].nunique()), "rates": {}}
for label, mu in (("amortised", MU_AMORTISED), ("batch", MU_BATCH)):
    waits, weights, peak_q, busy, seconds = [], [], 0.0, 0, 0
    for _, g in f.groupby("s"):
        sec = np.floor(g["t"].to_numpy() - g["t"].min()).astype(np.int64)
        a = np.bincount(sec)
        q, Q = 0.0, np.empty(len(a))
        for i, x in enumerate(a):
            q = max(0.0, q + x - mu)
            Q[i] = q
        nz = a > 0
        waits.append(Q[nz] / mu); weights.append(a[nz])
        peak_q = max(peak_q, float(Q.max()))
        busy += int((Q > 0).sum()); seconds += len(a)
    w, n = np.concatenate(waits), np.concatenate(weights).astype(float)
    order = np.argsort(w); cw = np.cumsum(n[order]) / n.sum()
    q_ = lambda p: float(w[order][np.searchsorted(cw, p)])  # noqa: E731
    report["rates"][label] = {"mu_records_per_s": mu, "seconds_with_backlog_share": busy / max(seconds, 1),
                              "peak_backlog_records": peak_q, "max_wait_s": float(w.max()),
                              "record_wait_p50_s": q_(0.5), "record_wait_p99_s": q_(0.99), "record_wait_p999_s": q_(0.999),
                              "records_waiting_over_1s_share": float(n[w > 1].sum() / n.sum()),
                              "records_waiting_over_10s_share": float(n[w > 10].sum() / n.sum()),
                              "records_waiting_over_60s_share": float(n[w > 60].sum() / n.sum())}
    log(label, json.dumps(report["rates"][label]))
json.dump(report, open(os.path.join(OUT, "queue.json"), "w"), indent=1, default=float)
log("done")

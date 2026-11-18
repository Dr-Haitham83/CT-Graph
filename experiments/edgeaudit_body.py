# flake8: noqa -- executed after the COMMON prelude and temporal_lib.py (manifest, root, classes, shard_path).
# ORDERING-LEAKAGE AUDIT OF TEMPORAL FLOW GRAPHS (label-free definition; labels only for the per-class breakdown).
# Every prepared record i carries an availability interval [lo_i, hi_i]: the collector certainly holds i at hi_i and
# cannot hold it before lo_i (lo = last packet time, hi = lo + export-delay bound g; g = 0 on capture-clock data).
# A temporal flow graph links each root i to the K most recent earlier flows of each of its endpoints. "Earlier" is
#   * START order: flow start time, then source order   (common in temporal-graph intrusion detection)
#   * END order:   flow end time, then source order
#   * CAUSAL:      availability order with admission hi_j <= lo_i (the CT-Graph rule; must give zero violations)
# For every edge j -> i:
#   definite look-ahead  lo_j > hi_i : j reaches the collector only after every moment at which i can be decided;
#   not certifiable      hi_j > lo_i : j cannot be guaranteed to have reached the collector when i is decided.
# Graphs are built per segment (state is reset at segment boundaries), on every modelling split.
OUT = "/data/output/edgeaudit"
os.makedirs(OUT, exist_ok=True)
started = time.time()
KNB = int(globals().get("KNB", 6))
UNSPEC_LOW = (0xFFFF << 32)  # low 64 bits of ::ffff:0.0.0.0 (IPv4-mapped unspecified address)
COLS = ["first", "last", "lo", "hi", "s0", "s1", "d0", "d1", "label", "stream", "source_order"]


def read_split(split):
    parts = []
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=COLS)
        d = {c: t[c].to_numpy(zero_copy_only=False) for c in COLS if c not in ("label", "stream")}
        d["y"] = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        d["stream"] = t["stream"].to_numpy(zero_copy_only=False)
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)


def observed(high, low):
    return (high != 0) | ((low != 0) & (low != UNSPEC_LOW))


def endpoint_codes(f):
    """One integer code per endpoint address (source and destination share the code space)."""
    a = np.concatenate([f["s0"].to_numpy(), f["d0"].to_numpy()])
    b = np.concatenate([f["s1"].to_numpy(), f["d1"].to_numpy()])
    _, code = np.unique(np.stack([a, b], 1), axis=0, return_inverse=True)
    n = len(f)
    ok = np.concatenate([observed(f["s0"].to_numpy(), f["s1"].to_numpy()), observed(f["d0"].to_numpy(), f["d1"].to_numpy())])
    return code.reshape(-1), ok, n


def edges(f, rule):
    """(root, predecessor) index pairs of a K-most-recent-per-endpoint neighbourhood under one ordering rule."""
    n = len(f)
    if rule == "start":
        rank = np.lexsort((f["source_order"].to_numpy(), f["first"].to_numpy()))
    elif rule == "end":
        rank = np.lexsort((f["source_order"].to_numpy(), f["last"].to_numpy()))
    else:
        rank = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy()))
    pos = np.empty(n, np.int64)
    pos[rank] = np.arange(n)
    code, ok, _ = endpoint_codes(f)
    idx = np.concatenate([np.arange(n), np.arange(n)])
    code, idx = code[ok], idx[ok]
    order = np.lexsort((pos[idx], code))
    c, ix = code[order], idx[order]
    roots, preds = [], []
    if rule == "causal":
        # admission hi_j <= lo_i (records with hi <= lo_i form a prefix of the availability order), as in build_ctlg:
        # the predecessors of i at an endpoint are the K latest events of that endpoint inside that prefix.
        hi_sorted = f["hi"].to_numpy()[rank]
        prefix = np.searchsorted(hi_sorted, f["lo"].to_numpy(), side="right") - 1
        latest = np.minimum(prefix, pos - 1)
        keys = c.astype(np.int64) * (n + 1) + pos[ix]
        query = c.astype(np.int64) * (n + 1) + latest[ix]
        q = np.searchsorted(keys, query, side="right") - 1
        for m in range(KNB):
            j = q - m
            v = (j >= 0) & (c[np.maximum(j, 0)] == c) & (latest[ix] >= 0)
            roots.append(ix[v]); preds.append(ix[j[v]])
    else:
        for m in range(1, KNB + 1):
            ok_m = np.zeros(len(c), bool)
            ok_m[m:] = c[m:] == c[:-m]
            r = np.flatnonzero(ok_m)
            roots.append(ix[r]); preds.append(ix[r - m])
    r = np.concatenate(roots) if roots else np.zeros(0, np.int64)
    p = np.concatenate(preds) if preds else np.zeros(0, np.int64)
    keep = r != p
    pair = np.unique(r[keep].astype(np.int64) * n + p[keep])
    return pair // n, pair % n


report = {"prepared": PREPARED_NAME, "K_per_endpoint": KNB, "classes": classes, "splits": {}, "definition": {
    "definite": "lo_j > hi_i", "not_certifiable": "hi_j > lo_i", "intervals": "lo = last packet time; hi = lo + g"}}
acc = {rule: {"edges": 0, "definite": 0, "uncert": 0, "roots": 0, "roots_def": 0, "lead": [],
              "cls_roots": np.zeros(len(classes) + 1, np.int64), "cls_def": np.zeros(len(classes) + 1, np.int64)}
       for rule in ("start", "end", "causal")}
for split in ("train", "val", "calibration", "test"):
    if split not in manifest["shards"]:
        continue
    f = read_split(split)
    log(split, len(f), "rows")
    for s_code, g in f.groupby("stream", sort=False):
        g = g.reset_index(drop=True)
        lo, hi, y = g["lo"].to_numpy(), g["hi"].to_numpy(), g["y"].to_numpy()
        for rule in ("start", "end", "causal"):
            ri, pj = edges(g, rule)
            a = acc[rule]
            definite = lo[pj] > hi[ri]
            uncert = hi[pj] > lo[ri]
            a["edges"] += len(ri); a["definite"] += int(definite.sum()); a["uncert"] += int(uncert.sum())
            has = np.zeros(len(g), bool)
            has[ri[definite]] = True
            roots = np.zeros(len(g), bool)
            roots[ri] = True
            a["roots"] += int(roots.sum()); a["roots_def"] += int(has.sum())
            yy = np.where(y >= 0, y, len(classes))
            a["cls_roots"] += np.bincount(yy[roots], minlength=len(classes) + 1)
            a["cls_def"] += np.bincount(yy[has], minlength=len(classes) + 1)
            if definite.any():
                lead = lo[pj[definite]] - hi[ri[definite]]
                a["lead"].append(np.random.default_rng(0).choice(lead, min(len(lead), 200_000), replace=False))
    del f
for rule, a in acc.items():
    lead = np.concatenate(a["lead"]) if a["lead"] else np.zeros(0)
    report[rule] = {"edges": a["edges"], "definite_edges": a["definite"], "definite_frac": a["definite"] / max(a["edges"], 1),
                    "not_certifiable_edges": a["uncert"], "not_certifiable_frac": a["uncert"] / max(a["edges"], 1),
                    "roots_with_edges": a["roots"], "roots_with_definite": a["roots_def"],
                    "roots_with_definite_frac": a["roots_def"] / max(a["roots"], 1),
                    "per_class_roots_with_definite_frac": {c: (float(a["cls_def"][k] / a["cls_roots"][k]) if a["cls_roots"][k] else None)
                                                           for k, c in enumerate(classes)},
                    "lead_s": {q: float(np.quantile(lead, v)) for q, v in (("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0))} if len(lead) else None}
    log(rule, {k: v for k, v in report[rule].items() if k not in ("per_class_roots_with_definite_frac",)})
assert report["causal"]["definite_edges"] == 0 and report["causal"]["not_certifiable_edges"] == 0, "causal admission violated"
report["elapsed_s"] = time.time() - started
json.dump(report, open(os.path.join(OUT, "edgeaudit.json"), "w"), indent=1, default=float)
log("done")

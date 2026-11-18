
# STREAMING EXTRACTION OF THE RELATIONAL CONTEXT (systems evidence; label-free, test split of each dataset).
# The relational features are sliding-window (10 s / 60 s) sums of per-flow indicators per source / destination /
# (source, destination) pair, plus "first seen within 60 s" tests. They are computed at the FLOW COLLECTOR from the
# exported flow records (NetFlow v9 / IPFIX), never in the switch / router data plane. This kernel implements the
# one-pass streaming form a collector would run and measures it:
#   * exactness: streaming output == the batch library (relational()) on the same flows
#   * cost: single-core nanoseconds per flow (hash-map lookups + O(1) amortised window updates)
#   * live state: active keys, flows inside the 60 s windows, live first-seen entries (bytes)
#   * load: peak flow-record arrival rate (1 s bins) and the resulting headroom = capacity / peak rate
from numba import njit
from numba import types
from numba.typed import Dict

OUT = "/data/output/relsys"
os.makedirs(OUT, exist_ok=True)
started = time.time()
f = load("test")
n = len(f)
stream = pd.factorize(f["stream"])[0].astype(np.int64)
order0 = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), stream))
t_abs = f["hi"].to_numpy(np.float64)[order0]
st = stream[order0]
rel = t_abs - pd.Series(t_abs).groupby(st).transform("min").to_numpy()
src = key(f["s0"].to_numpy()[order0], f["s1"].to_numpy()[order0], st)
dst = key(f["d0"].to_numpy()[order0], f["d1"].to_numpy()[order0], st)
pair = key(src, dst)
dport = f["dport"].to_numpy()[order0].astype(np.int64)
sport = f["sport"].to_numpy()[order0].astype(np.int64)
flags = np.nan_to_num(f[FLAGS].to_numpy()[order0], nan=0).astype(np.int64) if FLAGS else np.zeros(n, np.int64)
syn = (((flags & 0x02) > 0) & ((flags & 0x10) == 0)).astype(np.float64)
rst = ((flags & 0x04) > 0).astype(np.float64)
empty = (np.nan_to_num(f[OUT_BYTES].to_numpy()[order0], nan=0) <= 0).astype(np.float64) if OUT_BYTES else np.zeros(n)
# stream boundaries: the batch library computes per segment (keys carry the stream id, times restart)
log("flows", n, "segments", len(np.unique(st)))


@njit(cache=False)
def mix(a, b):
    return (a * 1099511628211) ^ (b + 2654435761)


@njit(cache=False)
def run(rel, keys, peers, dport, sport, syn, rst, empty, out):
    # keys: (3, n) src / dst / pair hashed keys; peers: (3, n) the other endpoint (unused for pair)
    # out: (n, 3 roles, 2 windows, 9 slots): n, newdport, syn, rst, empty, newpeer, newsport, lgap, lgap2 (window sums)
    n = rel.shape[0]
    W0, W1, H = 10_000_000, 60_000_000, 60_000_000  # integer microseconds (exact window edges)
    S = 9
    stats = np.zeros(4)  # -, -, max live first-seen entries (all roles of the current role pass), purges
    for r in range(3):
        slot = Dict.empty(key_type=types.int64, value_type=types.int64)
        seen_d = Dict.empty(key_type=types.int64, value_type=types.int64)
        seen_p = Dict.empty(key_type=types.int64, value_type=types.int64)
        seen_s = Dict.empty(key_type=types.int64, value_type=types.int64)
        cap = 1024
        head = np.zeros((cap, 2), np.int64)
        sums = np.zeros((cap, 2, S))
        last_t = np.full(cap, -1, np.int64)
        last_dp = np.zeros(cap, np.int64)
        nxt = np.full(n, -1, np.int64)
        tail = np.full(cap, -1, np.int64)
        sig = np.zeros((n, S))
        next_purge = -1
        for i in range(n):
            k = keys[r, i]
            if k in slot:
                s = slot[k]
            else:
                s = len(slot)
                if s >= cap:
                    cap2 = cap * 2
                    h2 = np.zeros((cap2, 2), np.int64); h2[:cap] = head; head = h2
                    s2 = np.zeros((cap2, 2, S)); s2[:cap] = sums; sums = s2
                    l2 = np.full(cap2, -1, np.int64); l2[:cap] = last_t; last_t = l2
                    d2 = np.zeros(cap2, np.int64); d2[:cap] = last_dp; last_dp = d2
                    t2 = np.full(cap2, -1, np.int64); t2[:cap] = tail; tail = t2
                    cap = cap2
                slot[k] = s
                head[s, 0] = i
                head[s, 1] = i
            t = rel[i]
            # per-flow indicators
            v = sig[i]
            v[0] = 1.0
            hk = mix(k, dport[i])
            v[1] = 1.0 if (hk not in seen_d or t - seen_d[hk] > H) else 0.0
            seen_d[hk] = t
            v[2] = syn[i]; v[3] = rst[i]; v[4] = empty[i]
            if r < 2:
                hp = mix(k, peers[r, i])
                v[5] = 1.0 if (hp not in seen_p or t - seen_p[hp] > H) else 0.0
                seen_p[hp] = t
                hs = mix(k, sport[i])
                v[6] = 1.0 if (hs not in seen_s or t - seen_s[hs] > H) else 0.0
                seen_s[hs] = t
            if r == 0:
                if last_t[s] >= 0:
                    g = np.log1p((t - last_t[s]) / 1e6)
                    v[7] = g; v[8] = g * g
            last_t[s] = t
            last_dp[s] = dport[i]
            # append to the key's chain and slide both windows (exact; O(1) amortised)
            if tail[s] >= 0:
                nxt[tail[s]] = i
            tail[s] = i
            for w in range(2):
                width = W0 if w == 0 else W1
                for c in range(S):
                    sums[s, w, c] += v[c]
                h = head[s, w]
                while rel[h] <= t - width:
                    for c in range(S):
                        sums[s, w, c] -= sig[h, c]
                    h = nxt[h]
                head[s, w] = h
                for c in range(S):
                    out[i, r, w, c] = sums[s, w, c]
            # bounded first-seen maps: drop entries older than the horizon once per second of stream time
            if t >= next_purge or t < next_purge - 2 * H:
                for m in (seen_d, seen_p, seen_s):
                    dead = []
                    for kk, vv in m.items():
                        if t - vv > H or vv > t:
                            dead.append(kk)
                    for kk in dead:
                        del m[kk]
                live = len(seen_d) + len(seen_p) + len(seen_s)
                if live > stats[2]:
                    stats[2] = live
                stats[3] += 1
                next_purge = t + 1_000_000
    return stats


rel_us = np.round(rel * 1e6).astype(np.int64)
keys = np.stack([src, dst, pair])
peers = np.stack([dst, src, np.zeros(n, np.int64)])
# compile on a small prefix, then time the full pass
tiny = slice(0, min(n, 5000))
run(rel_us[tiny], keys[:, tiny], peers[:, tiny], dport[tiny], sport[tiny], syn[tiny], rst[tiny], empty[tiny], np.zeros((min(n, 5000), 3, 2, 9)))
out = np.zeros((n, 3, 2, 9), np.float32)
t0 = time.perf_counter()
stats = run(rel_us, keys, peers, dport, sport, syn, rst, empty, out)
elapsed = time.perf_counter() - t0
log("streaming pass", round(elapsed, 2), "s", round(1e9 * elapsed / n), "ns/flow")
# exactness against the batch library (same flows, same order)
batch = relational(f)
inv = np.empty(n, np.int64)
inv[order0] = np.arange(n)
checks = {}
for role, r in (("src", 0), ("dst", 1), ("pair", 2)):
    for w, wi in ((10, 0), (60, 1)):
        cnt = out[:, r, wi, 0]
        mapping = {f"{role}_flows_{w}s": np.log1p(cnt), f"{role}_newdport_frac_{w}s": out[:, r, wi, 1] / cnt,
                   f"{role}_syn_frac_{w}s": out[:, r, wi, 2] / cnt, f"{role}_empty_frac_{w}s": out[:, r, wi, 4] / cnt}
        if r < 2:
            mapping[f"{role}_newpeer_{w}s"] = np.log1p(out[:, r, wi, 5])
            mapping[f"{role}_newsport_frac_{w}s"] = out[:, r, wi, 6] / cnt
        for name, val in mapping.items():
            if name in batch:
                ref = batch[name].to_numpy(np.float64)[order0]
                checks[name] = float(np.nanmax(np.abs(ref - val)))
log("max |streaming - batch| per feature", json.dumps({k: round(v, 6) for k, v in checks.items()}))
# Exact referee for the window counts: the batch library encodes (key, time) as a float composite key_id * 1e7 + t, which
# loses ~1e-6 s at large key ids, so flows lying exactly on a window edge can fall on either side. Recount every role /
# window with an INTEGER-microsecond composite and compare both implementations against it.
referee = {}
for role, r, g in (("src", 0, src), ("dst", 1, dst), ("pair", 2, pair)):
    order = np.lexsort((np.arange(n), rel_us, g))
    gid = pd.factorize(g[order])[0].astype(np.int64)
    span = int(rel_us.max()) + 60_000_001
    assert (int(gid.max()) + 1) * span < 2**62, "integer composite would overflow"
    comp = gid * span + rel_us[order]
    for w, wi in ((10, 0), (60, 1)):
        start = np.searchsorted(comp, comp - w * 10**6, side="right")
        exact_n = np.empty(n)
        exact_n[order] = np.arange(n) - start + 1
        b_n = np.expm1(batch[f"{role}_flows_{w}s"].to_numpy(np.float64)[order0])
        s_n = out[:, r, wi, 0].astype(np.float64)
        referee[f"{role}_flows_{w}s"] = {"streaming_mismatch_rows": int((np.abs(s_n - exact_n) > 0.5).sum()),
                                         "batch_mismatch_rows": int((np.abs(np.round(b_n) - exact_n) > 0.5).sum()),
                                         "rows": int(n)}
# first-seen-within-60 s window sums (new destination port / new peer / new source port), same integer referee
for role, r, g, peer_arr in (("src", 0, src, dst), ("dst", 1, dst, src), ("pair", 2, pair, None)):
    order = np.lexsort((np.arange(n), rel_us, g))
    gid = pd.factorize(g[order])[0].astype(np.int64)
    span = int(rel_us.max()) + 60_000_001
    comp = gid * span + rel_us[order]
    sigs = {1: ("newdport", dport)}
    if peer_arr is not None:
        sigs[5], sigs[6] = ("newpeer", peer_arr), ("newsport", sport)
    for idx, (sname, val) in sigs.items():
        prev = pd.Series(rel_us).groupby(key(g, val)).shift(1).to_numpy()
        new = (np.isnan(prev) | (rel_us - prev > 60_000_000)).astype(np.float64)
        c = np.cumsum(new[order])
        for w, wi in ((10, 0), (60, 1)):
            start = np.searchsorted(comp, comp - w * 10**6, side="right")
            sw = c - np.where(start > 0, c[np.maximum(start - 1, 0)], 0.0)
            exact = np.empty(n)
            exact[order] = sw
            cnt = out[:, r, wi, 0].astype(np.float64)
            bname = f"{role}_{sname}_{w}s" if sname != "newdport" or role != "pair" else None
            fname = f"{role}_{sname}_frac_{w}s"
            if fname in batch:
                b_s = np.round(batch[fname].to_numpy(np.float64)[order0] * cnt)
            elif f"{role}_{sname}_{w}s" in batch:
                b_s = np.round(np.expm1(batch[f"{role}_{sname}_{w}s"].to_numpy(np.float64)[order0]))
            else:
                b_s = None
            referee[f"{role}_{sname}_sum_{w}s"] = {
                "streaming_mismatch_rows": int((np.abs(out[:, r, wi, idx] - exact) > 0.5).sum()),
                "batch_mismatch_rows": (int((np.abs(b_s - exact) > 0.5).sum()) if b_s is not None else None), "rows": int(n)}
log("integer-exact referee", json.dumps(referee))
t1 = time.perf_counter()
_ = relational(f)
batch_s = time.perf_counter() - t1
# load: peak flow-record arrival rate per 1 s of each segment's clock
sec = np.floor(rel).astype(np.int64)
per_sec = pd.Series(1, index=pd.MultiIndex.from_arrays([st, sec])).groupby(level=[0, 1]).size()
peak = int(per_sec.max())
p99 = float(np.percentile(per_sec.to_numpy(), 99))
ns = 1e9 * elapsed / n
# exact live state of a collector that sweeps idle keys: flows inside the 60 s windows (each kept once per role) and
# distinct keys active in the last 60 s (peak over time, per role), via a sliding window over each segment's clock
live_flows, live_keys = 0, {}
for sgm in np.unique(st):
    m = np.flatnonzero(st == sgm)
    tt = rel[m]
    start = np.searchsorted(tt, tt - 60.0, side="right")
    live_flows = max(live_flows, int((np.arange(len(tt)) - start + 1).max()))
    b = np.floor(tt / 60.0).astype(np.int64)
    for role, kk in (("src", src), ("dst", dst), ("pair", pair)):
        per_bin = pd.Series(kk[m]).groupby(b).nunique()
        two = per_bin.to_numpy()[:-1] + per_bin.to_numpy()[1:] if len(per_bin) > 1 else per_bin.to_numpy()
        live_keys[role] = max(live_keys.get(role, 0), int(two.max()))  # upper bound: two adjacent 60 s bins
state_bytes = int(sum(live_keys.values()) * (2 * 9 * 8 + 48) + 3 * live_flows * (9 * 8 + 8) + stats[2] * 24)
report = {"flows": int(n), "segments": int(len(np.unique(st))),
          "streaming_ns_per_flow_single_core": ns, "streaming_flows_per_s_single_core": 1e9 / ns,
          "batch_library_us_per_flow": 1e6 * batch_s / n,
          "peak_flow_records_per_s": peak, "p99_flow_records_per_s": p99,
          "headroom_single_core_vs_peak": (1e9 / ns) / max(peak, 1),
          "peak_flows_in_any_60s_window": live_flows, "peak_active_keys_60s": live_keys,
          "max_live_first_seen_entries": int(stats[2]),
          "state_bytes_estimate": state_bytes,
          "exactness_max_abs_diff_vs_batch": checks, "window_count_referee_integer_us": referee,
          "streaming_exact_vs_integer_referee": bool(all(v["streaming_mismatch_rows"] == 0 for v in referee.values()))}
json.dump(report, open(os.path.join(OUT, "relsys.json"), "w"), indent=1)
print(json.dumps(report, indent=1))
log("relational systems benchmark done")

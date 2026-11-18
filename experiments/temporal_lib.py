# flake8: noqa -- shared library text, executed after the COMMON prelude of build_runs.py (glob, os, json, time, log,
# PREPARED_NAME). Loader + the root-cause relational context + the TEMPORAL feature library: causal multi-scale sliding
# windows (1 s .. 300 s), Hawkes-style exponentially decaying intensities (tau 1 .. 1000 s), multi-scale burst ratios,
# fan-out / fan-in novelty (first contact within 300 s), TCP-flag / protocol / state composition, flow-volume moments,
# exact-size repetition, inter-arrival regularity, time since the last flow, and segment-relative z-scores. Every value
# of a flow uses only flows of the same segment that are at or before it in availability order; labels are never used.
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import lightgbm as lgb
try:
    from numba import njit
except ImportError:  # the runtime image ships numba; install only if it is missing
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numba"], check=True)
    from numba import njit

WINDOWS = (10.0, 60.0)  # relational (root-cause) windows
matches = sorted(glob.glob(f"/data/input/**/{PREPARED_NAME}/manifest.json", recursive=True))
root = next(os.path.dirname(m) for m in matches if os.path.isdir(os.path.join(os.path.dirname(m), "prepared")))
manifest = json.load(open(os.path.join(root, "manifest.json")))
prep = json.load(open(os.path.join(root, "preprocess.json")))
classes = prep["classes"]
K = len(classes)
width = len(prep["means"])
names = prep["feature_names"]
CAT = prep["schema"]["categorical"]
FLOW = names + CAT
label_set = pa.array(classes)


def pick(*candidates):
    return next((c for c in candidates if c in names), None)


FLAGS = pick("TCP_FLAGS", "CLIENT_TCP_FLAGS")
OUT_BYTES = pick("OUT_BYTES", "DstBytes")
STATE = "State" if "State" in CAT else None
RAW = ("in_bytes", "out_bytes", "in_pkts", "out_pkts", "first", "proto")


def shard_path(split, shard):
    return shard["path"] if os.path.exists(shard["path"]) else os.path.join(root, "prepared", split, os.path.basename(shard["path"]))


STREAMS = {}  # stream name -> integer code (strings per row would cost gigabytes on the largest splits)


def load(split, ids=False):
    """All rows of a split (unknown-class rows are kept: they are part of every flow's causal context)."""
    frames = []
    for shard in manifest["shards"][split]:
        t = pq.read_table(shard_path(split, shard), columns=["numeric", "categories", "label", "hi", "source_order", "source_id",
                                                             "s0", "s1", "d0", "d1", "sport", "dport", "stream", *RAW]
                           if ids else ["numeric", "categories", "label", "hi", "source_order",
                                        "s0", "s1", "d0", "d1", "sport", "dport", "stream", *RAW])
        x = t["numeric"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), width).astype(np.float32)
        cats = []
        for j, vocab in enumerate(prep["vocabs"]):
            values = pc.list_element(t["categories"], j)
            index = pc.index_in(values, value_set=pa.array(vocab, type=pa.string()))
            cats.append(pc.if_else(pc.is_null(values), -1, pc.fill_null(index, len(vocab))).to_numpy(zero_copy_only=False).astype(np.float32))
        f = pd.DataFrame(np.concatenate([x, np.column_stack(cats)], 1) if cats else x, columns=FLOW)
        del x, cats
        f["y"] = pc.fill_null(pc.index_in(t["label"], value_set=label_set), -1).to_numpy(zero_copy_only=False)
        for c in ("hi", "source_order", "s0", "s1", "d0", "d1", "sport", "dport", *RAW):
            f[c] = t[c].to_numpy(zero_copy_only=False)
        if ids:
            f["source_id"] = t["source_id"].to_numpy(zero_copy_only=False)
        u, inv = np.unique(t["stream"].to_numpy(zero_copy_only=False), return_inverse=True)
        f["stream"] = np.asarray([STREAMS.setdefault(v, len(STREAMS)) for v in u], np.int32)[inv]
        frames.append(f)
        del t
    return pd.concat(frames, ignore_index=True)


def key(*cols):
    h = np.zeros(len(cols[0]), np.uint64)
    for i, c in enumerate(cols):
        h ^= (np.asarray(c).astype(np.int64).astype(np.uint64) + np.uint64(i + 1)) * np.uint64([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9,
                                                                                            0x27D4EB2F165667C5][i % 4])
        h = (h ^ (h >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9)
    return h.view(np.int64)


def relational(f):
    """Causal window statistics per group (only rows at or before each row in availability order within its stream)."""
    stream = pd.factorize(f["stream"])[0].astype(np.int64)
    order0 = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), stream))
    t = f["hi"].to_numpy(np.float64)[order0]
    st = stream[order0]
    rel = t - pd.Series(t).groupby(st).transform("min").to_numpy()
    src = key(f["s0"].to_numpy()[order0], f["s1"].to_numpy()[order0], st)
    dst = key(f["d0"].to_numpy()[order0], f["d1"].to_numpy()[order0], st)
    dport = f["dport"].to_numpy()[order0].astype(np.int64)
    sport = f["sport"].to_numpy()[order0].astype(np.int64)
    flags = np.nan_to_num(f[FLAGS].to_numpy()[order0], nan=0).astype(np.int64) if FLAGS else np.zeros(len(t), np.int64)
    syn_only = (((flags & 0x02) > 0) & ((flags & 0x10) == 0)).astype(np.float64)
    rst = ((flags & 0x04) > 0).astype(np.float64)
    empty = (np.nan_to_num(f[OUT_BYTES].to_numpy()[order0], nan=0) <= 0).astype(np.float64) if OUT_BYTES else np.zeros(len(t))
    out = {}

    def first_seen(group, value, horizon):
        prev = pd.Series(rel).groupby(key(group, value)).shift(1).to_numpy()
        return (np.isnan(prev) | (rel - prev > horizon)).astype(np.float64)

    for role, g in (("src", src), ("dst", dst), ("pair", key(src, dst))):
        order = np.lexsort((np.arange(len(t)), rel, g))
        gid = pd.factorize(g[order])[0].astype(np.float64)
        composite = gid * 1e7 + rel[order]
        signals = {"newdport": first_seen(g, dport, 60.0)[order], "syn": syn_only[order], "rst": rst[order], "empty": empty[order]}
        if role != "pair":
            peer = dst if role == "src" else src
            signals["newpeer"] = first_seen(g, peer, 60.0)[order]
            signals["newsport"] = first_seen(g, sport, 60.0)[order]
        if role == "src":
            dp = dport[order]
            same = np.r_[False, gid[1:] == gid[:-1]]
            signals["seqport"] = (same & (np.abs(dp - np.r_[dp[0], dp[:-1]]) == 1)).astype(np.float64)
            gap = np.where(same, np.diff(rel[order], prepend=0.0), np.nan)
            lg = np.log1p(np.nan_to_num(gap, nan=0.0))
            signals["lgap"], signals["lgap2"] = lg, lg * lg
        counts = {}
        for w in WINDOWS:
            start = np.searchsorted(composite, composite - w, side="right")
            n = np.arange(len(order)) - start + 1
            counts[w] = n
            res = {f"{role}_flows_{int(w)}s": np.log1p(n)}
            for name, sig in signals.items():
                c = np.cumsum(sig)
                s = c - np.where(start > 0, c[np.maximum(start - 1, 0)], 0.0)
                if name in ("lgap", "lgap2"):
                    continue
                res[f"{role}_{name}_frac_{int(w)}s"] = s / n
                if name.startswith("new"):
                    res[f"{role}_{name}_{int(w)}s"] = np.log1p(s)
            if role == "src":
                c1, c2 = np.cumsum(signals["lgap"]), np.cumsum(signals["lgap2"])
                s1 = c1 - np.where(start > 0, c1[np.maximum(start - 1, 0)], 0.0)
                s2 = c2 - np.where(start > 0, c2[np.maximum(start - 1, 0)], 0.0)
                mean = s1 / n
                sd = np.sqrt(np.maximum(s2 / n - mean * mean, 0))
                res[f"src_gap_cv_{int(w)}s"] = sd / np.maximum(mean, 1e-3)
            for k, v in res.items():
                col = np.empty(len(t), np.float32)
                col[order0[order]] = v
                out[k] = col
        burst = np.empty(len(t), np.float32)
        burst[order0[order]] = np.log1p(counts[10.0]) - np.log1p(counts[60.0])
        out[f"{role}_burstiness"] = burst
    return pd.DataFrame(out)


@njit(cache=False)
def window_starts(g, t, w):
    """Index of the first flow of the same group within (t_i - w, t_i]; rows sorted by (group, time, order)."""
    n = len(g)
    s = np.empty(n, np.int64)
    j = 0
    for i in range(n):
        if i == 0 or g[i] != g[i - 1]:
            j = i
        while t[j] < t[i] - w:
            j += 1
        s[i] = j
    return s


@njit(cache=False)
def ewma_intensity(g, t, tau):
    """Hawkes-style self-exciting intensity: sum over earlier flows of the group of exp(-(t_i - t_j) / tau), plus 1."""
    n = len(g)
    lam = np.empty(n, np.float64)
    for i in range(n):
        if i > 0 and g[i] == g[i - 1]:
            lam[i] = 1.0 + lam[i - 1] * np.exp(-(t[i] - t[i - 1]) / tau)
        else:
            lam[i] = 1.0
    return lam


TWIN = (1.0, 10.0, 60.0, 300.0)      # count windows
FWIN = (10.0, 60.0, 300.0)           # composition / novelty / volume / timing windows
TAUS = (1.0, 10.0, 100.0, 1000.0)    # decay constants of the intensities
HORIZON = 300.0                      # first contact = not seen within the last 300 s
GROUPS = ("T_count", "T_novelty", "T_flags", "T_volume", "T_timing", "T_segz")
# Robustness variants of T_segz (requested explicitly via groups=; never part of GROUPS):
#   T_segz_cont   causal cumulative mean / sd over the WHOLE split in availability order, no reset at segment boundaries
#   T_segz_ewma   causal exponentially forgetting mean / sd (window EWMA_FLOWS flows) across segment boundaries
#   T_segz_static the raw log window counts (a fixed train-fitted z-score is a no-op for trees)
SEGZ_GROUPS = ("T_segz", "T_segz_cont", "T_segz_ewma", "T_segz_static")
EWMA_FLOWS = 50_000


@njit(cache=False)
def ewma_z(v, window):
    """Causal z-score against an exponentially forgetting mean / variance (rate 1 / min(i + 1, window))."""
    n = len(v)
    z = np.empty(n, np.float64)
    m = 0.0
    s = 0.0
    for i in range(n):
        r = 1.0 / min(i + 1, window)
        d = v[i] - m
        m += r * d
        s = (1.0 - r) * (s + r * d * d)
        z[i] = (v[i] - m) / np.sqrt(s + 1e-2)
    return z


def continuous_order(f, order0):
    """Availability order of the whole split across segment boundaries (positions in order0 space). Absolute clocks:
    global (time, source order). Capture-relative clocks (every segment starts near 0): segments concatenated in
    stream-name order."""
    codes = f["stream"].to_numpy()[order0]
    hi = f["hi"].to_numpy(np.float64)[order0]
    so = f["source_order"].to_numpy()[order0]
    starts = pd.Series(hi).groupby(codes).min()
    if np.median(starts.to_numpy()) > 1e7:  # epoch-like clocks
        log("continuous normalisation: global time order across", len(starts), "segments")
        return np.lexsort((so, hi))
    name_of = {c: n for n, c in STREAMS.items()}
    lut = np.zeros(int(codes.max()) + 1, np.int64)
    for r, c in enumerate(sorted(starts.index, key=lambda c: str(name_of.get(c, c)))):
        lut[c] = r
    log("continuous normalisation: capture-relative clocks,", len(starts), "segments concatenated in name order")
    return np.lexsort((so, hi, lut[codes]))


def top_state_codes(f, k=4):
    if STATE is None:
        return []
    v = f[STATE].to_numpy()
    v = v[v >= 0]
    values, counts = np.unique(v, return_counts=True)
    return [float(c) for c in values[np.argsort(-counts)][:k]]


def temporal(f, rows, state_codes, groups=None):
    """Temporal features of the rows `rows` (indices into f); f holds every flow of the split. `groups` restricts the
    computation and the stored columns to those feature groups (None = all)."""
    want = set(GROUPS if groups is None else groups)
    N = len(f)
    st = pd.factorize(f["stream"])[0].astype(np.int64)
    order0 = np.lexsort((f["source_order"].to_numpy(), f["hi"].to_numpy(), st))
    inv0 = np.empty(N, np.int64)
    inv0[order0] = np.arange(N)
    pos0 = inv0[rows]
    st0 = st[order0]
    t0 = f["hi"].to_numpy(np.float64)[order0]
    t0 = t0 - pd.Series(t0).groupby(st0).transform("min").to_numpy()
    arr = lambda c: np.nan_to_num(f[c].to_numpy(np.float64)[order0], nan=0.0)
    src = key(f["s0"].to_numpy()[order0], f["s1"].to_numpy()[order0], st0)
    dst = key(f["d0"].to_numpy()[order0], f["d1"].to_numpy()[order0], st0)
    dport, sport = f["dport"].to_numpy()[order0].astype(np.int64), f["sport"].to_numpy()[order0].astype(np.int64)
    ib, ob, ip = arr("in_bytes"), arr("out_bytes"), arr("in_pkts")
    first = f["first"].to_numpy(np.float64)[order0]
    dur = np.nan_to_num(np.maximum(f["hi"].to_numpy(np.float64)[order0] - first, 0.0), nan=0.0)
    proto = f["proto"].to_numpy()[order0].astype(np.int64)
    sig = {"empty": (ob <= 0).astype(np.float32), "onepkt": (ip <= 1).astype(np.float32),
           "tcp": (proto == 6).astype(np.float32), "udp": (proto == 17).astype(np.float32), "icmp": (proto == 1).astype(np.float32)}
    if FLAGS:
        fl = np.nan_to_num(f[FLAGS].to_numpy()[order0], nan=0).astype(np.int64)
        sig.update(syn=(((fl & 0x02) > 0) & ((fl & 0x10) == 0)).astype(np.float32), rst=((fl & 0x04) > 0).astype(np.float32),
                   fin=((fl & 0x01) > 0).astype(np.float32), psh=((fl & 0x08) > 0).astype(np.float32))
    if STATE:
        sv = f[STATE].to_numpy()[order0]
        for j, c in enumerate(state_codes):
            sig[f"state{j}"] = (sv == c).astype(np.float32)
    vol = {"lib": np.log1p(ib).astype(np.float32), "lob": np.log1p(ob).astype(np.float32),
           "lip": np.log1p(ip).astype(np.float32), "ldur": np.log1p(dur).astype(np.float32)}
    sizes = key(ib.astype(np.int64), ob.astype(np.int64), ip.astype(np.int64))

    def novelty(g, v):
        prev = pd.Series(t0).groupby(key(g, v)).shift(1).to_numpy()
        return (np.isnan(prev) | (t0 - prev > HORIZON)).astype(np.float32)

    pair = key(src, dst)
    roles = {"src": (src, {"newdport": (src, dport), "newpeer": (src, dst), "newsport": (src, sport)}),
             "dst": (dst, {"newdport": (dst, dport), "newpeer": (dst, src), "newsport": (dst, sport)}),
             "pair": (pair, {"newdport": (pair, dport), "newsport": (pair, sport)}),
             "svc": (key(dst, dport), {"newpeer": (key(dst, dport), src)}),
             "scan": (key(src, dport), {"newpeer": (key(src, dport), dst)})}
    small = [s for s in ("empty", "onepkt", "syn", "rst") if s in sig]
    out, group = {}, {}
    segz_sources = {}

    def put(g, name, values):
        if g in want:
            out[name] = np.asarray(values, np.float32)
            group[name] = g

    idx = np.arange(N)
    for role, (g, novel) in roles.items():
        order = np.lexsort((idx, t0, g))
        gs, ts = g[order], t0[order]
        rank = np.empty(N, np.int64)
        rank[order] = idx
        take = rank[pos0]
        same = np.r_[False, gs[1:] == gs[:-1]]
        starts = {w: window_starts(gs, ts, w) for w in TWIN}
        n = {w: (take - starts[w][take] + 1).astype(np.float64) for w in TWIN}

        def wsum(x_sorted, w):
            c = np.cumsum(x_sorted, dtype=np.float64)
            s = starts[w][take]
            return c[take] - np.where(s > 0, c[np.maximum(s - 1, 0)], 0.0)

        for w in TWIN:
            put("T_count", f"{role}_n_{int(w)}s", np.log1p(n[w]))
            if want & set(SEGZ_GROUPS) and w >= 10:
                full = np.empty(N, np.float32)
                full[order] = np.log1p(idx - starts[w] + 1)
                segz_sources[f"{role}_n_{int(w)}s"] = full
        for a, b in ((1.0, 10.0), (10.0, 60.0), (60.0, 300.0)):
            put("T_count", f"{role}_burst_{int(a)}_{int(b)}", np.log1p(n[a]) - np.log1p(n[b]))
        # novelty: distinct new ports / peers per window (counts) and their fractions (scale-free)
        for name, (gk, v) in (novel.items() if want & {"T_count", "T_novelty"} else ()):
            nv = novelty(gk, v)[order]
            for w in FWIN:
                s = wsum(nv, w)
                put("T_count", f"{role}_d{name}_{int(w)}s", np.log1p(s))
                put("T_novelty", f"{role}_{name}_frac_{int(w)}s", s / n[w])
        if role == "src" and "T_novelty" in want:
            dp = dport[order]
            seq = (same & (np.abs(dp - np.r_[dp[0], dp[:-1]]) == 1)).astype(np.float32)
            for w in FWIN:
                put("T_novelty", f"src_seqport_frac_{int(w)}s", wsum(seq, w) / n[w])
        # composition: flags / protocol / state / empty responses / single-packet probes / exact-size repetition
        sz = sizes[order]
        rep = (same & (sz == np.r_[sz[0], sz[:-1]])).astype(np.float32)
        comp = {"repeat": rep, **{k: sig[k][order] for k in (sig if role in ("src", "dst") else small)}}
        for name, x in (comp.items() if "T_flags" in want else ()):
            for w in FWIN:
                put("T_flags", f"{role}_{name}_frac_{int(w)}s", wsum(x, w) / n[w])
        # volume moments
        if role in ("src", "dst", "pair", "svc") and "T_volume" in want:
            for name, x in vol.items():
                if role == "svc" and name != "lob":
                    continue
                xs = x[order]
                for w in FWIN:
                    m1 = wsum(xs, w) / n[w]
                    put("T_volume", f"{role}_{name}_mean_{int(w)}s", m1)
                    if name in ("lib", "lob") and role != "svc":
                        m2 = wsum(xs.astype(np.float64) ** 2, w) / n[w]
                        put("T_volume", f"{role}_{name}_sd_{int(w)}s", np.sqrt(np.maximum(m2 - m1 * m1, 0)))
        # timing: inter-arrival regularity, time since the previous flow, decaying intensities
        if "T_timing" not in want:
            del order, gs, ts, rank
            continue
        gap = np.where(same, np.diff(ts, prepend=0.0), np.nan)
        lg = np.log1p(np.nan_to_num(gap, nan=0.0))
        has = same.astype(np.float64)
        put("T_timing", f"{role}_since_last", np.where(same[take], lg[take], np.nan))
        if role in ("src", "dst", "pair"):
            for w in FWIN:
                k_ = wsum(has, w)
                m1 = wsum(lg, w) / np.maximum(k_, 1)
                m2 = wsum(lg * lg, w) / np.maximum(k_, 1)
                sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
                put("T_timing", f"{role}_lgap_mean_{int(w)}s", np.where(k_ > 0, m1, np.nan))
                put("T_timing", f"{role}_lgap_cv_{int(w)}s", np.where(k_ > 1, sd / np.maximum(m1, 1e-3), np.nan))
        lam = {}
        for tau in TAUS:
            lam[tau] = np.log(ewma_intensity(gs, ts, tau)[take])
            put("T_timing", f"{role}_intensity_{int(tau)}", lam[tau])
        if role in ("src", "dst", "svc"):
            put("T_timing", f"{role}_intensity_ratio_1_100", lam[1.0] - lam[100.0])
            put("T_timing", f"{role}_intensity_ratio_10_1000", lam[10.0] - lam[1000.0])
        del order, gs, ts, rank
    # segment-relative z-scores of the window counts (causal running mean / sd of the segment)
    for name, full in (segz_sources.items() if "T_segz" in want else ()):
        v = full.astype(np.float64)  # already in availability order
        c1 = pd.Series(v).groupby(st0).cumsum().to_numpy()
        c2 = pd.Series(v * v).groupby(st0).cumsum().to_numpy()
        cnt = pd.Series(np.ones(N)).groupby(st0).cumsum().to_numpy()
        m = c1 / cnt
        sd = np.sqrt(np.maximum(c2 / cnt - m * m, 0) + 1e-2)
        put("T_segz", f"z_{name}", ((v - m) / sd)[pos0])
    if want & {"T_segz_cont", "T_segz_ewma"}:
        oc = continuous_order(f, order0)
        for name, full in segz_sources.items():
            v = full.astype(np.float64)[oc]
            z0 = np.empty(N, np.float64)
            if "T_segz_cont" in want:
                cnt = np.arange(1, N + 1, dtype=np.float64)
                m = np.cumsum(v) / cnt
                sd = np.sqrt(np.maximum(np.cumsum(v * v) / cnt - m * m, 0) + 1e-2)
                z0[oc] = (v - m) / sd
                put("T_segz_cont", f"zc_{name}", z0[pos0])
            if "T_segz_ewma" in want:
                z0[oc] = ewma_z(v, EWMA_FLOWS)
                put("T_segz_ewma", f"ze_{name}", z0[pos0])
    for name, full in (segz_sources.items() if "T_segz_static" in want else ()):
        put("T_segz_static", f"raw_{name}", full[pos0])
    cols = list(out)
    M = np.empty((len(rows), len(cols)), np.float32)
    for j, c in enumerate(cols):
        M[:, j] = out.pop(c)
    return M, cols, [group[c] for c in cols]


def merged_clock(f):
    """(hi, stream) of a single continuous monitor timeline for the split."""
    codes = f["stream"].to_numpy()
    hi = f["hi"].to_numpy(np.float64)
    starts = pd.Series(hi).groupby(codes).min()
    ends = pd.Series(hi).groupby(codes).max()
    if np.median(starts.to_numpy()) > 1e7:
        return hi, np.zeros(len(f), np.int32), "absolute clocks: segments merged on their own times"
    name_of = {c: n for n, c in STREAMS.items()}
    order = sorted(starts.index, key=lambda c: str(name_of.get(c, c)))
    offset, t = {}, 0.0
    for c in order:
        offset[c] = t - starts[c]
        t += (ends[c] - starts[c]) + 1.0
    lut = np.zeros(int(codes.max()) + 1)
    for c, o in offset.items():
        lut[c] = o
    return hi + lut[codes], np.zeros(len(f), np.int32), f"capture-relative clocks: {len(order)} segments laid back to back"


def self_check():
    """Brute-force and causality checks of the temporal library on synthetic flows (aborts the kernel on failure)."""
    r = np.random.default_rng(0)
    g = np.sort(r.integers(0, 5, 400)).astype(np.int64)
    t = np.concatenate([np.sort(r.random(int((g == k).sum())) * 60) for k in range(5)])
    s = window_starts(g, t, 10.0)
    lam = ewma_intensity(g, t, 5.0)
    for i in range(len(g)):
        j = min(k for k in range(i + 1) if g[k] == g[i] and t[k] >= t[i] - 10.0)
        assert s[i] == j, ("window start", i)
        brute = sum(np.exp(-(t[i] - t[k]) / 5.0) for k in range(i + 1) if g[k] == g[i])
        assert abs(lam[i] - brute) < 1e-8 * max(1.0, brute), ("intensity", i)
    n = 3000
    f = pd.DataFrame(np.zeros((n, len(FLOW)), np.float32), columns=FLOW)
    if FLAGS:
        f[FLAGS] = r.integers(0, 64, n).astype(np.float32)
    if STATE:
        f[STATE] = r.integers(0, 4, n).astype(np.float32)
    f["hi"] = np.sort(r.random(n) * 900.0)
    f["source_order"] = np.arange(n)
    for c, hi in (("s0", 6), ("s1", 3), ("d0", 6), ("d1", 3), ("sport", 50), ("dport", 30)):
        f[c] = r.integers(0, hi, n)
    for c in ("in_bytes", "out_bytes", "in_pkts"):
        f[c] = r.integers(0, 4, n).astype(float)
    f["out_pkts"] = 1.0
    f["first"] = f["hi"] - r.random(n)
    f["proto"] = r.choice([1, 6, 17], n)
    f["stream"] = r.integers(0, 2, n).astype(np.int32)
    f["y"] = 0
    rows = np.arange(n)
    A, names_, _ = temporal(f, rows, [0.0, 1.0])
    # brute force: flows of the same source (and segment) within the last 10 s, availability order
    col = names_.index("src_n_10s")
    for i in r.choice(n, 50, replace=False):
        same = (f["s0"] == f["s0"][i]) & (f["s1"] == f["s1"][i]) & (f["stream"] == f["stream"][i])
        w = same & (f["hi"] <= f["hi"][i]) & (f["hi"] >= f["hi"][i] - 10.0) & (f["source_order"] <= f["source_order"][i])
        assert abs(A[i, col] - np.log1p(w.sum())) < 1e-5, ("src_n_10s", i)
    # causality: rewriting every flow after the cut must not change any feature of a flow before it
    cut = 450.0
    g2 = f.copy()
    late = g2["hi"] > cut
    for c in ("s0", "d0", "dport", "sport"):
        g2.loc[late, c] = r.integers(0, 6, int(late.sum()))
    g2.loc[late, "in_bytes"] = 999.0
    B, _, _ = temporal(g2, rows, [0.0, 1.0])
    early = (~late).to_numpy()
    assert np.allclose(np.nan_to_num(A[early], nan=-9), np.nan_to_num(B[early], nan=-9), atol=1e-5), "causality violated"
    Z, znames, _ = temporal(f, rows, [0.0, 1.0], groups=["T_segz"])
    for j, nme in enumerate(znames):
        assert np.allclose(np.nan_to_num(Z[:, j], nan=-9), np.nan_to_num(A[:, names_.index(nme)], nan=-9), atol=1e-6), nme
    one = f.copy()
    one["stream"] = 0
    Z1, n1, _ = temporal(one, rows, [0.0, 1.0], groups=["T_segz", "T_segz_cont"])
    for j, nme in enumerate(n1):
        if nme.startswith("zc_"):
            assert np.allclose(Z1[:, j], Z1[:, n1.index("z_" + nme[3:])], atol=1e-4), ("continuous != reset on one segment", nme)
    v = r.normal(size=5000)
    cnt = np.arange(1, 5001)
    m = np.cumsum(v) / cnt
    assert np.allclose(ewma_z(v, 10 ** 9), (v - m) / np.sqrt(np.maximum(np.cumsum(v * v) / cnt - m * m, 0) + 1e-2), atol=1e-8), "ewma"
    one2 = one.copy()
    one2.loc[late, ["s0", "d0", "dport"]] = r.integers(0, 6, (int(late.sum()), 3))
    C1, nc, _ = temporal(one, rows, [0.0, 1.0], groups=["T_segz_cont", "T_segz_ewma", "T_segz_static"])
    C2, _, _ = temporal(one2, rows, [0.0, 1.0], groups=["T_segz_cont", "T_segz_ewma", "T_segz_static"])
    assert np.allclose(C1[early], C2[early], atol=1e-5), "continuous normalisation not causal"
    log("temporal self-check passed", len(names_), "features; group-restricted path consistent; continuous modes consistent")



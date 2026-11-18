
# ToN TEST-ERROR ANALYSIS (development dataset; diagnosis only, nothing is selected here). Deployed ToN model = equal
# weight hybrid of the v4b graph seeds and the smoothed CT-GBDT v4 context branch (Bayes rule).
#  1 confusion of the hybrid on test: per-class recall / precision and where errors go
#  2 anatomy of the test scanning campaign vs train / validation scanning and test Benign (sources, targets, ports,
#    flags, payload, rates, per-source fan-out)
#  3 separability: univariate AUC of every flow / relational / temporal feature for scanning vs Benign on TEST and on
#    TRAIN (a feature that separates in test but not in train cannot be learned; one that separates in both but is
#    unused points to a model problem)
#  4 label-conflict ceiling: test scanning flow vectors identical to Benign-labelled vectors (train or test)
#  5 mixed-label sources: do scanning sources also emit Benign-labelled flows?
#  6 the same for mitm and ransomware (smaller)
import scipy.stats as ss

OUT = "/data/output/tonerr"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261010)
started = time.time()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def read_preds(directory):
    files = sorted(glob.glob(os.path.join(directory, "predictions-rank*.parquet")))
    t = pa.concat_tables([pq.read_table(p, columns=["source_id", "label", "probabilities"]) for p in files])
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object).astype(str)
    o = np.argsort(ids)
    P = t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1)[o]
    return ids[o], t["label"].to_numpy()[o], P.astype(np.float64)


runs = [p for p in sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True)) if "-v4bs" in p]
log("graph runs", runs)
test = [read_preds(os.path.dirname(p)) for p in runs]
ids_t, y_t = test[0][0], test[0][1].astype(np.int64)
G = np.mean([r[2] for r in test], 0)
ct = [np.load(p, allow_pickle=False) for p in sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))]
C = []
for c in ct:
    cid = c["test_ids"].astype(str)
    o = np.argsort(cid)
    pos = np.searchsorted(cid[o], ids_t)
    assert np.array_equal(cid[o][pos], ids_t)
    C.append(c["test_proba"].astype(np.float64)[o][pos])
C = np.mean(C, 0)
H = 0.5 * G + 0.5 * C
pred = H.argmax(1)
keep = y_t >= 0
report = {}
cm = np.bincount(y_t[keep] * K + pred[keep], minlength=K * K).reshape(K, K)
report["hybrid_confusion"] = {classes[i]: {classes[j]: int(cm[i, j]) for j in range(K) if cm[i, j]} for i in range(K) if cm[i].sum()}
rec = {classes[i]: round(float(cm[i, i] / max(cm[i].sum(), 1)), 4) for i in range(K)}
prec = {classes[j]: round(float(cm[j, j] / max(cm[:, j].sum(), 1)), 4) for j in range(K)}
report["recall"], report["precision"] = rec, prec
log("hybrid test recall", rec)

# ---- test / train frames with relational + temporal features
def frame(split, sample_cap=None):
    f = load(split)  # no string ids (20M training ids exhausted memory)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    if sample_cap and len(rows) > sample_cap:
        # keep every scanning / mitm / ransomware row, subsample the rest
        rare = np.isin(y[rows], [classes.index(c) for c in ("scanning", "mitm", "ransomware") if c in classes])
        other = rows[~rare]
        rows = np.sort(np.r_[rows[rare], rng.choice(other, min(len(other), sample_cap), replace=False)])
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    T, tn, tg = temporal(f, rows, [], groups=["T_count", "T_novelty", "T_flags"])  # scanning-relevant groups only
    d = {"y": y[rows],
         "X": np.concatenate([np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW]), rel, T], 1),
         "src": key(f["s0"].to_numpy()[rows], f["s1"].to_numpy()[rows], np.zeros(len(rows), np.int64)),
         "dst": key(f["d0"].to_numpy()[rows], f["d1"].to_numpy()[rows], np.zeros(len(rows), np.int64)),
         "dport": f["dport"].to_numpy()[rows], "sport": f["sport"].to_numpy()[rows], "t": f["hi"].to_numpy(np.float64)[rows],
         "seg": f["stream"].to_numpy()[rows], "in_bytes": f["in_bytes"].to_numpy()[rows], "out_bytes": f["out_bytes"].to_numpy()[rows],
         "in_pkts": f["in_pkts"].to_numpy()[rows], "proto": f["proto"].to_numpy()[rows]}
    names_ = FLOW + list(relational_names) + tn
    del f
    log("frame", split, len(rows), round(time.time() - started), "s")
    return d, names_


TE, NAMES = frame("test")
import gc
gc.collect()
TR, _ = frame("train", sample_cap=1_000_000)
gc.collect()
VA, _ = frame("val", sample_cap=300_000)
gc.collect()
SC, BE = classes.index("scanning"), classes.index("Benign")


def anatomy(d, k, label):
    m = d["y"] == k
    if not m.any():
        return {"rows": 0}
    src, dst, dp = d["src"][m], d["dst"][m], d["dport"][m]
    per_src = pd.DataFrame({"src": src, "dst": dst, "dp": dp}).groupby("src").agg(flows=("dst", "size"), dsts=("dst", "nunique"),
                                                                                    dports=("dp", "nunique"))
    t = d["t"][m]
    return {"rows": int(m.sum()), "sources": int(per_src.shape[0]), "targets": int(len(np.unique(dst))), "dports": int(len(np.unique(dp))),
            "segments": int(len(np.unique(d["seg"][m]))), "duration_s": float(t.max() - t.min()),
            "flows_per_src_median": float(per_src.flows.median()), "dports_per_src_median": float(per_src.dports.median()),
            "dsts_per_src_median": float(per_src.dsts.median()), "top_dports": pd.Series(dp).value_counts().head(8).to_dict(),
            "proto": pd.Series(d["proto"][m]).value_counts().head(4).to_dict(),
            "in_pkts_median": float(np.median(d["in_pkts"][m])), "out_bytes_zero_frac": float((d["out_bytes"][m] <= 0).mean()),
            "in_bytes_median": float(np.median(d["in_bytes"][m]))}


report["scanning_anatomy"] = {s: anatomy(d, SC, s) for s, d in (("train", TR), ("val", VA), ("test", TE))}
report["benign_anatomy_test"] = anatomy(TE, BE, "test")
# where do test scanning flows go, per model
m_sc = (y_t == SC)
report["test_scanning_destinations"] = {name: pd.Series(np.asarray(classes)[P[m_sc].argmax(1)]).value_counts().head(5).to_dict()
                                        for name, P in (("graph", G), ("context", C), ("hybrid", H))}
report["test_scanning_mean_prob_scanning"] = {name: float(P[m_sc, SC].mean()) for name, P in (("graph", G), ("context", C), ("hybrid", H))}


# ---- univariate separability scanning vs Benign, train vs test
def auc(a, b):
    v = np.nan_to_num(np.r_[a, b].astype(np.float64), nan=-1e30)
    r = ss.rankdata(v)
    na, nb = len(a), len(b)
    return float((r[:na].sum() - na * (na + 1) / 2) / (na * nb))


def sep(d, k1, k2, cap=200_000):
    a_rows, b_rows = np.flatnonzero(d["y"] == k1), np.flatnonzero(d["y"] == k2)
    a_rows = rng.choice(a_rows, min(cap, len(a_rows)), replace=False)
    b_rows = rng.choice(b_rows, min(cap, len(b_rows)), replace=False)
    return np.array([abs(auc(d["X"][a_rows, j], d["X"][b_rows, j]) - 0.5) * 2 for j in range(len(NAMES))])


s_tr, s_te = sep(TR, SC, BE), sep(TE, SC, BE)
order = np.argsort(-s_te)
report["separability_scanning_vs_benign"] = [{"feature": NAMES[j], "test": round(float(s_te[j]), 3), "train": round(float(s_tr[j]), 3)}
                                             for j in order[:40]]
report["max_separability"] = {"test": float(s_te.max()), "train": float(s_tr.max())}


# ---- label conflicts and mixed-label sources
def hashes(X):
    return pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(X[:, :len(FLOW)], nan=-7.77e30)), index=False).to_numpy()


h_te, h_tr = hashes(TE["X"]), hashes(TR["X"])
benign_vecs = set(h_tr[TR["y"] == BE]) | set(h_te[TE["y"] == BE])
report["test_scanning_vector_also_benign"] = float(np.isin(h_te[TE["y"] == SC], list(benign_vecs)).mean())
sc_src = set(TE["src"][TE["y"] == SC])
be_from_sc_src = np.isin(TE["src"], list(sc_src)) & (TE["y"] == BE)
report["mixed_label_sources_test"] = {"scanning_sources": len(sc_src), "benign_rows_from_scanning_sources": int(be_from_sc_src.sum()),
                                      "benign_rows_total": int((TE["y"] == BE).sum())}
# ---- mitm and ransomware anatomy + separability
for cname in ("mitm", "ransomware"):
    if cname in classes:
        k = classes.index(cname)
        report[f"{cname}_anatomy"] = {s: anatomy(d, k, s) for s, d in (("train", TR), ("test", TE))}
        s_te_k = sep(TE, k, BE)
        report[f"{cname}_top_separability_test"] = [(NAMES[j], round(float(s_te_k[j]), 3)) for j in np.argsort(-s_te_k)[:15]]
        report[f"test_{cname}_destinations_hybrid"] = pd.Series(np.asarray(classes)[H[y_t == k].argmax(1)]).value_counts().head(5).to_dict()
json.dump(report, open(os.path.join(OUT, "tonerr.json"), "w"), indent=1, default=str)
print(json.dumps({k: report[k] for k in ("recall", "precision", "test_scanning_destinations", "test_scanning_mean_prob_scanning",
                                         "max_separability", "test_scanning_vector_also_benign", "mixed_label_sources_test")}, indent=1, default=str))
print("SCANNING ANATOMY", json.dumps(report["scanning_anatomy"], indent=1, default=str))
print("BENIGN ANATOMY (test)", json.dumps(report["benign_anatomy_test"], indent=1, default=str))
print("TOP SEPARABILITY scanning vs Benign (|AUC-0.5|*2), test vs train:")
for r in report["separability_scanning_vs_benign"][:30]:
    print(f"   {r['feature']:34s} test {r['test']:.3f} train {r['train']:.3f}")
for cname in ("mitm", "ransomware"):
    if f"{cname}_anatomy" in report:
        print(cname.upper(), json.dumps(report[f"{cname}_anatomy"], indent=1, default=str)[:2500])
        print(" top separability test", report[f"{cname}_top_separability_test"])
        print(" hybrid destinations", report[f"test_{cname}_destinations_hybrid"])
log("ToN error analysis done")

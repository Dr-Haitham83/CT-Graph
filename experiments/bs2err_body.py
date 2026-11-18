
# 5G BS2->BS1 TEST-ERROR ANALYSIS (diagnosis only; nothing is selected here, and any later change informed by it is
# disclosed as test-informed). Models: graph seed ensembles v4bc (= frozen v4b configuration) and v6rm (merged-timeline
# relational inputs), the frozen CT-GBDT v4 context branch, and their equal-weight hybrids (decision rule chosen on
# validation by segment CVaR_0.25, exactly as in the post-hoc).
#  A  per-model test macro-MCC and per-class recall / precision (seeds, ensembles, context, hybrids)
#  B  hybrid confusion matrix
#  C  per-capture (test segment) error map
#  D  error attribution by feature-vector evidence: for each wrong test flow, is its per-flow vector (i) seen in TRAIN with
#     the predicted label (the error follows the training labels: label collision / cross-site label shift), (ii) seen in
#     train only with other labels, (iii) never seen in train (novel), and is it ambiguous anywhere in the dataset
#  E  class anatomy train (BS2) vs test (BS1)
#  F  univariate separability of the largest confusions (test vs train) on flow + relational + burst features
#  G  graph vs context complementarity (who is right where; oracle headroom of the fusion)
import scipy.stats as ss

OUT = "/data/output/bs2err"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261012)
started = time.time()
BEN = classes.index("Benign")
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def read_preds(directory):
    files = sorted(glob.glob(os.path.join(directory, "predictions-rank*.parquet")))
    if not files:
        return None
    t = pa.concat_tables([pq.read_table(p, columns=["source_id", "label", "probabilities"]) for p in files])
    ids = np.asarray(t["source_id"].to_pylist(), dtype=object).astype(str)
    o = np.argsort(ids)
    P = t["probabilities"].combine_chunks().values.to_numpy(zero_copy_only=False).reshape(len(t), -1)[o]
    return ids[o], t["label"].to_numpy()[o].astype(np.int64), P.astype(np.float64)


def cm_of(y, pred):
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def scores(y, pred):
    cm = cm_of(y, pred).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    present = t > 0
    return {"macro_mcc": round(float(mcc[present].mean()), 4), "accuracy": round(float(tp.sum() / s), 4),
            "per_class": {classes[i]: {"recall": round(float(tp[i] / t[i]), 4) if t[i] else None,
                                       "precision": round(float(tp[i] / pp[i]), 4) if pp[i] else None,
                                       "mcc": round(float(mcc[i]), 4), "support": int(t[i])} for i in range(K) if present[i]}}


def aligned(ref_ids, ids, values):
    o = np.argsort(ids)
    pos = np.searchsorted(ids[o], ref_ids)
    assert np.all(pos < len(ids)) and np.array_equal(ids[o][pos], ref_ids), "rows do not match"
    return values[o][pos]


# ---- predictions (FAMILIES: graph run tags; the first one is the reference hybrid of B-D)
FAMILIES = tuple(globals().get("FAMILIES", ("v4bc", "v6rm")))
REF = FAMILIES[0]
fam = {}
for tag in FAMILIES:
    runs = [p for p in sorted(glob.glob("/data/input/**/evaluations/*/results.json", recursive=True)) if f"-{tag}s" in p]
    fam[tag] = [{"test": read_preds(os.path.dirname(p)), "val": read_preds(os.path.join(os.path.dirname(p), "validation"))}
                for p in runs]
    log(tag, len(runs), "graph runs")
ref = fam[REF][0]["test"]
ids_t, y_t = ref[0], ref[1]
keep = y_t >= 0
ct = [np.load(p, allow_pickle=False) for p in sorted(glob.glob("/data/input/**/ctgbdt_seed*.npz", recursive=True))]
C_test = np.mean([aligned(ids_t, c["test_ids"].astype(str), c["test_proba"].astype(np.float64)) for c in ct], 0)
prior = np.asarray(ct[0]["prior"], dtype=np.float64)
rules = {"bayes": lambda p: p.argmax(1), "balanced": lambda p: (p / np.maximum(prior, 1e-12)).argmax(1)}
val_ref = fam[REF][0]["val"]
vk = val_ref[1] >= 0
ids_v, y_v = val_ref[0][vk], val_ref[1][vk]
C_val = np.mean([aligned(ids_v, c["val_ids"].astype(str), c["val_proba"].astype(np.float64)) for c in ct], 0)
f_val = load("val", ids=True)
vid = f_val["source_id"].to_numpy().astype(str)
seg_v = aligned(ids_v, vid, f_val["stream"].to_numpy())
del f_val


def cvar(y, pred, seg, alpha=0.25):
    out = []
    for s_ in np.unique(seg):
        m_ = seg == s_
        if m_.sum() >= 100 and len(np.unique(y[m_])) >= 2:
            out.append(scores(y[m_], pred[m_])["macro_mcc"])
    return float(np.sort(out)[:max(1, int(np.ceil(alpha * len(out))))].mean()) if out else -1.0


models, report = {}, {"protocol": "test-error diagnosis (test labels used); nothing selected here"}
for tag, runs in fam.items():
    G_test = np.mean([aligned(ids_t, r["test"][0], r["test"][2]) for r in runs], 0)
    G_val = np.mean([aligned(ids_v, r["val"][0], r["val"][2]) for r in runs], 0)
    for i, r in enumerate(runs):
        models[f"{tag}_seed{i}"] = aligned(ids_t, r["test"][0], r["test"][2]).argmax(1)
    g_rule = max(rules, key=lambda k: scores(y_v, rules[k](G_val))["macro_mcc"])
    models[f"{tag}_graph_ensemble"] = rules[g_rule](G_test)
    h_rule = max(rules, key=lambda k: cvar(y_v, rules[k](0.5 * G_val + 0.5 * C_val), seg_v))
    models[f"{tag}_hybrid"] = rules[h_rule](0.5 * G_test + 0.5 * C_test)
    report.setdefault("rules", {})[tag] = {"graph_ensemble": g_rule, "hybrid": h_rule,
                                           "val_graph": scores(y_v, rules[g_rule](G_val))["macro_mcc"],
                                           "val_hybrid": scores(y_v, rules[h_rule](0.5 * G_val + 0.5 * C_val))["macro_mcc"]}
    fam[tag] = {"G_test": G_test}
c_rule = max(rules, key=lambda k: scores(y_v, rules[k](C_val))["macro_mcc"])
models["context_ctgbdt4"] = rules[c_rule](C_test)
yk = y_t[keep]
report["A_models"] = {n: scores(yk, p[keep]) for n, p in models.items()}
for n, r in report["A_models"].items():
    log("A", n, r["macro_mcc"], {c: v["recall"] for c, v in r["per_class"].items()})
H = models[f"{REF}_hybrid"]
cm = cm_of(yk, H[keep])
report["B_hybrid_confusion"] = {classes[i]: {classes[j]: int(cm[i, j]) for j in range(K) if cm[i, j]} for i in range(K) if cm[i].sum()}
off = [(int(cm[i, j]), classes[i], classes[j]) for i in range(K) for j in range(K) if i != j and cm[i, j]]
report["B_top_confusions"] = sorted(off, reverse=True)[:10]
log("B top confusions", report["B_top_confusions"])

# ---- frames (test ALL labelled rows, train all labelled rows; BS2 is small)
TE = load("test", ids=True)
TR = load("train")
te_rows = np.flatnonzero(TE["y"].to_numpy() >= 0)
te_ids = TE["source_id"].to_numpy().astype(str)[te_rows]
o = np.argsort(te_ids)
pos = np.searchsorted(te_ids[o], ids_t[keep])
assert np.array_equal(te_ids[o][pos], ids_t[keep])
te_take = te_rows[o][pos]  # frame rows in prediction order (labelled test rows)
wrong = H[keep] != yk


def hashes(f, rows):
    X = f[FLOW].to_numpy(np.float32)[rows]
    return pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(X, nan=-7.77e30)), index=False).to_numpy()


tr_rows = np.flatnonzero(TR["y"].to_numpy() >= 0)
h_tr, y_tr = hashes(TR, tr_rows), TR["y"].to_numpy()[tr_rows]
h_te = hashes(TE, te_take)
# train label distribution per vector
tr_tab = pd.DataFrame({"h": h_tr, "y": y_tr}).groupby(["h", "y"]).size().unstack(fill_value=0)
tr_tab = tr_tab.reindex(columns=range(K), fill_value=0)
tr_idx = pd.Index(tr_tab.index)
loc = tr_idx.get_indexer(h_te)
seen = loc >= 0
M = tr_tab.to_numpy()
pred_k = H[keep]
cnt_pred = np.where(seen, M[np.maximum(loc, 0), pred_k], 0)
cnt_true = np.where(seen, M[np.maximum(loc, 0), yk], 0)
cnt_all = np.where(seen, M[np.maximum(loc, 0)].sum(1), 0)
cat = np.where(~seen, "novel_in_train",
               np.where(cnt_pred > cnt_true, "train_majority_is_predicted_label",
                        np.where(cnt_pred > 0, "train_has_both_labels_true_majority", "train_has_only_other_labels")))
report["D_error_attribution"] = {
    "errors": int(wrong.sum()), "test_rows": int(len(yk)),
    "by_category": pd.Series(cat[wrong]).value_counts().to_dict(),
    "by_true_class": {classes[k]: pd.Series(cat[wrong & (yk == k)]).value_counts().to_dict() for k in range(K) if (wrong & (yk == k)).any()},
    "novel_fraction_by_class": {classes[k]: round(float((~seen)[yk == k].mean()), 4) for k in range(K) if (yk == k).any()},
    "error_rate_seen_vs_novel_by_class": {classes[k]: [round(float(wrong[(yk == k) & seen].mean()), 4) if ((yk == k) & seen).any() else None,
                                                     round(float(wrong[(yk == k) & ~seen].mean()), 4) if ((yk == k) & ~seen).any() else None]
                                          for k in range(K) if (yk == k).any()},
    "bayes_ceiling_note": "errors in 'train_majority_is_predicted_label' follow the training labels of an identical vector"}
# the training label mix of test Benign vectors predicted as another class
for k in range(K):
    m_ = wrong & (yk == k) & seen
    if m_.sum():
        mix = M[loc[m_]].sum(0)
        report.setdefault("D_train_label_mix_of_wrong_seen_vectors", {})[classes[k]] = {classes[j]: int(mix[j]) for j in range(K) if mix[j]}
log("D", json.dumps(report["D_error_attribution"]["by_category"]))

# ---- C per-capture error map
inv_streams = {v: k for k, v in STREAMS.items()}
seg_t = TE["stream"].to_numpy()[te_take]
rows_c = []
for s_ in np.unique(seg_t):
    m_ = seg_t == s_
    comp = pd.Series(np.asarray(classes)[yk[m_]]).value_counts().to_dict()
    err = pd.Series([f"{classes[a]}->{classes[b]}" for a, b in zip(yk[m_ & wrong], pred_k[m_ & wrong])]).value_counts().head(4).to_dict()
    t_ = TE["hi"].to_numpy()[te_take][m_]
    rows_c.append({"capture": str(inv_streams.get(int(s_), s_)), "rows": int(m_.sum()), "errors": int((m_ & wrong).sum()),
                   "duration_s": float(t_.max() - t_.min()), "classes": comp, "top_errors": err})
report["C_per_capture"] = sorted(rows_c, key=lambda r: -r["errors"])
for r in report["C_per_capture"][:12]:
    log("C", r["capture"], r["rows"], "errors", r["errors"], r["classes"], r["top_errors"])


# ---- E anatomy
def anatomy(f, rows, y, k):
    m = rows[y == k]
    if not len(m):
        return {"rows": 0}
    src = key(f["s0"].to_numpy()[m], f["s1"].to_numpy()[m])
    dst = key(f["d0"].to_numpy()[m], f["d1"].to_numpy()[m])
    dp = f["dport"].to_numpy()[m]
    per = pd.DataFrame({"s": src, "d": dst, "p": dp}).groupby("s").agg(n=("d", "size"), nd=("d", "nunique"), npo=("p", "nunique"))
    t = f["hi"].to_numpy()[m]
    return {"rows": int(len(m)), "sources": int(len(per)), "targets": int(len(np.unique(dst))), "dports": int(len(np.unique(dp))),
            "captures": int(len(np.unique(f["stream"].to_numpy()[m]))), "span_s": round(float(t.max() - t.min()), 1),
            "flows_per_src_median": float(per.n.median()), "dports_per_src_median": float(per.npo.median()),
            "proto": pd.Series(f["proto"].to_numpy()[m]).value_counts().head(3).to_dict(),
            "top_dports": pd.Series(dp).value_counts().head(5).to_dict(),
            "in_pkts_median": float(np.median(f["in_pkts"].to_numpy()[m])), "in_bytes_median": float(np.median(f["in_bytes"].to_numpy()[m])),
            "out_bytes_zero_frac": round(float((f["out_bytes"].to_numpy()[m] <= 0).mean()), 4)}


report["E_anatomy"] = {classes[k]: {"train": anatomy(TR, tr_rows, y_tr, k), "test": anatomy(TE, te_take, yk, k)} for k in range(K)}
for k in range(K):
    log("E", classes[k], "train", json.dumps(report["E_anatomy"][classes[k]]["train"], default=str)[:400])
    log("E", classes[k], "test ", json.dumps(report["E_anatomy"][classes[k]]["test"], default=str)[:400])

# ---- F separability of the largest confusions (flow + relational + burst features)
def feats(f, rows):
    rel = relational(f).to_numpy(np.float32)[rows]
    T, tn, _ = temporal(f, rows, [], groups=["T_count", "T_novelty", "T_flags"])
    return np.concatenate([f[FLOW].to_numpy(np.float32)[rows], rel, T], 1), FLOW + list(relational_names) + tn


X_te, NAMES = feats(TE, te_take)
X_tr, _ = feats(TR, tr_rows)


def auc(a, b):
    v = np.nan_to_num(np.r_[a, b].astype(np.float64), nan=-1e30)
    r = ss.rankdata(v)
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def sep(X, y, k1, k2, cap=100_000):
    a, b = np.flatnonzero(y == k1), np.flatnonzero(y == k2)
    if not len(a) or not len(b):
        return None
    a, b = rng.choice(a, min(cap, len(a)), replace=False), rng.choice(b, min(cap, len(b)), replace=False)
    return np.array([abs(auc(X[a, j], X[b, j]) - 0.5) * 2 for j in range(X.shape[1])])


report["F_separability"] = {}
for n_, a_, b_ in report["B_top_confusions"][:6]:
    k1, k2 = classes.index(a_), classes.index(b_)
    s_te, s_tr = sep(X_te, yk, k1, k2), sep(X_tr, y_tr, k1, k2)
    if s_te is None:
        continue
    order = np.argsort(-s_te)[:15]
    report["F_separability"][f"{a_} vs {b_} ({n_} errors)"] = {
        "max_test": round(float(s_te.max()), 3), "max_train": (round(float(s_tr.max()), 3) if s_tr is not None else None),
        "top": [(NAMES[j], round(float(s_te[j]), 3), (round(float(s_tr[j]), 3) if s_tr is not None else None)) for j in order]}
    log("F", a_, "vs", b_, report["F_separability"][f"{a_} vs {b_} ({n_} errors)"]["top"][:6])

# ---- G graph vs context complementarity
for tag in FAMILIES:
    g = models[f"{tag}_graph_ensemble"][keep]
    c = models["context_ctgbdt4"][keep]
    h = models[f"{tag}_hybrid"][keep]
    gr, cr = g == yk, c == yk
    oracle = np.where(gr, yk, np.where(cr, yk, h))
    report.setdefault("G_complementarity", {})[tag] = {
        "both_right": int((gr & cr).sum()), "graph_only_right": int((gr & ~cr).sum()), "context_only_right": int((~gr & cr).sum()),
        "both_wrong": int((~gr & ~cr).sum()), "oracle_macro_mcc": scores(yk, oracle)["macro_mcc"],
        "by_class": {classes[k]: {"graph_only_right": int((gr & ~cr & (yk == k)).sum()), "context_only_right": int((~gr & cr & (yk == k)).sum()),
                                  "both_wrong": int((~gr & ~cr & (yk == k)).sum())} for k in range(K) if (yk == k).any()},
        "graph_destinations": {classes[k]: pd.Series(np.asarray(classes)[g[(yk == k) & ~gr]]).value_counts().head(3).to_dict()
                               for k in range(K) if ((yk == k) & ~gr).any()}}
    log("G", tag, {k: v for k, v in report["G_complementarity"][tag].items() if k in ("both_right", "graph_only_right", "context_only_right", "both_wrong", "oracle_macro_mcc")})
json.dump(report, open(os.path.join(OUT, "bs2err.json"), "w"), indent=1, default=str)
print("RULES", json.dumps(report["rules"]))
print("A MODELS")
for n, r in report["A_models"].items():
    print(f"  {n:24s} {r['macro_mcc']:.4f} ", " ".join(f"{c[:8]}={v['recall']}" for c, v in r["per_class"].items()))
print("B HYBRID CONFUSION", json.dumps(report["B_hybrid_confusion"], indent=1))
print("D ERROR ATTRIBUTION", json.dumps(report["D_error_attribution"], indent=1))
print("D TRAIN LABEL MIX", json.dumps(report.get("D_train_label_mix_of_wrong_seen_vectors"), indent=1))
print("C PER CAPTURE", json.dumps(report["C_per_capture"][:15], indent=1, default=str))
print("F SEPARABILITY", json.dumps(report["F_separability"], indent=1))
print("G COMPLEMENTARITY", json.dumps(report["G_complementarity"], indent=1))
log("BS2 error analysis done")

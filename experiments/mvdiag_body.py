
# ROOT CAUSE OF THE MULTI-VIEW CALIBRATION DROP (train / validation / calibration only; test never opened; calibration
# labels are used for DIAGNOSIS only, nothing is selected here). Same views and LightGBM settings as ctgbdt8.
#  1 per-class decomposition of the macro-MCC change (each view and the multi-view mean vs flow-only), both rules
#  2 per-capture breakdown of calibration (class mix, accuracy per capture and model)
#  3 ensemble bias vs variance: Krogh-Vedelsby ambiguity decomposition of the Brier score and view (dis)agreement
#  4 SHAP (LightGBM pred_contrib) of the collapsing class's calibration flows, aggregated per feature group
#  5 covariate shift of the top features for that class: train vs calibration (standardised shift, KS, out-of-support)
#  6 counterfactual group swap: replace one feature group of those flows by same-class TRAIN flows; recovery => cause
#  7 label conflicts: calibration flow vectors identical to training vectors with a different label
#  8 decision-rule effect: which classes the prior-balanced rule over-predicts on calibration
import scipy.stats as ss
OUT = "/data/output/mvdiag"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261003)
started = time.time()
self_check()
relational_names = []
_relational = relational


def relational(f):
    frame = _relational(f)
    relational_names[:] = list(frame.columns)
    return frame


def build(split):
    f = load(split)
    y = f["y"].to_numpy()
    rows = np.flatnonzero(y >= 0)
    rel = relational(f).iloc[rows].to_numpy(np.float32)
    T, t_names, t_groups = temporal(f, rows, STATE_CODES)
    flow = np.column_stack([f[c].to_numpy(np.float32)[rows] for c in FLOW])
    X = np.concatenate([flow, rel, T], 1)
    stream = f["stream"].to_numpy()[rows]
    fl_hash = pd.util.hash_pandas_object(pd.DataFrame(np.nan_to_num(flow, nan=-7.77e30)), index=False).to_numpy()
    del f, rel, T
    log("built", split, X.shape, round(time.time() - started), "s")
    return X, y[rows], stream, fl_hash, list(relational_names), t_names, t_groups


STATE_CODES = top_state_codes(load("train")) if STATE else []
Xtr, ytr, str_tr, htr, REL, TN, TG = build("train")
Xva, yva, str_va, hva, *_ = build("val")
Xca, yca, str_ca, hca, *_ = build("calibration")
n = np.bincount(ytr, minlength=K)
prior = n / n.sum()
wtr = np.ones(len(ytr))  # 5G training splits are below the ctgbdt8 cap, so ctgbdt8 also used every row with weight 1
NAMES = FLOW + REL + TN
GROUP = ["flow"] * len(FLOW) + ["REL"] * len(REL) + list(TG)
F = len(NAMES)
gidx = {g: [j for j in range(F) if GROUP[j] == g] for g in dict.fromkeys(GROUP)}
VIEWS = {"flow": gidx["flow"], "flow+REL": gidx["flow"] + gidx["REL"], "flow+T_segz": gidx["flow"] + gidx.get("T_segz", []),
         "all": list(range(F))}
names_by_code = {v: k for k, v in STREAMS.items()}


def fit(c, seed=42):
    params = {"objective": "multiclass", "num_class": K, "learning_rate": 0.1, "num_leaves": 127, "min_data_in_leaf": 200,
              "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 127,
              "num_threads": os.cpu_count() or 4, "verbose": -1, "seed": seed}
    return lgb.train(params, lgb.Dataset(Xtr[:, c], ytr, weight=wtr), num_boost_round=250)


def decide(P, rule):
    return P.argmax(1) if rule == "bayes" else (P / np.maximum(prior, 1e-12)).argmax(1)


def per_class(y, pred):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(float)
    t, pp = cm.sum(1), cm.sum(0)
    tp = np.diag(cm); s = cm.sum(); fn, fp = t - tp, pp - tp; tn = s - tp - fn - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros(K), where=den > 0)
    return mcc, t > 0, cm


report = {"prepared": PREPARED_NAME, "protocol": "diagnosis on train/validation/calibration; nothing selected; test never opened"}
boosters, P = {}, {}
for v, c in VIEWS.items():
    boosters[v] = fit(c)
    P[v] = {"val": boosters[v].predict(Xva[:, c]), "cal": boosters[v].predict(Xca[:, c])}
    log("fitted", v, len(c))
MV = ["flow+REL", "flow+T_segz", "all"]
P["multiview"] = {s: np.mean([P[v][s] for v in MV], 0) for s in ("val", "cal")}

# ---- 1. per-class decomposition of the macro-MCC change vs flow-only (both rules)
decomp = {}
for rule in ("bayes", "balanced"):
    decomp[rule] = {}
    for split, y in (("val", yva), ("cal", yca)):
        base, present, _ = per_class(y, decide(P["flow"][split], rule))
        npres = present.sum()
        for m in ("flow+REL", "flow+T_segz", "all", "multiview"):
            mcc, _, _ = per_class(y, decide(P[m][split], rule))
            contrib = {classes[k]: round(float((mcc[k] - base[k]) / npres), 4) for k in range(K) if present[k]}
            decomp[rule].setdefault(m, {})[split] = {"macro": float(mcc[present].mean()), "flow_macro": float(base[present].mean()),
                                                     "per_class_mcc": {classes[k]: round(float(mcc[k]), 4) for k in range(K) if present[k]},
                                                     "flow_per_class_mcc": {classes[k]: round(float(base[k]), 4) for k in range(K) if present[k]},
                                                     "contribution_to_macro_change": contrib}
report["decomposition"] = decomp
d = decomp["bayes"]["multiview"]["cal"]["contribution_to_macro_change"]
worst = min(d, key=d.get)
kw = classes.index(worst)
report["collapsing_class"] = worst
log("collapsing class on calibration (bayes):", worst, d[worst])

# ---- 2. per-capture breakdown on calibration + confusion destinations of the collapsing class
cap = {}
for code in np.unique(str_ca):
    m = str_ca == code
    y = yca[m]
    counts = np.bincount(y, minlength=K)
    cap[str(names_by_code.get(code, code))] = {
        "rows": int(m.sum()), "classes": {classes[k]: int(counts[k]) for k in range(K) if counts[k]},
        **{f"acc_{mdl}": round(float((decide(P[mdl]["cal"][m], "bayes") == y).mean()), 4) for mdl in ("flow", "flow+REL", "all", "multiview")}}
report["calibration_captures"] = cap
dest = {}
for mdl in ("flow", "flow+REL", "flow+T_segz", "all", "multiview"):
    pred = decide(P[mdl]["cal"], "bayes")[yca == kw]
    cnt = np.bincount(pred, minlength=K)
    dest[mdl] = {classes[k]: int(cnt[k]) for k in np.argsort(-cnt)[:4] if cnt[k]}
report["collapsing_class_destinations"] = dest
tr_caps = {str(names_by_code.get(c, c)): int((ytr[str_tr == c] == kw).sum()) for c in np.unique(str_tr[ytr == kw])}
report["collapsing_class_train_captures"] = tr_caps

# ---- 3. ensemble bias vs variance (Krogh & Vedelsby 1995 ambiguity decomposition on the Brier score)
amb = {}
for split, y in (("val", yva), ("cal", yca)):
    Y = np.eye(K)[y]
    members = [P[v][split] for v in MV]
    ens = np.mean(members, 0)
    member_err = float(np.mean([((p - Y) ** 2).sum(1).mean() for p in members]))
    ens_err = float(((ens - Y) ** 2).sum(1).mean())
    ambiguity = float(np.mean([((p - ens) ** 2).sum(1).mean() for p in members]))
    preds = np.stack([decide(p, "bayes") for p in members])
    wrong_all = (preds != y).all(0)
    same_wrong = wrong_all & (preds == preds[0]).all(0)
    amb[split] = {"mean_member_brier": member_err, "ensemble_brier": ens_err, "ambiguity": ambiguity,
                  "ambiguity_share": ambiguity / max(member_err, 1e-12),
                  "rows_all_views_wrong": float(wrong_all.mean()), "rows_all_views_same_wrong_class": float(same_wrong.mean()),
                  "rows_views_disagree": float((preds != preds[0]).any(0).mean())}
    mk = y == kw
    if mk.any():
        amb[split]["collapsing_class"] = {"all_views_wrong": float(wrong_all[mk].mean()), "same_wrong_class": float(same_wrong[mk].mean()),
                                          "views_disagree": float((preds[:, mk] != preds[0, mk]).any(0).mean())}
report["ambiguity_decomposition"] = amb

# ---- 4. SHAP of the collapsing class's calibration flows (all-features view), per group, toward true vs predicted class
rows_k = np.flatnonzero(yca == kw)
rows_k = rows_k[rng.permutation(len(rows_k))[:5000]]
shap_groups = {}
if len(rows_k):
    contrib = boosters["all"].predict(Xca[rows_k], pred_contrib=True).reshape(len(rows_k), K, F + 1)
    pred_k = decide(P["all"]["cal"][rows_k], "bayes")
    toward_true = contrib[:, kw, :F]
    toward_pred = contrib[np.arange(len(rows_k)), pred_k, :F]
    for g, cols_ in gidx.items():
        shap_groups[g] = {"toward_true_class": float(toward_true[:, cols_].sum(1).mean()),
                          "toward_predicted_class": float(toward_pred[:, cols_].sum(1).mean())}
    net = (toward_pred - toward_true).mean(0)
    top = np.argsort(-net)[:20]
    report["shap_top_features_pushing_away"] = [(NAMES[j], GROUP[j], round(float(net[j]), 4)) for j in top]
report["shap_by_group"] = shap_groups

# ---- 5. covariate shift of the top features for the collapsing class (train class vs calibration class)
shift = []
tr_k = np.flatnonzero(ytr == kw)
for name, grp, _ in report.get("shap_top_features_pushing_away", []):
    j = NAMES.index(name)
    a, b = Xtr[tr_k, j].astype(np.float64), Xca[yca == kw, j].astype(np.float64)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 10 or len(b) < 10:
        continue
    lo, hi = np.quantile(a, [0.01, 0.99])
    shift.append({"feature": name, "group": grp, "train_median": float(np.median(a)), "cal_median": float(np.median(b)),
                  "std_shift": float((b.mean() - a.mean()) / (a.std() + 1e-9)), "ks": float(ss.ks_2samp(a, b).statistic),
                  "cal_outside_train_1_99": float(((b < lo) | (b > hi)).mean())})
report["collapsing_class_feature_shift"] = shift

# ---- 6. counterfactual group swap (all-features view): same-class TRAIN values for one group => recovery?
cf = {}
mcc0, present, _ = per_class(yca, decide(P["all"]["cal"], "bayes"))
for g, cols_ in gidx.items():
    Xs = Xca.copy()
    idx_ca = np.flatnonzero(yca == kw)
    donors = rng.choice(tr_k, len(idx_ca), replace=True)
    Xs[np.ix_(idx_ca, cols_)] = Xtr[np.ix_(donors, cols_)]
    mcc1, _, _ = per_class(yca, decide(boosters["all"].predict(Xs), "bayes"))
    cf[g] = {"class_mcc_before": round(float(mcc0[kw]), 4), "class_mcc_after": round(float(mcc1[kw]), 4),
             "macro_before": round(float(mcc0[present].mean()), 4), "macro_after": round(float(mcc1[present].mean()), 4)}
    del Xs
report["counterfactual_group_swap"] = cf

# ---- 7. label conflicts: calibration flow vectors seen in training with another label
tr_lab = pd.DataFrame({"h": htr, "y": ytr}).groupby("h").y.agg(lambda s: s.value_counts().index[0])
tr_n = pd.DataFrame({"h": htr, "y": ytr}).groupby("h").y.nunique()
conf = {}
for k in range(K):
    m = yca == k
    if not m.any():
        continue
    hk = pd.Series(hca[m])
    seen = hk.isin(tr_lab.index)
    maj = hk[seen].map(tr_lab)
    conf[classes[k]] = {"rows": int(m.sum()), "vector_seen_in_train": float(seen.mean()),
                        "seen_with_other_majority_label": float((maj != k).mean()) if seen.any() else None,
                        "seen_vectors_multilabel_in_train": float(hk[seen].map(tr_n).gt(1).mean()) if seen.any() else None}
report["label_conflicts_calibration_vs_train"] = conf

# ---- 8. decision rule: classes over-predicted on calibration by the prior-balanced rule
rule_fx = {}
for mdl in ("flow", "all", "multiview"):
    pb, pl = decide(P[mdl]["cal"], "bayes"), decide(P[mdl]["cal"], "balanced")
    cb, cl, ct = np.bincount(pb, minlength=K), np.bincount(pl, minlength=K), np.bincount(yca, minlength=K)
    rule_fx[mdl] = {classes[k]: {"true": int(ct[k]), "bayes_pred": int(cb[k]), "balanced_pred": int(cl[k])} for k in range(K)
                    if ct[k] or cb[k] or cl[k]}
report["decision_rule_effect"] = rule_fx
json.dump(report, open(os.path.join(OUT, "mvdiag.json"), "w"), indent=1, default=float)

print("collapsing class on calibration:", worst)
for rule in ("bayes", "balanced"):
    for m in ("flow+REL", "flow+T_segz", "all", "multiview"):
        r = decomp[rule][m]["cal"]
        print(f"[{rule}] {m:12s} cal macro {r['macro']:.4f} (flow {r['flow_macro']:.4f}); contribution per class {r['contribution_to_macro_change']}")
print("flow per-class cal:", decomp["bayes"]["all"]["cal"]["flow_per_class_mcc"])
print("destinations of", worst, "calibration flows:", dest)
print(worst, "training captures:", tr_caps)
print("calibration captures:", json.dumps(cap, indent=0)[:3000])
print("ambiguity decomposition:", json.dumps(amb, indent=1))
print("SHAP by group (collapsing class, all view):", json.dumps(shap_groups, indent=1))
print("top features pushing away:", report.get("shap_top_features_pushing_away"))
print("feature shift (train vs cal, collapsing class):")
for s_ in shift:
    print("  ", s_)
print("counterfactual group swap:", json.dumps(cf, indent=1))
print("label conflicts:", json.dumps(conf, indent=1))
print("decision rule effect:", json.dumps(rule_fx, indent=1))
log("multi-view diagnosis done")

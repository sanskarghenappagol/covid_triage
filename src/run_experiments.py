"""End-to-end experiment runner.

  1. 5-fold CV hyper-parameter tuning on 70% of Cohort 1
  2. Model selection (data type x feature engineering x classifier) on the 30% held-out part of Cohort 1
  3. External validation of selected models on Cohort 2 (+ 30x bootstrap, paired one-sided t-tests)
  4. Feature importance (permutation) for RadioClinLab models
Usage: python src/run_experiments.py [--data data/synthetic_covid.csv] [--out results]
"""
import argparse, json, os, sys, time
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, precision_recall_curve

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (DATA_TYPES, FE_CONFIGS, PARAM_GRID, TASKS, Fitted, cv_auroc, feature_columns, metrics,
                      split_data, HAS_LGBM)

COLORS = dict(zip(DATA_TYPES, ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:cyan"]))
TITLE = {"icu": "ICU admission", "mv": "Mechanical ventilation", "death": "Death"}


def select_model(tr, te, task, dtype, log):
    cols = feature_columns(tr, dtype)
    Xtr, ytr, Xte, yte = tr[cols].values, tr[task].values, te[cols].values, te[task].values
    fes = [("raw", {})] if dtype == "TriageScore" else FE_CONFIGS          # R-score baseline: plain LR (as in paper)
    models = ["LR"] if dtype == "TriageScore" else list(PARAM_GRID)
    rows = []
    for fe, fp in fes:
        for m in models:
            best = max(PARAM_GRID[m], key=lambda p: cv_auroc(fe, fp, m, p, Xtr, ytr))        # step 1: tune
            fit = Fitted(fe, fp, m, best).fit(Xtr, ytr)
            r = metrics(yte, fit.predict_proba(Xte))                                          # step 2: test
            rows.append(dict(task=task, data=dtype, fe=fe, fe_C=fp.get("C"), model=m, params=json.dumps(best), **r))
    res = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
    b = res.iloc[0]
    log(f"  {task:5s} {dtype:13s} best={b.fe}/{b.fe_C}/{b.model}  testAUROC={b.AUROC:.3f}")
    return res, b


def permutation_importance(fit, X, y, names, repeats, rng):
    from sklearn.metrics import roc_auc_score
    base = roc_auc_score(y, fit.predict_proba(X))
    imp = np.zeros(X.shape[1])
    cand = np.where(fit.selected_mask(X.shape[1]))[0]            # unselected features cannot matter
    for j in cand:
        d = []
        for _ in range(repeats):
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            d.append(base - roc_auc_score(y, fit.predict_proba(Xp)))
        imp[j] = np.mean(d)
    return pd.Series(imp, index=names).sort_values(ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sbu_covid.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_boot", type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs("models", exist_ok=True)
    t0 = time.time()
    logf = open(os.path.join(a.out, "run_log.txt"), "w")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    if not os.path.exists(a.data):
        sys.exit(f"{a.data} not found - run: python src/prepare_data.py")
    df = pd.read_csv(a.data)
    tr, te, c2 = split_data(df)
    log(f"LightGBM available: {HAS_LGBM} | train={len(tr)} test={len(te)} cohort2={len(c2)}")
    for t in TASKS:
        log(f"  positives {t}: train={tr[t].sum()} test={te[t].sum()} cohort2={c2[t].sum()}")

    # ---- steps 1+2 : tuning and selection
    all_rows, best = [], {}
    for task in TASKS:
        for dtype in DATA_TYPES:
            res, b = select_model(tr, te, task, dtype, log)
            all_rows.append(res); best[(task, dtype)] = b
    pd.concat(all_rows).to_csv(os.path.join(a.out, "all_candidates_cohort1_test.csv"), index=False)
    t1 = pd.DataFrame([{**{"task": t, "data": d}, **{k: best[(t, d)][k] for k in ["AUROC", "ACC", "AUPRC", "fe", "fe_C", "model"]}}
                       for t in TASKS for d in DATA_TYPES])
    t1.to_csv(os.path.join(a.out, "table1_cohort1_test.csv"), index=False)

    # ---- step 3 : external validation on Cohort 2
    probs, fitted = {}, {}
    for task in TASKS:
        for dtype in DATA_TYPES:
            b = best[(task, dtype)]
            cols = feature_columns(tr, dtype)
            f = Fitted(b.fe, {"C": b.fe_C} if b.fe_C == b.fe_C and b.fe_C is not None else {}, b.model,
                       json.loads(b.params)).fit(tr[cols].values, tr[task].values)
            fitted[(task, dtype)] = f
            probs[(task, dtype)] = f.predict_proba(c2[cols].values)
            joblib.dump({"model": f, "columns": cols}, f"models/{task}_{dtype}.joblib")
    rows = [{"task": t, "data": d, **metrics(c2[t].values, probs[(t, d)])} for t in TASKS for d in DATA_TYPES]
    t2 = pd.DataFrame(rows); t2.to_csv(os.path.join(a.out, "table2_cohort2.csv"), index=False)
    log("\nCohort 2 external validation:\n" + t2.round(3).to_string(index=False))

    # ROC / PR curves
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, task in enumerate(TASKS):
        y = c2[task].values
        for d in DATA_TYPES:
            m = t2[(t2.task == task) & (t2.data == d)].iloc[0]
            fpr, tpr, _ = roc_curve(y, probs[(task, d)]); pr, rc, _ = precision_recall_curve(y, probs[(task, d)])
            ax[0, j].plot(fpr, tpr, color=COLORS[d], label=f"{d} {m.AUROC:.3f}")
            ax[1, j].plot(rc, pr, color=COLORS[d], label=f"{d} {m.AUPRC:.3f}")
        ax[0, j].plot([0, 1], [0, 1], "--", color="gray"); ax[0, j].set_title(f"{TITLE[task]} - ROC")
        ax[0, j].set_xlabel("False positive rate"); ax[0, j].set_ylabel("True positive rate")
        ax[1, j].set_title(f"{TITLE[task]} - PRC"); ax[1, j].set_xlabel("Recall"); ax[1, j].set_ylabel("Precision")
        ax[0, j].legend(fontsize=8, loc="lower right"); ax[1, j].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig_roc_prc_cohort2.png"), dpi=150); plt.close()

    # bootstrap (same resampled indices for every data type -> paired comparison)
    rng = np.random.default_rng(0); n = len(c2)
    boot = {k: {"AUROC": [], "AUPRC": []} for k in probs}
    for _ in range(a.n_boot):
        idx = rng.integers(0, n, n)
        for (t, d), p in probs.items():
            y = c2[t].values[idx]
            if y.sum() == 0:
                continue
            m = metrics(y, p[idx]); boot[(t, d)]["AUROC"].append(m["AUROC"]); boot[(t, d)]["AUPRC"].append(m["AUPRC"])
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    summ = []
    for i, met in enumerate(["AUROC", "AUPRC"]):
        for j, task in enumerate(TASKS):
            ax[i, j].boxplot([boot[(task, d)][met] for d in DATA_TYPES], notch=False)
            ax[i, j].set_xticks(range(1, len(DATA_TYPES) + 1)); ax[i, j].set_xticklabels(DATA_TYPES)
            ax[i, j].set_title(f"{TITLE[task]} - {met} ({a.n_boot} bootstraps)"); ax[i, j].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig_bootstrap_boxplots.png"), dpi=150); plt.close()
    for (t, d), v in boot.items():
        summ.append({"task": t, "data": d, **{f"{m}_mean": np.mean(v[m]) for m in v},
                     **{f"{m}_ci95": f"{np.percentile(v[m], 2.5):.3f}-{np.percentile(v[m], 97.5):.3f}" for m in v}})
    pd.DataFrame(summ).to_csv(os.path.join(a.out, "table3_bootstrap_summary.csv"), index=False)

    tests = []
    for t in TASKS:
        for better, worse in [("Clin", "TriageScore"), ("ClinVit", "Clin"), ("ClinVitLab", "ClinVit"),
                              ("ClinVitLab", "Clin"), ("ClinVitLab", "VitLab")]:
            for met in ["AUROC", "AUPRC"]:
                x, y = boot[(t, better)][met], boot[(t, worse)][met]
                k = min(len(x), len(y)); r = stats.ttest_rel(x[:k], y[:k], alternative="greater")
                tests.append({"task": t, "H1": f"{better} > {worse}", "metric": met,
                              "mean_diff": np.mean(np.array(x[:k]) - np.array(y[:k])), "p_value": r.pvalue})
    tt = pd.DataFrame(tests); tt.to_csv(os.path.join(a.out, "table4_paired_ttests.csv"), index=False)
    log("\nPaired one-sided t-tests:\n" + tt.round(4).to_string(index=False))

    # ---- step 4 : feature importance (RadioClinLab)
    cols = feature_columns(tr, "ClinVitLab"); fi_rows = []
    for task in TASKS:
        imp = permutation_importance(fitted[(task, "ClinVitLab")], c2[cols].values, c2[task].values, cols, 5,
                                     np.random.default_rng(1))
        for k, v in imp.head(10).items():
            fi_rows.append({"task": task, "feature": k, "importance(AUROC drop)": round(v, 4)})
    pd.DataFrame(fi_rows).to_csv(os.path.join(a.out, "table5_top_features.csv"), index=False)
    log("\nTop features:\n" + pd.DataFrame(fi_rows).to_string(index=False))
    log(f"\nDone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

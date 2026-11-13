"""Core ML pipeline: data groups, SMOTE-ENN resampling, LASSO selection, model zoo, CV tuning."""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
try:  # optional, as in the original paper
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

TASKS = ["icu", "mv", "death"]
DATA_TYPES = ["Clin", "ClinVit", "ClinVitLab", "VitLab", "TriageScore"]


# ---------------------------------------------------------------- data
def feature_columns(df, data_type):
    """Column-prefix convention: demo_+clin_ (history) / vit_ (vitals) / lab_ (blood tests) / base_ (baseline score)."""
    pre = lambda *p: [c for c in df.columns if c.startswith(p)]
    hist, vit, lab, base = pre("demo_", "clin_"), pre("vit_"), pre("lab_"), pre("base_")
    return {"Clin": hist, "ClinVit": hist + vit, "ClinVitLab": hist + vit + lab,
            "VitLab": vit + lab, "TriageScore": base}[data_type]


def split_data(df, seed=0):
    """Cohort 1 -> 70% train / 30% test (stratified on ICU+MV+death any-event); Cohort 2 = external validation."""
    from sklearn.model_selection import train_test_split
    c1, c2 = df[df.cohort == 1], df[df.cohort == 2]
    strat = c1[TASKS].max(axis=1)
    tr, te = train_test_split(c1, test_size=0.3, random_state=seed, stratify=strat)
    return tr, te, c2


# ---------------------------------------------------------------- resampling
def smote(X, y, ratio, rng, k=5):
    """Oversample minority class with SMOTE until minority/majority == ratio."""
    Xm = X[y == 1]
    n_new = int(ratio * (y == 0).sum()) - len(Xm)
    if n_new <= 0 or len(Xm) < 3:
        return X, y
    k = min(k, len(Xm) - 1)
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    base = rng.integers(0, len(Xm), n_new)
    nb = nbr[base, rng.integers(0, k, n_new)]
    new = Xm[base] + rng.random((n_new, 1)) * (Xm[nb] - Xm[base])
    return np.vstack([X, new]), np.concatenate([y, np.ones(n_new, dtype=int)])


def enn(X, y, k=3):
    """Edited Nearest Neighbours: drop samples whose k neighbours mostly disagree with their label."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    keep = ((y[idx].mean(axis=1) > 0.5).astype(int) == y)
    if (y[keep] == 1).sum() < 3:
        return X, y
    return X[keep], y[keep]


def smote_enn(X, y, ratio=0.5, seed=0):
    """SMOTEENN (Batista et al., 2003); own implementation so imbalanced-learn is not required."""
    rng = np.random.default_rng(seed)
    return enn(*smote(X, y, ratio, rng))


# ---------------------------------------------------------------- models
def make_model(name, p, seed=0):
    if name == "LR":
        return LogisticRegression(C=p["C"], max_iter=2000, random_state=seed)
    if name == "SVM":
        return SVC(C=p["C"], kernel="rbf", gamma="scale", probability=True, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(n_estimators=150, max_depth=p["max_depth"], min_samples_leaf=3,
                                      n_jobs=1, random_state=seed)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(p["h"],), alpha=p["alpha"], max_iter=300,
                             early_stopping=True, random_state=seed)
    if name == "GBM":  # LightGBM if available, otherwise sklearn's histogram GBM (same algorithm family)
        if HAS_LGBM:
            return LGBMClassifier(n_estimators=150, learning_rate=p["lr"], num_leaves=15, subsample=0.8,
                                  colsample_bytree=0.8, verbose=-1, random_state=seed)
        return HistGradientBoostingClassifier(max_iter=150, learning_rate=p["lr"], max_leaf_nodes=15, random_state=seed)
    raise ValueError(name)


PARAM_GRID = {
    "LR": [{"C": 0.1}, {"C": 1.0}],
    "SVM": [{"C": 1.0}, {"C": 10.0}],
    "RF": [{"max_depth": 5}, {"max_depth": 10}],
    "MLP": [{"h": 32, "alpha": 1e-2}, {"h": 64, "alpha": 1e-1}],
    "GBM": [{"lr": 0.05}, {"lr": 0.1}],
}
FE_CONFIGS = [("raw", {}), ("smoteenn", {}), ("smoteenn_lasso", {"C": 0.1}), ("smoteenn_lasso", {"C": 1.0})]


class Fitted:
    """Imputer(median) -> scaler -> [SMOTEENN] -> [LASSO select] -> classifier. Predicts probabilities."""

    def __init__(self, fe, fe_p, model, params, seed=0):
        self.fe, self.fe_p, self.model, self.params, self.seed = fe, fe_p, model, params, seed

    def fit(self, X, y):
        self.med = np.nan_to_num(np.nanmedian(X, axis=0))
        X = np.where(np.isnan(X), self.med, X)
        self.sc = StandardScaler().fit(X)
        X = self.sc.transform(X)
        if self.fe != "raw":
            X, y = smote_enn(X, y, seed=self.seed)
        self.sel = None
        if self.fe == "smoteenn_lasso":
            l1 = LogisticRegression(penalty="l1", solver="liblinear", C=self.fe_p["C"] * 0.1, random_state=self.seed)
            sel = SelectFromModel(l1, threshold=1e-6).fit(X, y)
            if sel.get_support().sum() >= 2:
                self.sel, X = sel, sel.transform(X)
        self.clf = make_model(self.model, self.params, self.seed).fit(X, y)
        return self

    def _prep(self, X):
        X = self.sc.transform(np.where(np.isnan(X), self.med, X))
        return self.sel.transform(X) if self.sel is not None else X

    def predict_proba(self, X):
        return self.clf.predict_proba(self._prep(X))[:, 1]

    def selected_mask(self, n_feat):
        return self.sel.get_support() if self.sel is not None else np.ones(n_feat, bool)


def cv_auroc(fe, fe_p, model, params, X, y, folds=5, seed=0):
    """5-fold CV AUROC; resampling/selection happen INSIDE each training fold (no leakage)."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    s = []
    for a, b in skf.split(X, y):
        m = Fitted(fe, fe_p, model, params, seed).fit(X[a], y[a])
        s.append(roc_auc_score(y[b], m.predict_proba(X[b])))
    return float(np.mean(s))


def metrics(y, p):
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "ACC": accuracy_score(y, p >= 0.5)}

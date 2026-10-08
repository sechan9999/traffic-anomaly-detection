"""Feature engineering + 5-class classification.

Classes: legitimate | duplicate | retry | anomalous | ddos.
Features are engineered from transaction/connection telemetry plus the
detection signals (velocity, ratios, robust z's, co-spike counts):

  velocity, syn_ratio, retry_ratio, dup_ratio, dports, bytes_per_conn,
  rz_c, rz_b, co_spike_n, subnet_spike_frac, how_sin, how_cos

Compares a HistGradientBoosting classifier against an interpretable
rules-only baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

CLASSES = ["legitimate", "duplicate", "retry", "anomalous", "ddos"]
FEATURES = ["velocity", "syn_ratio", "retry_ratio", "dup_ratio", "dports",
            "bytes_per_conn", "rz_c", "rz_b", "co_spike_n",
            "subnet_spike_frac", "how_sin", "how_cos"]


def engineer(alerts: pd.DataFrame) -> pd.DataFrame:
    df = alerts.copy()
    conns = df["conns"].clip(lower=1)
    df["velocity"] = df["conns"] / 5.0
    df["syn_ratio"] = df["syn_count"] / conns
    df["retry_ratio"] = df["retries"] / conns
    df["dup_ratio"] = df["dups"] / conns
    df["bytes_per_conn"] = df["nbytes"] / conns
    subnet_size = df.groupby("subnet")["ip"].transform("nunique")
    df["subnet_spike_frac"] = df["co_spike_n"] / subnet_size
    df["how_sin"] = np.sin(2 * np.pi * df["how"] / 168.0)
    df["how_cos"] = np.cos(2 * np.pi * df["how"] / 168.0)
    return df


def rules_baseline(df: pd.DataFrame) -> pd.Series:
    """Interpretable rules-only classifier for comparison."""
    pred = pd.Series("legitimate", index=df.index)
    ddos = (df["syn_ratio"] > 5.0) | ((df["rz_c"] >= 6.0) & (df["co_spike_n"] >= 8))
    dup = df["dup_ratio"] > 0.40
    ret = df["retry_ratio"] > 0.30
    anom = df["rz_c"] >= 4.0
    pred[anom] = "anomalous"
    pred[ret] = "retry"
    pred[dup] = "duplicate"
    pred[ddos] = "ddos"
    return pred


def train_classifier(df: pd.DataFrame, seed: int = 7) -> dict:
    X = df[FEATURES].fillna(0.0).values
    y = df["label"].values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.30, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         class_weight="balanced",
                                         random_state=seed)
    clf.fit(X[tr], y[tr])
    pred_clf = clf.predict(X[te])
    pred_rules = rules_baseline(df.iloc[te]).values
    rep_clf = classification_report(y[te], pred_clf, labels=CLASSES,
                                    output_dict=True, zero_division=0)
    rep_rules = classification_report(y[te], pred_rules, labels=CLASSES,
                                      output_dict=True, zero_division=0)

    def table(rep):
        return pd.DataFrame(
            {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                 "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])}
             for c in CLASSES}).T.round(3)

    # HistGradientBoosting has no built-in importances -> permutation importance
    # on a small stratified subsample of the test set (fast, model-agnostic).
    from sklearn.inspection import permutation_importance
    sub_rng = np.random.default_rng(seed)
    sub = sub_rng.choice(te, size=min(3000, len(te)), replace=False)
    pi = permutation_importance(clf, X[sub], y[sub], n_repeats=3,
                                random_state=seed)
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)
    return {"clf_report": table(rep_clf), "rules_report": table(rep_rules),
            "feature_importance": imp.round(3), "test_index": te,
            "test_true": y[te], "test_pred_clf": pred_clf,
            "test_pred_rules": pred_rules, "model": clf}

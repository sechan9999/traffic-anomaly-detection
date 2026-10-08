"""Behavioral-drift monitor.

Compares each IP's recent hour-of-week median profile against the fitted
baseline profile: drift_score = mean over the 168 slots of
|recent_med - base_med| / (base_med + 1). Flags IPs above a threshold so
models can be retuned before drift turns into false positives (or missed
attacks hiding inside a shifted baseline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import add_how


def drift_report(df_hist: pd.DataFrame, df_recent: pd.DataFrame,
                 threshold: float = 0.35) -> pd.DataFrame:
    base = (add_how(df_hist).groupby(["ip", "how"])["conns"].median()
            .unstack(fill_value=0))
    recent = (add_how(df_recent).groupby(["ip", "how"])["conns"].median()
              .unstack(fill_value=0))
    common = base.index.intersection(recent.index)
    base, recent = base.loc[common], recent.loc[common]
    score = (recent - base).abs().div(base + 1.0).mean(axis=1)
    out = pd.DataFrame({"drift_score": score.round(3)})
    out["drifted"] = out["drift_score"] > threshold
    return out.sort_values("drift_score", ascending=False)

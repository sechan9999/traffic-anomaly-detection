"""Validation: binary FPR/FNR, per-incident detection latency,
threshold tuning against a target FPR, and a legitimate-traffic sanity check
(the nightly batch job must not alert).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import detect


def binary_rates(alerts: pd.DataFrame) -> dict:
    y = (alerts["label"] != "legitimate").values
    p = alerts["alert"].values
    tp, fp = int((p & y).sum()), int((p & ~y).sum())
    fn, tn = int((~p & y).sum()), int((~p & ~y).sum())
    return {"FPR": round(fp / max(fp + tn, 1), 4),
            "FNR": round(fn / max(fn + tp, 1), 4),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def detection_latency(alerts: pd.DataFrame, incidents: list[dict],
                      win_min: int = 5) -> pd.DataFrame:
    """Windows (and minutes) from incident start to the first alert on any
    participating IP. None = missed entirely."""
    rows = []
    for inc in incidents:
        ips = set(inc["ips"])
        sub = alerts[(alerts["ip"].isin(ips)) & (alerts["widx"] >= inc["w0"])
                     & (alerts["widx"] < inc["w1"]) & alerts["alert"]]
        if len(sub):
            first = int(sub["widx"].min()) - inc["w0"]
            rows.append({"incident": inc["id"], "class": inc["class"],
                         "latency_windows": first,
                         "latency_min": first * win_min,
                         "n_ips_alerted": int(sub["ip"].nunique()),
                         "n_ips_total": len(ips)})
        else:
            rows.append({"incident": inc["id"], "class": inc["class"],
                         "latency_windows": None, "latency_min": None,
                         "n_ips_alerted": 0, "n_ips_total": len(ips)})
    return pd.DataFrame(rows)


def tune_threshold(scored: pd.DataFrame, df_hist: pd.DataFrame, baseline,
                   target_fpr: float = 0.01,
                   val_max_widx: int | None = None) -> tuple[pd.DataFrame, float, float]:
    """2-D sweep over robust-z threshold k and count-model alpha, on a
    validation slice (avoids tuning on the same windows we finally report).
    Picks the feasible point (FPR <= target) with the lowest FNR; falls back
    to the lowest-FPR point if the target is unreachable."""
    sv = scored if val_max_widx is None else scored[scored["widx"] < val_max_widx]
    rows = []
    for k in [3.0, 4.0, 5.0, 6.0]:
        for alpha in [1e-3, 1e-4, 1e-5]:
            a = detect.build_alerts(sv, df_hist, baseline, k=k, alpha=alpha)
            r = binary_rates(a)
            rows.append({"k": k, "alpha": alpha, "FPR": r["FPR"],
                         "FNR": r["FNR"], "alerts": int(a["alert"].sum())})
    sweep = pd.DataFrame(rows)
    feas = sweep[sweep["FPR"] <= target_fpr]
    if len(feas):
        best = feas.sort_values("FNR").iloc[0]
    else:
        best = sweep.sort_values("FPR").iloc[0]
    return sweep, float(best["k"]), float(best["alpha"])


def batch_job_check(alerts: pd.DataFrame, meta: dict) -> dict:
    """The nightly batch spike is legitimate: alert rate on its windows
    should be ~0 for batch IPs."""
    is_batch = (alerts["ip"].isin(meta["batch_ips"])
                & (alerts["widx"] % 288 >= 24) & (alerts["widx"] % 288 < 36))
    sub = alerts[is_batch]
    return {"batch_windows": int(len(sub)),
            "batch_alerts": int(sub["alert"].sum()),
            "batch_alert_rate": round(float(sub["alert"].mean()), 4)}

"""Per-incident evidence bundles (JSON) for agentic-AI investigation.

Each bundle is self-contained: incident id/class, time window, top-5 entities
by robust z, key feature stats, which detectors fired, correlated-entity
count, and the classifier's verdict. Designed to be dropped straight into an
investigation agent's context.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_bundle(alerts: pd.DataFrame, incident: dict,
                 clf_pred: pd.Series | None = None) -> dict:
    ips = set(incident["ips"])
    sub = alerts[(alerts["ip"].isin(ips)) & (alerts["widx"] >= incident["w0"])
                 & (alerts["widx"] < incident["w1"])].copy()
    top = (sub.groupby("ip")["rz_c"].max().sort_values(ascending=False)
           .head(5).round(2).to_dict())
    conns = sub["conns"]
    detectors = [d for d, c in
                 [("robust-z", "z_alert"), ("count-model", "count_alert"),
                  ("co-spike-upgrade", "ddos_upgrade")] if sub[c].any()]
    bundle = {
        "incident_id": incident["id"],
        "true_class": incident["class"],
        "window": {"start_ts": str(sub["ts"].min()), "end_ts": str(sub["ts"].max()),
                   "n_windows": int(incident["w1"] - incident["w0"])},
        "entities": {"n_total": len(ips), "n_alerted": int(sub[sub["alert"]]["ip"].nunique()),
                     "top5_by_robust_z": top,
                     "max_co_spike_in_subnet": int(sub["co_spike_n"].max())},
        "key_stats": {
            "peak_conns_per_5min": int(conns.max()),
            "median_conns_per_5min": float(conns.median()),
            "max_syn_ratio": round(float((sub["syn_count"] / sub["conns"].clip(lower=1)).max()), 2),
            "max_retry_ratio": round(float((sub["retries"] / sub["conns"].clip(lower=1)).max()), 3),
            "max_dup_ratio": round(float((sub["dups"] / sub["conns"].clip(lower=1)).max()), 3),
            "max_rz_c": round(float(sub["rz_c"].max()), 2)},
        "detectors_fired": detectors,
        "reasons_sample": sub[sub["alert"]]["reason"].head(3).tolist(),
    }
    if clf_pred is not None:
        votes = clf_pred.loc[sub.index].value_counts()
        bundle["classifier_verdict"] = {
            "majority_class": str(votes.index[0]),
            "vote_share": round(float(votes.iloc[0] / votes.sum()), 3)}
    return bundle


def write_bundles(alerts: pd.DataFrame, incidents: list[dict],
                  outdir: Path, clf_pred: pd.Series | None = None) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for inc in incidents:
        b = build_bundle(alerts, inc, clf_pred)
        p = outdir / f"{inc['id']}.json"
        p.write_text(json.dumps(b, indent=2))
        paths.append(p)
    return paths

"""End-to-end: baseline -> detect -> classify -> evaluate -> drift -> evidence.

Run `gen_telemetry.py` first. Saves CSVs + evidence JSONs to outputs/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import baselines, detect, classify, evaluate, drift, evidence

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
START = pd.Timestamp("2026-06-01")
HIST_DAYS = 8
WIN_PER_DAY = 288


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print("== 1. load telemetry ==")
    df = pd.read_csv(DATA / "telemetry.csv", parse_dates=["ts"])
    incidents = json.loads((DATA / "incidents.json").read_text())
    meta = json.loads((DATA / "meta.json").read_text())
    df["widx"] = ((df["ts"] - START) // pd.Timedelta("5min")).astype(int)
    hist = df[df["widx"] < HIST_DAYS * WIN_PER_DAY].copy()
    scoring = df[df["widx"] >= HIST_DAYS * WIN_PER_DAY].copy()
    print(f"{len(df)} rows, {df['ip'].nunique()} ips; "
          f"hist={len(hist)}, scoring={len(scoring)}")

    print("== 2. hour-of-week baselines (fit on clean days 0-7) ==")
    base = baselines.HourOfWeekBaseline().fit(hist)
    scored = base.score(scoring)
    print(f"profile slots: {len(base.prof_)} (ip x how)")

    print("== 3. detection (z + count-model + co-spike upgrade) ==")
    alerts = detect.build_alerts(scored, hist, base, k=4.0)
    print("alert counts:", alerts[["z_alert", "count_alert", "ddos_upgrade", "alert"]]
          .sum().to_dict())

    print("== 4. classification (5 classes) ==")
    feats = classify.engineer(alerts)
    res = classify.train_classifier(feats)
    print("\n[HistGradientBoosting per-class P/R/F1]")
    print(res["clf_report"].to_string())
    print("\n[rules-only baseline per-class P/R/F1]")
    print(res["rules_report"].to_string())
    print("\n[top features by permutation importance]")
    print(res["feature_importance"].head(6).to_string())
    res["clf_report"].to_csv(OUT / "classification_report_clf.csv")
    res["rules_report"].to_csv(OUT / "classification_report_rules.csv")

    print("== 5. threshold tuning on validation slice (days 8-10; target FPR<=1%) ==")
    val_max = 11 * WIN_PER_DAY
    sweep, best_k, best_alpha = evaluate.tune_threshold(
        scored, hist, base, target_fpr=0.01, val_max_widx=val_max)
    print(sweep.to_string(index=False))
    print(f"chosen: k={best_k}, alpha={best_alpha:g}")

    print("== 6. final binary evaluation with tuned thresholds ==")
    alerts = detect.build_alerts(scored, hist, base, k=best_k, alpha=best_alpha)
    rates = evaluate.binary_rates(alerts)
    print(f"FPR={rates['FPR']} FNR={rates['FNR']} "
          f"(TP={rates['TP']} FP={rates['FP']} FN={rates['FN']} TN={rates['TN']})")
    keep = ["ts", "widx", "ip", "subnet", "endpoint", "conns", "nbytes",
            "syn_count", "dports", "retries", "dups", "label",
            "rz_c", "rz_b", "co_spike_n", "z_alert", "count_alert",
            "ddos_upgrade", "alert", "reason", "how"]
    alerts[keep].to_csv(OUT / "alerts.csv", index=False)

    print("== 7. detection latency per incident ==")
    lat = evaluate.detection_latency(alerts, incidents)
    print(lat.to_string(index=False))
    lat.to_csv(OUT / "latency.csv", index=False)

    print("== 8. legitimate batch-job sanity check ==")
    bc = evaluate.batch_job_check(alerts, meta)
    print(f"batch windows={bc['batch_windows']} alerts={bc['batch_alerts']} "
          f"rate={bc['batch_alert_rate']}")

    print("== 9. behavioral drift (recent week vs baseline) ==")
    drep = drift.drift_report(hist, scoring)
    print(drep.head(8).to_string())
    n_flagged = int(drep["drifted"].sum())
    recovered = len(set(drep[drep["drifted"]].index) & set(meta["drifted_ips"]))
    print("flagged: %d IPs; injected drifted recovered: %d/5" % (n_flagged, recovered))
    drep.to_csv(OUT / "drift.csv")

    print("== 10. evidence bundles ==")
    X_all = feats[classify.FEATURES].fillna(0.0).values
    clf_pred = pd.Series(res["model"].predict(X_all), index=feats.index)
    paths = evidence.write_bundles(alerts, incidents, OUT / "evidence", clf_pred)
    print(f"wrote {len(paths)} bundles, e.g. {paths[0].name}")
    print("sample (A_syn_flood):")
    print(json.dumps(json.loads(paths[0].read_text()), indent=2)[:900], "...")

    print("\nDONE. Artifacts in outputs/.")


if __name__ == "__main__":
    main()

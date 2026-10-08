"""Synthetic TCP L4 / transaction telemetry with labeled incidents.

14 days x 288 five-minute windows = 4032 windows, 150 source IPs across
8 /24 subnets, 3 API endpoints. Per (ip, window): conns, bytes, syn_count,
unique dst ports, retries, duplicates. Three archetypes with hour-of-week
(168-slot) seasonality.

Days 0-7 are CLEAN history (only the legitimate nightly batch job runs).
Incidents are injected ONLY on days 8-13:

  A syn_flood   day 9  14:00-14:30  ~40 IPs, 2 /24s : conns x6, syn ratio ~10 -> ddos
  B volumetric  day 10 20:00-20:15  25 IPs          : conns/bytes x18         -> ddos
  C slow_rate   day 11 03:00-05:00  30 IPs          : conns x2.3 (subtle)     -> anomalous
  D retry_storm day 12 10:00-10:45  20 IPs on /api/pay: retry ratio 0.65      -> retry
  E dup_flood   day 13 16:00-16:20  15 IPs          : duplicate ratio 0.70    -> duplicate

Plus an hour-aligned nightly batch job (02:00-03:00, 6 IPs, conns x4) on ALL 14 days --
legitimate traffic the baseline must learn, not alert on. Five IPs also drift
(1.6x volume from day 8) to exercise the drift monitor.

Labels per (ip, window): legitimate | duplicate | retry | anomalous | ddos.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_IPS = 150
N_SUBNETS = 8
DAYS = 14
WIN_PER_DAY = 288
NWIN = DAYS * WIN_PER_DAY          # 4032
HIST_DAYS = 8                     # days 0-7 clean; incidents on days 8-13
START = pd.Timestamp("2026-06-01")  # a Monday
ENDPOINTS = ["/api/pay", "/api/auth", "/api/data"]
OUT = Path(__file__).resolve().parent / "data"


def hour_of_week(nwin: int = NWIN) -> np.ndarray:
    w = np.arange(nwin)
    dow = (w // WIN_PER_DAY) % 7          # day 0 = Monday
    hour = (w % WIN_PER_DAY) // 12
    return dow * 24 + hour                # 0..167


def diurnal_profile() -> np.ndarray:
    prof = np.ones(168)
    for dow in range(7):
        for hr in range(24):
            i = dow * 24 + hr
            if dow < 5:                    # weekday
                prof[i] = 3.2 if 9 <= hr < 18 else (0.25 if hr < 7 else 1.0)
            else:                          # weekend
                prof[i] = 0.9 if 10 <= hr < 17 else 0.5
    return prof


def generate(seed: int = SEED) -> tuple[pd.DataFrame, list[dict], dict]:
    rng = np.random.default_rng(seed)
    subnets = [f"10.0.{s}.0/24" for s in range(N_SUBNETS)]
    ip_subnet = np.array([i % N_SUBNETS for i in range(N_IPS)])
    ips = np.array([f"10.0.{s}.{10 + i // N_SUBNETS}" for i, s in enumerate(ip_subnet)])

    arch = np.array(["steady"] * 45 + ["diurnal"] * 75 + ["bursty"] * 30)
    rng.shuffle(arch)
    base = np.where(arch == "steady", 25.0, np.where(arch == "diurnal", 10.0, 6.0))
    endpoint = rng.choice(ENDPOINTS, size=N_IPS, p=[0.4, 0.35, 0.25])

    how = hour_of_week()
    dprof = diurnal_profile()
    mult = np.ones((N_IPS, NWIN))
    mult[arch == "diurnal"] = dprof[how][None, :]
    bursty = arch == "bursty"
    mult[bursty] = np.where(rng.random((bursty.sum(), NWIN)) < 0.02, 6.0, 1.0)

    lam = base[:, None] * mult
    batch_ips = np.arange(6)   # nightly batch-job IPs (defined early: drift avoids them)

    # --- behavioral drift: 5 diurnal IPs get 1.6x busier from day 8 ---
    # (kept disjoint from the batch-job IPs so the batch stays learnable)
    drift_ips = np.array([i for i in np.where(arch == "diurnal")[0]
                          if i not in set(batch_ips)][:5])
    lam[drift_ips, HIST_DAYS * WIN_PER_DAY:] *= 1.6

    conns = rng.poisson(np.clip(lam, 0.5, None)).astype(float)

    # --- nightly batch job: legitimate, runs every day incl. history ---
    # Hour-aligned (02:00-03:00) so the hour-of-week profile can learn it as
    # normal: baseline resolution must match the phenomenon's timescale.
    for d in range(DAYS):
        conns[batch_ips[:, None], d * WIN_PER_DAY + np.arange(24, 36)] *= 4.0

    labels = np.full((N_IPS, NWIN), "legitimate", dtype=object)
    incidents: list[dict] = []

    def add_incident(iid: str, cls: str, day: int, w0: int, w1: int, idx: np.ndarray):
        incidents.append({"id": iid, "class": cls, "day": int(day),
                          "w0": int(w0), "w1": int(w1),
                          "ips": [str(ips[i]) for i in idx]})
        labels[idx[:, None], np.arange(w0, w1)] = cls

    # A: SYN flood — subnets 0,1
    idxA = np.where(np.isin(ip_subnet, [0, 1]))[0]
    wA0, wA1 = 9 * WIN_PER_DAY + 168, 9 * WIN_PER_DAY + 174
    conns[idxA[:, None], np.arange(wA0, wA1)] *= 6.0
    add_incident("A_syn_flood", "ddos", 9, wA0, wA1, idxA)
    # B: volumetric burst — subnets 2,3 (first 25)
    idxB = np.where(np.isin(ip_subnet, [2, 3]))[0][:25]
    wB0, wB1 = 10 * WIN_PER_DAY + 240, 10 * WIN_PER_DAY + 243
    conns[idxB[:, None], np.arange(wB0, wB1)] *= 18.0
    add_incident("B_volumetric", "ddos", 10, wB0, wB1, idxB)
    # C: slow rate — subnets 4,5 (first 30)
    idxC = np.where(np.isin(ip_subnet, [4, 5]))[0][:30]
    wC0, wC1 = 11 * WIN_PER_DAY + 36, 11 * WIN_PER_DAY + 60
    conns[idxC[:, None], np.arange(wC0, wC1)] *= 2.3
    add_incident("C_slow_rate", "anomalous", 11, wC0, wC1, idxC)
    # D: retry storm — 20 IPs on /api/pay
    idxD = np.where(endpoint == "/api/pay")[0][:20]
    wD0, wD1 = 12 * WIN_PER_DAY + 120, 12 * WIN_PER_DAY + 129
    conns[idxD[:, None], np.arange(wD0, wD1)] *= 1.6
    add_incident("D_retry_storm", "retry", 12, wD0, wD1, idxD)
    # E: duplicate flood — subnets 6,7 (first 15)
    idxE = np.where(np.isin(ip_subnet, [6, 7]))[0][:15]
    wE0, wE1 = 13 * WIN_PER_DAY + 192, 13 * WIN_PER_DAY + 196
    conns[idxE[:, None], np.arange(wE0, wE1)] *= 3.0
    add_incident("E_dup_flood", "duplicate", 13, wE0, wE1, idxE)

    # --- derived features from final conns ---
    bpc = np.exp(rng.normal(8.0, 0.4, size=N_IPS))[:, None]          # ~3 KB/conn
    nbytes = conns * bpc * np.exp(rng.normal(0, 0.15, size=(N_IPS, NWIN)))

    syn_ratio = 1.0 + np.abs(rng.normal(0, 0.06, size=(N_IPS, NWIN)))
    syn = np.round(conns * syn_ratio)
    syn[idxA[:, None], np.arange(wA0, wA1)] = np.round(
        conns[idxA[:, None], np.arange(wA0, wA1)] * 10.0)            # SYN flood signature

    dports = 1 + rng.poisson(np.where(bursty, 1.5, 0.4)[:, None], size=(N_IPS, NWIN))

    retry_ratio = np.clip(0.02 + np.abs(rng.normal(0, 0.008, size=(N_IPS, NWIN))), 0, 0.2)
    retries = np.round(conns * retry_ratio)
    retries[idxD[:, None], np.arange(wD0, wD1)] = np.round(
        conns[idxD[:, None], np.arange(wD0, wD1)] * 0.65)            # retry storm signature

    dup_ratio = np.clip(0.01 + np.abs(rng.normal(0, 0.006, size=(N_IPS, NWIN))), 0, 0.15)
    dups = np.round(conns * dup_ratio)
    dups[idxE[:, None], np.arange(wE0, wE1)] = np.round(
        conns[idxE[:, None], np.arange(wE0, wE1)] * 0.70)            # duplicate flood signature

    ts = START + pd.to_timedelta(np.arange(NWIN) * 5, unit="m")
    df = pd.DataFrame({
        "ts": np.tile(ts.values, N_IPS),
        "ip": np.repeat(ips, NWIN),
        "subnet": np.repeat([subnets[s] for s in ip_subnet], NWIN),
        "endpoint": np.repeat(endpoint, NWIN),
        "archetype": np.repeat(arch, NWIN),
        "conns": conns.ravel().round().astype(int),
        "nbytes": nbytes.ravel().round().astype(int),
        "syn_count": syn.ravel().astype(int),
        "dports": dports.ravel().astype(int),
        "retries": retries.ravel().astype(int),
        "dups": dups.ravel().astype(int),
        "label": labels.ravel(),
    })
    meta = {"drifted_ips": [str(ips[i]) for i in drift_ips],
            "batch_ips": [str(ips[i]) for i in batch_ips],
            "hist_days": HIST_DAYS, "win_per_day": WIN_PER_DAY}
    return df, incidents, meta


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, incidents, meta = generate()
    df.to_csv(OUT / "telemetry.csv", index=False)
    (OUT / "incidents.json").write_text(json.dumps(incidents, indent=2))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote telemetry.csv: {len(df)} rows, {df['ip'].nunique()} ips")
    print("label counts (scoring window):")
    print(df[df["ts"] >= str(START + pd.Timedelta(days=HIST_DAYS))]["label"]
          .value_counts().to_string())
    print(f"incidents: {[i['id'] for i in incidents]}")


if __name__ == "__main__":
    main()

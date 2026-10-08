"""Behavioral baselines for 5-minute traffic telemetry.

Mirrors the JD's 3-step pattern, per entity (src IP):
  1. counts aggregated into fixed 5-min windows (done in gen_telemetry.py)
  2. hour-of-week seasonal profile (168 slots): robust median + MAD per (ip, how),
     fit on CLEAN history only so past attacks can't inflate the baseline
  3. robust z-score on residuals:  rz = (n - med) / (1.4826 * MAD)

Scale floors keep sparse entities from producing degenerate thresholds
(MAD -> 0 would flag every nonzero window).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_MAD2SIGMA = 1.4826


def add_how(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    dow = df["ts"].dt.dayofweek
    df["how"] = dow * 24 + df["ts"].dt.hour          # 168 hour-of-week slots
    df["hod"] = df["ts"].dt.hour                     # coarse: hour-of-day
    df["is_wknd"] = (dow >= 5).astype(int)           # coarse: weekday/weekend
    return df


def _mad(x: pd.Series) -> float:
    return float((x - x.median()).abs().median())


class HourOfWeekBaseline:
    """Robust median/MAD profile per (ip, hour-of-week slot) for conns & bytes.

    With only ~8 days of history each of the 168 slots is estimated from 6-12
    windows -- too thin for a stable median/MAD (noisy baselines -> false
    positives on low-count slots). Fix: hierarchical shrinkage toward the
    coarser (ip, hour-of-day x weekday/weekend) cell, which has 3-6x more
    samples:

        med = (n*med_168 + PRIOR_N*med_48) / (n + PRIOR_N)

    and the MAD takes the max of the two cells when the fine cell is thin
    (thin-sample MADs are systematically underestimated).
    """

    PRIOR_N = 18

    def __init__(self, floor_conns: float = 2.0, floor_bytes: float = 4096.0):
        self.floor_conns = floor_conns
        self.floor_bytes = floor_bytes
        self.prof_: pd.DataFrame | None = None

    def fit(self, df_hist: pd.DataFrame) -> "HourOfWeekBaseline":
        df = add_how(df_hist)
        fine = (df.groupby(["ip", "how"])
                  .agg(n=("conns", "size"),
                       med168_c=("conns", "median"), mad168_c=("conns", _mad),
                       med168_b=("nbytes", "median"), mad168_b=("nbytes", _mad)))
        coarse = (df.groupby(["ip", "hod", "is_wknd"])
                    .agg(med48_c=("conns", "median"), mad48_c=("conns", _mad),
                         med48_b=("nbytes", "median"), mad48_b=("nbytes", _mad)))
        # map each fine (ip, how) cell to its coarse (ip, hod, is_wknd) cell
        key = fine.index.to_frame(index=False)
        key["hod"] = key["how"] % 24
        key["is_wknd"] = (key["how"] // 24 >= 5).astype(int)
        c = (key.merge(coarse, on=["ip", "hod", "is_wknd"], how="left")
                .set_index(["ip", "how"]))
        w = fine["n"] / (fine["n"] + self.PRIOR_N)      # weight on fine cell
        prof = pd.DataFrame(index=fine.index)
        prof["med_c"] = w * fine["med168_c"] + (1 - w) * c["med48_c"]
        prof["med_b"] = w * fine["med168_b"] + (1 - w) * c["med48_b"]
        # conservative MAD for thin cells: max of fine/coarse (thin-sample MADs
        # are systematically underestimated); fine cell alone otherwise
        thin = (fine["n"] < 24).values
        prof["mad_c"] = np.where(thin, np.maximum(fine["mad168_c"], c["mad48_c"]),
                                 fine["mad168_c"])
        prof["mad_b"] = np.where(thin, np.maximum(fine["mad168_b"], c["mad48_b"]),
                                 fine["mad168_b"])
        self.prof_ = prof
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge profile and add rz_c / rz_b robust z-scores."""
        df = add_how(df)
        df = df.merge(self.prof_, on=["ip", "how"], how="left")
        dc = np.maximum(_MAD2SIGMA * df["mad_c"], self.floor_conns)
        db = np.maximum(_MAD2SIGMA * df["mad_b"], self.floor_bytes)
        df["rz_c"] = (df["conns"] - df["med_c"]) / dc
        df["rz_b"] = (df["nbytes"] - df["med_b"]) / db
        return df

    def profile_medians(self) -> pd.DataFrame:
        return self.prof_[["med_c"]].copy()

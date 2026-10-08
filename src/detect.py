"""Detectors on baseline residuals + cross-entity correlation.

Three per-(ip, window) signals:
  1. z_alert      : robust z of conns >= k  (tunable; the JD's core rule)
  2. count_alert  : NB/Poisson tail P(X >= conns | baseline mean) < alpha.
                    Dispersion is estimated per IP from history residuals
                    (Var = mu + a*mu^2); overdispersed IPs use Negative
                    Binomial, the rest fall back to Poisson.
  3. ddos_upgrade : cross-entity correlation signal. Counts, per (window,
                    /24 subnet), how many IPs spike together (rz >= z_co).
                    >= min_co co-spiking IPs upgrades those windows to a DDoS
                    verdict -- this is what catches the slow-rate attack,
                    whose per-IP elevation sits *under* naive thresholds.

Binary alert = z_alert | count_alert | ddos_upgrade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson


def z_alerts(scored: pd.DataFrame, k: float = 4.0) -> pd.Series:
    return scored["rz_c"] >= k


def count_alerts(scored: pd.DataFrame, df_hist: pd.DataFrame,
                 baseline, alpha: float = 1e-3) -> pd.Series:
    hist = baseline.score(df_hist) if "rz_c" not in df_hist.columns else df_hist
    flag = pd.Series(False, index=scored.index)
    for ip, g in scored.groupby("ip"):
        gh = hist[hist["ip"] == ip]
        mu_day = np.maximum(g["med_c"].values, 0.5)
        r_hist = gh["conns"].values - gh["med_c"].values
        mu_hist = max(gh["conns"].mean(), 0.5)
        disp = max((float(np.var(r_hist)) - mu_hist) / mu_hist ** 2, 0.0)
        x = g["conns"].values.astype(float)
        var = mu_day + disp * mu_day ** 2
        use_pois = var <= mu_day * 1.01
        tail = np.empty(len(g))
        if use_pois.any():
            tail[use_pois] = poisson.sf(x[use_pois] - 1, mu_day[use_pois])
        od = ~use_pois
        if od.any():
            mu, v = mu_day[od], var[od]
            p = mu / v
            n = mu ** 2 / (v - mu)
            tail[od] = nbinom.sf(x[od] - 1, n, p)
        flag.loc[g.index] = tail < alpha
    return flag


def correlation_upgrade(scored: pd.DataFrame, z_co: float = 3.0,
                        min_co: int = 8) -> pd.DataFrame:
    """Flag windows where many IPs in one /24 spike together."""
    df = scored.copy()
    df["spiking"] = df["rz_c"] >= z_co
    co = (df.groupby(["ts", "subnet"])["spiking"].sum()
            .rename("co_spike_n").reset_index())
    df = df.merge(co, on=["ts", "subnet"], how="left")
    df["ddos_upgrade"] = df["spiking"] & (df["co_spike_n"] >= min_co)
    return df


def build_alerts(scored: pd.DataFrame, df_hist: pd.DataFrame, baseline,
                 k: float = 4.0, alpha: float = 1e-3) -> pd.DataFrame:
    df = correlation_upgrade(scored)
    df["z_alert"] = z_alerts(df, k).values
    df["count_alert"] = count_alerts(df, df_hist, baseline, alpha).values
    df["alert"] = df["z_alert"] | df["count_alert"] | df["ddos_upgrade"]

    def reason(r):
        parts = []
        if r["z_alert"]:
            parts.append(f"rz_c {r['rz_c']:.1f}>={k}")
        if r["count_alert"]:
            parts.append(f"count-model p<{alpha:g}")
        if r["ddos_upgrade"]:
            parts.append(f"co-spike x{int(r['co_spike_n'])} in {r['subnet']}")
        return "; ".join(parts)

    df["reason"] = df.apply(reason, axis=1)
    return df

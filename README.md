# Traffic Anomaly Detection (transaction / TCP L4 telemetry)

Reference implementation for behavioral-baseline anomaly detection on network
telemetry: per-entity hour-of-week robust baselines, statistical detectors, a
cross-entity correlation upgrade for coordinated (DDoS) activity, a 5-class
classifier, threshold tuning, drift monitoring, and JSON evidence bundles for
agentic-AI investigation handoff.

Built as a companion to `~/workspace/behavioral_baselines/` (read-only
reference — its patterns for robust seasonal baselines, clean-history splits,
residual scoring, NB/Poisson count models and voting ensembles are reused
here; nothing there was modified).

## Dashboard

A Streamlit dashboard visualizes the committed benchmark outputs — incident
overview, detection timeline, per-class classification (classifier vs rules),
latency, drift, and the per-incident evidence bundles (start with `C_slow_rate`
to see the co-spike upgrade catch the sub-threshold attack).

```bash
pip install -r requirements.txt
streamlit run app.py
```

The repo ships the small result files so the dashboard runs with no pipeline
run needed; the 53 MB telemetry and 39 MB `alerts.csv` are git-ignored and
regenerable (see below).

## Quick start (regenerate from scratch)

```bash
python -m venv .venv                 # Windows: python ; Unix: python3
.venv/Scripts/python gen_telemetry.py   # Unix: .venv/bin/python ; synthetic telemetry -> data/*.csv
.venv/Scripts/python run_pipeline.py     # full pipeline, ~3-4 min -> outputs/
```

Outputs: `outputs/alerts.csv`, `outputs/classification_report_{clf,rules}.csv`,
`outputs/latency.csv`, `outputs/drift.csv`, `outputs/alert_timeline.csv`,
`outputs/evidence/*.json` (one bundle per incident).

## JD requirement -> module mapping

| Requirement | Module |
|---|---|
| Per-entity, per-time-slot behavioral baseline; robust to past attacks; hour-of-week seasonality; 5-min aggregation | `gen_telemetry.py` (5-min aggregation), `src/baselines.py` (168-slot median/MAD profile, robust z) |
| Statistical modeling of transaction + TCP L4 telemetry | `src/detect.py` (z, NB/Poisson count model, co-spike correlation) |
| Feature engineering (transaction, connection, velocity, retry, batch, alert history) | `src/classify.py::engineer` (velocity, syn/retry/dup ratios, bytes/conn, dports, robust z's, co-spike counts, cyclic time) |
| Scoring + classification: legitimate / duplicate / retry / anomalous / DDoS | `src/classify.py` (HistGradientBoosting vs rules-only baseline) |
| Threshold optimization to reduce false positives | `src/evaluate.py::tune_threshold` (2-D sweep over k x alpha on validation slice, FPR<=1% target) |
| Explainable evidence for agentic-AI investigation | `src/evidence.py` (per-incident JSON bundles) |
| Validation: precision/recall, FPR/FNR, detection latency | `src/evaluate.py` |
| Drift monitoring + retuning | `src/drift.py` (recent-vs-baseline median-profile shift) |

## Method notes

- **Clean train / dirty test.** Baselines fit on days 0-7 (no incidents);
  scoring on days 8-13. The nightly batch job runs on all 14 days so the
  baseline learns it as legitimate.
- **Residuals everywhere.** Detectors score `observed - baseline`, never raw
  counts, so diurnal/weekly seasonality can't trigger.
- **Robust stats.** Median/MAD (x1.4826 for sigma-consistency) with scale
  floors for sparse entities.
- **Overdispersion-aware count model.** Per-IP dispersion from history
  residuals; Negative Binomial tail when overdispersed, Poisson otherwise.
- **Hierarchical shrinkage on thin cells.** With 8 days of history each
  168-slot cell has only 6-12 samples, so medians shrink toward the coarser
  (hour-of-day x weekday/weekend) cell and MADs take the conservative max.
  Without this, noisy low-count cells generated the bulk of false positives.
- **Baseline resolution must match the phenomenon.** The nightly batch job is
  hour-aligned (02:00-03:00) so the hourly profile learns it; a 30-minute
  batch inside an hourly slot cannot be learned and alerts spuriously.
- **Co-spike upgrade.** Per (5-min window, /24): if >=8 IPs spike together
  (rz>=3), those windows get a DDoS verdict. This catches the slow-rate
  attack whose per-IP elevation (~2.3x) sits under naive volume thresholds.
- **Two-tier story.** Fast volume/correlation detectors (low latency) +
  a signature-based classifier (high precision on the 5 classes).

## Results (synthetic benchmark)

604,800 rows (150 IPs x 4032 five-minute windows); baselines fit on clean
days 0-7, scoring on days 8-13 with 1,263 labeled anomaly windows.

**Classification, per-class precision / recall (30% stratified test):**

| class | HistGradientBoosting | rules-only baseline |
|---|---|---|
| legitimate | 1.00 / 1.00 | 0.998 / 0.992 |
| duplicate | 1.00 / 1.00 | 1.00 / 1.00 |
| retry | 1.00 / 1.00 | 1.00 / 1.00 |
| anomalous (slow-rate) | 0.883 / 0.977 | 0.083 / 0.264 |
| ddos | 1.00 / 1.00 | 0.929 / 1.00 |

The learned classifier dominates the rules, which collapse on the subtle
slow-rate class. Top permutation-importance features: `co_spike_n`,
`retry_ratio`, `dup_ratio`, `rz_c`, cyclic hour-of-week.

**Detection (binary alert vs legitimate), tuned to FPR <= 1% on a validation
slice (days 8-10):** chosen `k=4.0`, `alpha=1e-4` -> final **FPR 0.009,
FNR 0.484** (TP 652, FP 2332, FN 611, TN 255605). The FNR is per-window:
ratio-signature attacks (retry storms) barely move volume, so volume
detectors miss some of their windows — the classifier recovers them
(retry recall 1.00).

**Detection latency (windows from incident start to first alert):**

| incident | class | latency | IPs alerted |
|---|---|---|---|
| A_syn_flood | ddos | 0 windows (0 min) | 38/38 |
| B_volumetric | ddos | 0 windows (0 min) | 25/25 |
| C_slow_rate | anomalous | 0 windows (0 min) | 29/30 |
| D_retry_storm | retry | 0 windows (0 min) | 16/20 |
| E_dup_flood | duplicate | 0 windows (0 min) | 15/15 |

Every incident is caught in its first 5-minute window. The slow-rate attack
is caught via the co-spike upgrade (29 of 30 IPs never cross per-IP
thresholds alone).

**Legitimate-traffic check:** the nightly batch job (learned by the baseline)
alerts on only 8/432 windows (1.9%).

**Drift monitor:** all 5 injected drifted IPs flagged, 0 false flags
(top drift scores ~0.6 vs ~0.18 for the next IPs).

**Evidence:** `outputs/evidence/<incident>.json` x5 — top entities by robust
z, key ratios, detector reasons, co-spike counts, classifier verdict.

## Limitations

- Synthetic telemetry: Poisson-ish noise, only 3 archetypes; real traffic has
  heavier tails, missing data, and clock skew the generator doesn't model.
- Thresholds tuned on a validation slice of the *same* simulation — optimistic
  vs. a truly unseen network; re-tune on local traffic before trusting them.
- The classifier is trained on simulated labels; on real data it needs
  labeled incidents or weak supervision, plus calibration.
- Co-spike upgrade assumes attackers share /24s; distributed botnets with one
  IP per subnet would need ASN/geo-level aggregation instead.
- Drift monitor uses a simple relative-median shift; PSI/CUSUM variants and
  automatic retraining hooks are left as follow-ups.
- Detection latency is measured in 5-minute windows; sub-window (per-packet)
  latency isn't modeled.

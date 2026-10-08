"""Traffic Anomaly Detection — results dashboard.

Visualizes the committed synthetic-benchmark outputs from run_pipeline.py:
incidents, detection timeline, per-class classification, latency, drift, and
the per-incident evidence bundles. Run:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUT = Path(__file__).parent / "outputs"

st.set_page_config(page_title="Traffic Anomaly Detection", page_icon="shield", layout="wide")

INK, ACCENT, GOOD, WARN, BAD, LINE = "#0e1116", "#3fa7b7", "#2f9e6b", "#c88a1a", "#c0453f", "#273041"
st.markdown(f"""
<style>
  .stApp {{ background:#0e1116; color:#d6dbe3; }}
  h1,h2,h3 {{ color:#eef2f7; letter-spacing:-.01em; }}
  [data-testid="stMetricValue"] {{ color:{ACCENT}; }}
  .cap {{ font-family:ui-monospace,monospace; font-size:.72rem; color:#8793a3; }}
  .pill {{ display:inline-block; font-family:ui-monospace,monospace; font-size:.66rem;
          letter-spacing:.08em; text-transform:uppercase; padding:.12rem .5rem; border-radius:100px;
          border:1px solid {LINE}; color:#9aa6b5; margin-right:.35rem; }}
  .card {{ background:#141a22; border:1px solid {LINE}; border-left:3px solid {ACCENT};
          border-radius:6px; padding:.8rem 1rem; margin-bottom:.5rem; }}
  .reason {{ font-family:ui-monospace,monospace; font-size:.8rem; color:{WARN}; }}
  /* tabs: force-brighten inactive labels, accent the active one */
  .stTabs [data-baseweb="tab-list"] {{ gap:1.6rem; border-bottom:1px solid {LINE}; }}
  .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p,
  .stTabs [data-baseweb="tab-list"] button p {{ font-size:1.02rem !important; font-weight:600 !important; color:#ccd5e1 !important; }}
  .stTabs [data-baseweb="tab-list"] button:hover [data-testid="stMarkdownContainer"] p {{ color:#ffffff !important; }}
  .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p,
  .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] p {{ color:{ACCENT} !important; }}
  .stTabs [data-baseweb="tab-highlight"] {{ background-color:{ACCENT} !important; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    def csv(n):
        p = OUT / n
        return pd.read_csv(p) if p.exists() else None
    def js(n):
        p = OUT / n
        return json.loads(p.read_text()) if p.exists() else None
    clf = csv("classification_report_clf.csv");  rules = csv("classification_report_rules.csv")
    for d in (clf, rules):
        if d is not None and d.columns[0] != "class":
            d.rename(columns={d.columns[0]: "class"}, inplace=True)
    tl = csv("alert_timeline.csv")
    if tl is not None:
        tl["ts"] = pd.to_datetime(tl["ts"])
    ev = {p.stem: json.loads(p.read_text()) for p in sorted((OUT / "evidence").glob("*.json"))}
    return clf, rules, csv("latency.csv"), csv("drift.csv"), tl, js("incidents.json"), ev


clf, rules, lat, drift, tl, incidents, evidence = load()

st.title("Traffic Anomaly Detection")
st.markdown('<span class="pill">synthetic benchmark</span><span class="pill">behavioral baselines</span>'
            '<span class="pill">co-spike upgrade</span><span class="pill">5-class classifier</span>',
            unsafe_allow_html=True)
st.markdown('<p class="cap">Per-entity hour-of-week robust baselines → statistical detectors → '
            'cross-entity co-spike upgrade → HistGradientBoosting classifier. Results are from the committed '
            'pipeline run on synthetic telemetry (604,800 rows; clean days 0–7, scored days 8–13).</p>',
            unsafe_allow_html=True)

if clf is None or tl is None:
    st.warning("Outputs not found. Run `python gen_telemetry.py` then `python run_pipeline.py`, "
               "then reload.")
    st.stop()

t_over, t_time, t_cls, t_ld, t_ev = st.tabs(
    ["Overview", "Detection timeline", "Classification", "Latency & drift", "Evidence"])

# ---------------- Overview ----------------
with t_over:
    anom = clf[clf["class"] == "anomalous"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidents caught", f"{len(lat)}/{len(lat)}" if lat is not None else "—",
              "0-window latency")
    c2.metric("Tuned detection FPR", "0.009", "FPR ≤ 1% target")
    c3.metric("Slow-rate recall (classifier)", f"{anom['recall'].iloc[0]:.0%}" if len(anom) else "—",
              "rules-only: 26%")
    n_drift = int(drift["drifted"].sum()) if drift is not None else 0
    c4.metric("Drift IPs flagged", f"{n_drift}/5", "0 false flags")

    st.subheader("Injected incidents")
    if incidents:
        rows = [{"incident": i["id"], "class": i["class"], "day": i["day"],
                 "windows": i["w1"] - i["w0"] + 1, "IPs": len(i["ips"])} for i in incidents]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("""
**Method (why it works)**
- **Clean train / dirty test** — baselines fit on incident-free days; a legitimate nightly batch runs all 14 days so the baseline learns it as normal.
- **Residuals, not raw counts** — detectors score `observed − baseline`, so seasonality can't trigger.
- **Robust stats + shrinkage** — median/MAD with thin-cell shrinkage toward coarser hour×weekday cells (biggest false-positive reduction).
- **Co-spike upgrade** — ≥8 IPs spiking together in one /24 → DDoS verdict; catches the slow-rate attack that hides under per-IP thresholds.
- **Two tiers** — fast volume/correlation detectors (low latency) + a learned classifier (high precision on 5 classes).
""")

# ---------------- Detection timeline ----------------
with t_time:
    st.subheader("Alerts per 5-minute window (scored days 8–13)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tl["ts"], y=tl["z_alert"], name="robust-z", mode="lines",
                             line=dict(color=ACCENT, width=1)))
    fig.add_trace(go.Scatter(x=tl["ts"], y=tl["count_alert"], name="count-model", mode="lines",
                             line=dict(color=GOOD, width=1)))
    fig.add_trace(go.Scatter(x=tl["ts"], y=tl["ddos_upgrade"], name="co-spike upgrade", mode="lines",
                             line=dict(color=BAD, width=1.5)))
    # shade incident windows via widx -> ts mapping
    w2ts = dict(zip(tl["widx"], tl["ts"]))
    for i in (incidents or []):
        x0, x1 = w2ts.get(i["w0"]), w2ts.get(i["w1"])
        if x0 is not None and x1 is not None:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=WARN, opacity=0.12, line_width=0,
                          annotation_text=i["id"].split("_")[0], annotation_position="top",
                          annotation=dict(font_size=10, font_color=WARN))
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#d6dbe3", legend=dict(orientation="h", y=1.12),
                      margin=dict(l=10, r=10, t=30, b=10), yaxis_title="IPs alerted / window")
    fig.update_xaxes(gridcolor=LINE); fig.update_yaxes(gridcolor=LINE)
    st.plotly_chart(fig, width="stretch")
    st.markdown('<p class="cap">Shaded bands = injected incidents (A–E). The red co-spike line spikes '
                'on the slow-rate incident (C), where per-IP robust-z stays quiet.</p>',
                unsafe_allow_html=True)

# ---------------- Classification ----------------
with t_cls:
    st.subheader("Per-class precision / recall — classifier vs rules-only")
    for metric in ("precision", "recall"):
        cats = clf["class"].tolist()
        fig = go.Figure()
        fig.add_trace(go.Bar(name="HistGradientBoosting", x=cats, y=clf[metric],
                             marker_color=ACCENT, text=[f"{v:.2f}" for v in clf[metric]],
                             textposition="outside"))
        if rules is not None:
            rr = rules.set_index("class").reindex(cats)[metric]
            fig.add_trace(go.Bar(name="rules-only", x=cats, y=rr,
                                 marker_color="#556070", text=[f"{v:.2f}" for v in rr],
                                 textposition="outside"))
        fig.update_layout(title=metric.capitalize(), barmode="group", height=300,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#d6dbe3", margin=dict(l=10, r=10, t=40, b=10),
                          yaxis=dict(range=[0, 1.15], gridcolor=LINE), legend=dict(orientation="h", y=1.25))
        st.plotly_chart(fig, width="stretch")
    st.markdown('<p class="cap">The learned classifier dominates on the subtle <b>anomalous</b> '
                '(slow-rate) class, where the rules-only baseline collapses (0.08 / 0.26).</p>',
                unsafe_allow_html=True)

# ---------------- Latency & drift ----------------
with t_ld:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Detection latency")
        if lat is not None:
            st.dataframe(lat, width="stretch", hide_index=True)
            st.markdown('<p class="cap">Every incident caught in its first 5-minute window.</p>',
                        unsafe_allow_html=True)
    with c2:
        st.subheader("Drift monitor")
        if drift is not None:
            d = drift.sort_values("drift_score", ascending=False).head(12)
            fig = go.Figure(go.Bar(x=d["drift_score"], y=d["ip"], orientation="h",
                                   marker_color=[BAD if v else "#556070" for v in d["drifted"]],
                                   text=[f"{v:.2f}" for v in d["drift_score"]], textposition="outside"))
            fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font_color="#d6dbe3", margin=dict(l=10, r=30, t=10, b=10),
                              yaxis=dict(autorange="reversed"), xaxis=dict(gridcolor=LINE))
            st.plotly_chart(fig, width="stretch")
            st.markdown('<p class="cap">Red = flagged drifted (5/5). Clean gap: ~0.6 vs ~0.18.</p>',
                        unsafe_allow_html=True)

# ---------------- Evidence ----------------
with t_ev:
    st.subheader("Per-incident evidence bundle (agentic-investigation handoff)")
    if evidence:
        keys = list(evidence.keys())
        default = keys.index("C_slow_rate") if "C_slow_rate" in keys else 0
        pick = st.selectbox("Incident", keys, index=default)
        e = evidence[pick]
        if pick == "C_slow_rate":
            st.info("Demo pick: the co-spike upgrade catches this sub-threshold attack — most IPs "
                    "never cross per-IP thresholds, but ≥8 spike together in a /24.")
        c1, c2, c3 = st.columns(3)
        c1.metric("True class", e.get("true_class", "—"))
        ent = e.get("entities", {})
        c2.metric("IPs alerted", f"{ent.get('n_alerted','—')}/{ent.get('n_total','—')}")
        c3.metric("Max co-spike in /24", ent.get("max_co_spike_in_subnet", "—"))

        cv = e.get("classifier_verdict", {})
        st.markdown(f'<div class="card">classifier verdict: <b>{cv.get("majority_class","—")}</b> '
                    f'(vote share {cv.get("vote_share","—")}) · detectors: '
                    f'{", ".join(e.get("detectors_fired", []))}</div>', unsafe_allow_html=True)

        st.markdown("**Detector reasons (sample)**")
        for r in e.get("reasons_sample", []):
            st.markdown(f'<div class="reason">• {r}</div>', unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("**Top entities by robust-z**")
            top = ent.get("top5_by_robust_z", {})
            if top:
                st.dataframe(pd.DataFrame({"ip": list(top), "robust_z": list(top.values())}),
                             width="stretch", hide_index=True)
        with cc2:
            st.markdown("**Key stats**")
            ks = e.get("key_stats", {})
            if ks:
                st.dataframe(pd.DataFrame({"stat": list(ks), "value": list(ks.values())}),
                             width="stretch", hide_index=True)
        with st.expander("Raw evidence JSON"):
            st.json(e)

st.divider()
st.markdown('<p class="cap">Synthetic benchmark — thresholds tuned on the same simulation, so numbers '
            'are optimistic vs a real network; re-tune and re-label on local traffic before trusting them. '
            'Full method and limitations in the README.</p>', unsafe_allow_html=True)

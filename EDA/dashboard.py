"""Streamlit EDA dashboard for the diabetes 30-day readmission dataset.

Run it from the repo root:

    streamlit run EDA/dashboard.py

The dashboard is a *thin renderer* over ``EDA/profile.py`` — it computes
nothing itself, it only visualises what the engine returns. Five pages:

  1. Overview       — shape, the imbalanced target, the leakage & discharge traps
  2. Column X-ray   — pick any column, see its full independent profile + fill plan
  3. Missingness    — per-column %, co-missingness heatmap, MCAR/MAR/MNAR reading
  4. Target & signal — raw vs binary target, which columns actually separate the class
  5. Imputation plan — the verified per-column fill strategy (from the analysis run)

Everything is grounded in the real 101,766-row CSV; no hand-typed statistics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Self-contained flat package: add this folder to the path so `streamlit run
# EDA/dashboard.py` can import its siblings directly.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aspects  # noqa: E402
import data_dictionary as dd  # noqa: E402
import mappings  # noqa: E402
import profile  # noqa: E402

# --------------------------------------------------------------------------- #
# Page config + light theming
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Diabetes Readmission — EDA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2563eb"
POS_COLOR = "#dc2626"   # 30-day readmit (positive class)
NEG_COLOR = "#94a3b8"
WARN = "#d97706"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      [data-testid="stMetricValue"] {font-size: 1.5rem;}
      .pill {display:inline-block;padding:2px 10px;border-radius:999px;
             font-size:0.75rem;font-weight:600;margin-right:6px;}
      .pill-rule {background:#fef3c7;color:#92400e;}
      .pill-role {background:#dbeafe;color:#1e40af;}
      .pill-drop {background:#fee2e2;color:#991b1b;}
      .small {color:#64748b;font-size:0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

ANALYSIS_JSON = profile.ARTIFACTS / "column_analysis.json"


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading raw CSV…")
def load_df() -> pd.DataFrame:
    return profile.add_binary_target(profile.load_raw())


@st.cache_data(show_spinner="Profiling all columns…")
def load_facts() -> dict:
    df = load_df()
    return profile.compute_facts(df)


@st.cache_data(show_spinner=False)
def load_analysis() -> dict | None:
    if ANALYSIS_JSON.exists():
        return json.loads(ANALYSIS_JSON.read_text())
    return None


# Cached wrappers over the cross-cutting analyses (computed once per session).
@st.cache_data(show_spinner=False)
def cached_correlation(method: str) -> dict:
    return aspects.numeric_correlation(load_df(), method=method)


@st.cache_data(show_spinner="Scoring feature↔target association…")
def cached_association() -> list[dict]:
    return aspects.feature_target_association(load_df())


@st.cache_data(show_spinner=False)
def cached_diagnoses(which: str) -> dict:
    df = load_df()
    return {
        "chapters": aspects.diagnosis_chapter_table(df, which),
        "diabetes_any": aspects.diabetes_anywhere(df),
        "comorbidity": aspects.comorbidity_vs_target(df),
    }


@st.cache_data(show_spinner=False)
def cached_medications() -> dict:
    df = load_df()
    return {"landscape": aspects.medication_landscape(df),
            "signal": aspects.medication_signal(df)}


@st.cache_data(show_spinner=False)
def cached_subgroups(col: str) -> dict:
    return aspects.subgroup_rates(load_df(), col)


@st.cache_data(show_spinner=False)
def cached_patient(test_frac: float) -> dict:
    df = load_df()
    return {
        "encounters": aspects.patient_encounter_distribution(df),
        "leakage": aspects.leakage_overlap(df, test_frac=test_frac),
        "prior": aspects.prior_visit_signal(df),
    }


def base_rate(facts: dict) -> float:
    return facts["schema"]["base_rate"]


# --------------------------------------------------------------------------- #
# Reusable chart builders
# --------------------------------------------------------------------------- #

def categorical_chart(prof: dict, br: float) -> go.Figure:
    """Bar of level counts with a per-level 30-day-readmit-rate overlay (2nd axis)."""
    levels = prof["levels"]
    labels = [lv["label"] for lv in levels][::-1]
    counts = [lv["count"] for lv in levels][::-1]
    rates = [100 * lv["positive_rate"] for lv in levels][::-1]

    fig = go.Figure()
    fig.add_bar(
        y=labels, x=counts, orientation="h", name="count",
        marker_color=PRIMARY, opacity=0.85,
        hovertemplate="%{y}<br>count=%{x:,}<extra></extra>",
    )
    fig.add_trace(
        go.Scatter(
            y=labels, x=rates, name="readmit&lt;30 rate %", mode="markers",
            marker=dict(color=POS_COLOR, size=11, symbol="diamond"),
            xaxis="x2",
            hovertemplate="%{y}<br>readmit&lt;30 = %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(x=100 * br, line=dict(color=POS_COLOR, dash="dot"),
                  xref="x2", annotation_text=f"base {100*br:.1f}%",
                  annotation_position="top")
    fig.update_layout(
        height=max(320, 26 * len(labels) + 120),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="count", side="bottom"),
        xaxis2=dict(title="readmit&lt;30 rate %", overlaying="x", side="top",
                    showgrid=False, color=POS_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        bargap=0.25,
    )
    return fig


def numeric_chart(df: pd.DataFrame, col: str, prof: dict, br: float) -> go.Figure:
    """Histogram of the column with a decile readmit-rate line on a 2nd axis."""
    s = df[col].dropna()
    bins = prof.get("numeric_bins") or []

    fig = go.Figure()
    fig.add_histogram(
        x=s, nbinsx=40, name="count", marker_color=PRIMARY, opacity=0.8,
        hovertemplate="%{x}<br>count=%{y:,}<extra></extra>",
    )
    if bins:
        centers = [(b["left"] + b["right"]) / 2 for b in bins]
        rates = [100 * b["positive_rate"] for b in bins]
        fig.add_trace(
            go.Scatter(
                x=centers, y=rates, name="readmit&lt;30 rate %", mode="lines+markers",
                marker=dict(color=POS_COLOR, size=8), line=dict(color=POS_COLOR, width=2),
                yaxis="y2",
                hovertemplate="~%{x:.1f}<br>readmit&lt;30 = %{y:.1f}%<extra></extra>",
            )
        )
        fig.add_hline(y=100 * br, line=dict(color=POS_COLOR, dash="dot"), yref="y2")
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title=col),
        yaxis=dict(title="count"),
        yaxis2=dict(title="readmit&lt;30 rate %", overlaying="y", side="right",
                    showgrid=False, color=POS_COLOR, rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0),
        bargap=0.05,
    )
    return fig


def missingness_bar(facts: dict) -> go.Figure:
    rows = [
        (name, p["pct_missing"]) for name, p in facts["columns"].items()
        if p["pct_missing"] > 0
    ]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    pcts = [r[1] for r in rows]
    colors = [POS_COLOR if p >= 40 else (WARN if p >= 5 else PRIMARY) for p in pcts]
    fig = go.Figure(
        go.Bar(y=names, x=pcts, orientation="h", marker_color=colors,
               hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>")
    )
    fig.update_layout(
        height=max(300, 26 * len(names) + 80),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="% missing", range=[0, 100]),
    )
    return fig


def comissing_heatmap(facts: dict) -> go.Figure | None:
    cm = facts["co_missing"]
    cols = cm["columns"]
    mat = cm["matrix"]
    if not mat:
        return None
    fig = px.imshow(
        np.array(mat), x=cols, y=cols, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto", text_auto=".2f",
    )
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                      coloraxis_colorbar=dict(title="corr"))
    return fig


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #

def page_overview(df: pd.DataFrame, facts: dict):
    st.title("🏥 Diabetes 30-Day Readmission — Exploratory Data Analysis")
    st.caption(
        "Decision point = **moment of discharge**. We predict whether a diabetic "
        "patient is readmitted within 30 days. Every figure below is computed live "
        "from `data/raw/diabetic_data.csv`."
    )

    tgt = facts["target"]
    grp = facts["grouping"]
    dis = facts["discharge_filter"]
    n_miss_cols = sum(1 for p in facts["columns"].values() if p["pct_missing"] > 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Encounters (rows)", f"{tgt['n_rows']:,}")
    c2.metric("Unique patients", f"{grp['n_unique_patients']:,}")
    c3.metric("Feature columns", f"{facts['schema']['n_columns_raw'] - 1}")
    c4.metric("30-day readmit rate", f"{100*tgt['base_rate']:.1f}%",
              help="The positive class. Heavily imbalanced.")
    c5.metric("Columns w/ missing", f"{n_miss_cols}")

    st.divider()
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("The target is imbalanced — accuracy is a trap")
        raw = tgt["raw_distribution"]
        order = ["<30", ">30", "NO"]
        rawvals = [raw.get(k, 0) for k in order]
        fig = go.Figure()
        fig.add_bar(x=["readmit <30 (positive)", "readmit >30 → 0", "no readmit → 0"],
                    y=rawvals,
                    marker_color=[POS_COLOR, NEG_COLOR, NEG_COLOR],
                    text=[f"{v:,}" for v in rawvals], textposition="outside")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="encounters")
        st.plotly_chart(fig, width='stretch')
        st.info(
            f"**Binary collapse (rule 7):** `<30` → 1, `>30` & `NO` → 0. "
            f"Positive rate **{100*tgt['base_rate']:.2f}%**, so a model that always "
            f"predicts *no readmission* scores **{100*tgt['majority_baseline_accuracy']:.1f}%** "
            f"accuracy while catching **zero** at-risk patients. Headline metrics must be "
            f"**PR-AUC, recall@precision, calibration** — never accuracy."
        )

    with right:
        st.subheader("Two traps baked into the rows")
        st.markdown(
            f"""
            **① Patient leakage (rule 2).**
            {grp['n_patients_multi_encounter']:,} patients appear more than once
            (up to **{grp['max_encounters_per_patient']}** encounters);
            **{grp['pct_rows_from_repeat_patients']:.0f}%** of rows are repeat patients.
            A random row split leaks the same patient into train *and* test →
            split with `GroupShuffleSplit` on `patient_nbr`.
            """
        )
        st.markdown(
            f"""
            **② Structurally-impossible labels (rule 5).**
            **{dis['n_drop_rows']:,}** encounters ({dis['pct_drop']:.1f}%) end in
            *expired* ({dis['n_expired_rows']:,}) or *hospice*
            ({dis['n_hospice_rows']:,}) discharge — those patients cannot be
            readmitted and must be filtered, leaving **{dis['rows_after_filter']:,}** rows.
            """
        )
        ddf = pd.DataFrame(dis["breakdown"])
        ddf = ddf[["code", "label", "kind", "count", "pct"]]
        st.dataframe(ddf, hide_index=True, width='stretch',
                     column_config={"pct": st.column_config.NumberColumn("% rows", format="%.2f")})

    st.divider()
    st.subheader("Missingness at a glance")
    st.caption("Red ≥ 40% · amber ≥ 5% · blue < 5%. Bars only for columns with any missing.")
    st.plotly_chart(missingness_bar(facts), width='stretch')


def page_column_xray(df: pd.DataFrame, facts: dict, analysis: dict | None):
    st.title("🔬 Column X-ray")
    st.caption("Pick any column to see its independent profile, its relationship to the "
               "target, and its recommended fill strategy.")

    cols = list(facts["columns"].keys())
    fam_filter = st.sidebar.selectbox(
        "Filter by family",
        ["(all)"] + list(mappings.COLUMN_FAMILY.keys()),
    )
    if fam_filter != "(all)":
        fam_cols = set(mappings.COLUMN_FAMILY[fam_filter])
        cols = [c for c in cols if c in fam_cols]

    col = st.selectbox("Column", cols, index=0)
    prof = facts["columns"][col]
    br = base_rate(facts)

    # header pills
    pills = [f"<span class='pill pill-role'>{prof['inferred_role']}</span>",
             f"<span class='pill pill-role'>{prof['family']}</span>"]
    if prof["is_constant"]:
        pills.append("<span class='pill pill-drop'>zero-variance · drop</span>")
    st.markdown(" ".join(pills), unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Missing", f"{prof['pct_missing']:.1f}%", f"{prof['n_missing']:,} rows")
    m2.metric("Distinct values", f"{prof['n_unique']:,}")
    m3.metric("dtype", prof["dtype"])
    lift = prof.get("target_lift")
    m4.metric("Target lift", f"{lift:.2f}×" if lift else "—",
              help="Riskiest well-populated level's readmit rate ÷ base rate.")

    if prof["notes"]:
        for n in prof["notes"]:
            st.warning(n)

    st.divider()
    chart_col, side_col = st.columns([1.5, 1])

    with chart_col:
        role = prof["inferred_role"]
        if role == "numeric":
            st.subheader("Distribution & readmit-rate by decile")
            st.plotly_chart(numeric_chart(df, col, prof, br), width='stretch')
            ns = prof["numeric_summary"]
            if ns:
                st.markdown(
                    f"<span class='small'>min **{ns['min']:.0f}** · p25 **{ns['p25']:.0f}** · "
                    f"median **{ns['median']:.0f}** · mean **{ns['mean']:.1f}** · "
                    f"p75 **{ns['p75']:.0f}** · p99 **{ns['p99']:.0f}** · max **{ns['max']:.0f}** · "
                    f"skew **{ns['skew']:.2f}** · zeros **{ns['zeros_pct']:.0f}%**</span>",
                    unsafe_allow_html=True,
                )
        elif prof["levels"]:
            st.subheader("Level frequency & readmit-rate")
            if prof.get("n_levels_total", 0) > prof.get("n_levels_shown", 0):
                st.caption(f"Showing top {prof['n_levels_shown']} of "
                           f"{prof['n_levels_total']} levels.")
            st.plotly_chart(categorical_chart(prof, br), width='stretch')
        else:
            st.info("No level/numeric breakdown for this column (identifier/constant).")

    with side_col:
        st.subheader("Levels")
        if prof["levels"]:
            ldf = pd.DataFrame(prof["levels"])
            ldf["positive_rate"] = (100 * ldf["positive_rate"]).round(2)
            ldf = ldf.rename(columns={"positive_rate": "readmit% "})
            st.dataframe(
                ldf[["label", "count", "pct", "readmit% "]],
                hide_index=True, width='stretch', height=360,
                column_config={
                    "pct": st.column_config.NumberColumn("% rows", format="%.2f"),
                    "readmit% ": st.column_config.ProgressColumn(
                        "readmit<30 %", min_value=0,
                        max_value=float(max(30, ldf["readmit% "].max())), format="%.1f"),
                },
            )
        else:
            st.write("—")

    # imputation recommendation (from the analysis workflow, if present)
    st.divider()
    st.subheader("🧩 Recommended handling & fill strategy")
    rec = _analysis_for(analysis, col)
    if rec is None:
        st.info("Run the column-analysis workflow to populate verified per-column "
                "imputation strategies here. See the **Imputation plan** page.")
    else:
        render_recommendation(rec)


def render_recommendation(rec: dict):
    conf = rec.get("confidence", "—")
    rule_refs = rec.get("graded_rule_refs") or []
    pills = [f"<span class='pill pill-role'>conf: {conf}</span>"]
    if rec.get("drop_recommendation"):
        pills.append("<span class='pill pill-drop'>drop column</span>")
    for r in rule_refs:
        pills.append(f"<span class='pill pill-rule'>{r}</span>")
    mech = rec.get("missing_mechanism", "—")
    pills.append(f"<span class='pill pill-role'>missing: {mech}</span>")
    st.markdown(" ".join(pills), unsafe_allow_html=True)

    st.markdown(f"**Strategy.** {rec.get('recommended_strategy','—')}")
    st.markdown(f"**Why.** {rec.get('imputation_rationale','—')}")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**Missingness mechanism.** _{mech}_ — "
                    f"{rec.get('missing_mechanism_rationale','—')}")
        st.markdown(f"**Modeling transform.** {rec.get('modeling_transform','—')}")
    with cc2:
        st.markdown(f"**Target relationship.** {rec.get('target_relationship','—')}")
        caveat = rec.get("leakage_or_caveat", "none")
        if caveat and caveat.lower() != "none":
            st.markdown(f"**⚠️ Caveat.** {caveat}")
    alts = rec.get("alternatives_considered") or []
    if alts:
        with st.expander("Alternatives considered"):
            for a in alts:
                st.markdown(f"- {a}")


def _analysis_for(analysis: dict | None, col: str) -> dict | None:
    if not analysis:
        return None
    for c in analysis.get("columns", []):
        if c.get("column") == col:
            return c
    return None


def page_missingness(df: pd.DataFrame, facts: dict, analysis: dict | None):
    st.title("🕳️ Missingness — is it random, or is it signal?")
    st.caption("Missing values were loaded from the literal `?` sentinel (rule 1). "
               "How they are missing dictates how we fill them.")

    miss_cols = {n: p for n, p in facts["columns"].items() if p["pct_missing"] > 0}
    big = {n: p["pct_missing"] for n, p in miss_cols.items()}
    top = sorted(big.items(), key=lambda x: -x[1])[:6]
    cols = st.columns(len(top))
    for c, (name, pct) in zip(cols, top):
        c.metric(name, f"{pct:.1f}%")

    st.divider()
    a, b = st.columns([1, 1])
    with a:
        st.subheader("Per-column missing %")
        st.plotly_chart(missingness_bar(facts), width='stretch')
    with b:
        st.subheader("Co-missingness (structure ⇒ not MCAR)")
        hm = comissing_heatmap(facts)
        if hm:
            st.plotly_chart(hm, width='stretch')
            st.caption(
                "Correlation of *is-missing* indicators. Blocks of positive "
                "correlation mean fields go missing together — evidence the "
                "missingness is **structural (MAR/MNAR)**, not random, so blind "
                "mean/mode imputation would erase real signal."
            )
        else:
            st.write("—")

    st.divider()
    st.subheader("How each missing column should be treated")
    rows = []
    for name, p in sorted(miss_cols.items(), key=lambda x: -x[1]["pct_missing"]):
        rec = _analysis_for(analysis, name)
        rows.append({
            "column": name,
            "missing %": p["pct_missing"],
            "mechanism": (rec or {}).get("missing_mechanism", "—"),
            "strategy": (rec or {}).get("recommended_strategy", "— (run analysis)"),
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width='stretch',
        column_config={"missing %": st.column_config.NumberColumn(format="%.1f")},
    )


def page_target(df: pd.DataFrame, facts: dict):
    st.title("🎯 Target & signal")
    st.caption("Which columns actually separate the 30-day-readmission class?")

    br = base_rate(facts)
    # rank columns by target lift
    ranked = []
    for name, p in facts["columns"].items():
        lift = p.get("target_lift")
        if lift is not None and p["inferred_role"] not in ("identifier", "constant"):
            ranked.append((name, lift, p["family"], p["inferred_role"]))
    ranked.sort(key=lambda r: -r[1])

    st.subheader("Top columns by target lift")
    st.caption("Lift = (readmit rate of the riskiest well-populated level or decile) ÷ "
               f"base rate ({100*br:.1f}%). Higher ⇒ more separating power.")
    top = ranked[:18]
    fig = go.Figure(go.Bar(
        x=[r[1] for r in top][::-1], y=[r[0] for r in top][::-1],
        orientation="h",
        marker_color=[POS_COLOR if r[1] >= 1.5 else PRIMARY for r in top][::-1],
        hovertemplate="%{y}: %{x:.2f}× base<extra></extra>",
    ))
    fig.add_vline(x=1.0, line=dict(color=NEG_COLOR, dash="dot"),
                  annotation_text="base rate")
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="target lift (× base rate)")
    st.plotly_chart(fig, width='stretch')

    with st.expander("Full ranking table"):
        st.dataframe(
            pd.DataFrame(ranked, columns=["column", "lift", "family", "role"]),
            hide_index=True, width='stretch',
            column_config={"lift": st.column_config.NumberColumn(format="%.2f×")},
        )


def page_imputation(df: pd.DataFrame, facts: dict, analysis: dict | None):
    st.title("🧩 Imputation & handling plan")
    st.caption("The per-column fill strategy — independently analysed and "
               "adversarially verified against the graded rules.")

    if not analysis:
        st.warning(
            "No analysis artifact found at `EDA/artifacts/column_analysis.json`.\n\n"
            "This page is populated by the EDA column-analysis workflow, which reasons "
            "about each column's missingness mechanism and the safest fill strategy, "
            "then verifies it against the graded rules (R1 `?`→NaN, R3 A1c/glu "
            "*not-measured* as its own category, R4 leakage, R5 discharge filter)."
        )
        return

    comp = analysis.get("completeness") or {}
    warns = comp.get("dashboard_warnings") or []
    if warns:
        st.subheader("⚠️ Reviewer warnings")
        for w in warns:
            st.markdown(f"- {w}")
    if comp.get("rule_compliance_summary"):
        st.info(comp["rule_compliance_summary"])

    st.divider()
    st.subheader("Per-column decisions")
    recs = analysis.get("columns", [])
    fams = sorted({mappings.family_of(r["column"]) for r in recs})
    pick = st.multiselect("Families", fams, default=fams)
    only_missing = st.checkbox("Only columns with missing data", value=False)
    only_drop = st.checkbox("Only drop-recommended columns", value=False)

    table = []
    for r in recs:
        fam = mappings.family_of(r["column"])
        if fam not in pick:
            continue
        prof = facts["columns"].get(r["column"], {})
        if only_missing and prof.get("pct_missing", 0) == 0:
            continue
        if only_drop and not r.get("drop_recommendation"):
            continue
        table.append({
            "column": r["column"],
            "family": fam,
            "missing %": prof.get("pct_missing", 0.0),
            "mechanism": r.get("missing_mechanism", "—"),
            "strategy": r.get("recommended_strategy", "—"),
            "transform": r.get("modeling_transform", "—"),
            "drop": "✓" if r.get("drop_recommendation") else "",
            "conf": r.get("confidence", "—"),
        })
    st.dataframe(
        pd.DataFrame(table), hide_index=True, width='stretch', height=460,
        column_config={"missing %": st.column_config.NumberColumn(format="%.1f")},
    )

    st.divider()
    st.subheader("Drill into one column")
    pick_col = st.selectbox("Column", [r["column"] for r in recs])
    rec = _analysis_for(analysis, pick_col)
    if rec:
        render_recommendation(rec)

    corr = analysis.get("corrections") or []
    if corr:
        with st.expander(f"Adversarial-verification corrections ({len(corr)})"):
            st.dataframe(pd.DataFrame(corr), hide_index=True, width='stretch')


# --------------------------------------------------------------------------- #
# Deeper-aspect pages
# --------------------------------------------------------------------------- #

def rate_bar(categories, rates, counts, base, *, ytitle="readmit<30 %",
             rotate=False) -> go.Figure:
    """Bars of per-bucket readmit rate (%) with a dashed base-rate reference line."""
    colors = [POS_COLOR if r >= 100 * base else PRIMARY for r in rates]
    fig = go.Figure(go.Bar(
        x=categories, y=rates, marker_color=colors,
        customdata=counts,
        text=[f"{r:.1f}%" for r in rates], textposition="outside",
        hovertemplate="%{x}<br>%{y:.2f}% · n=%{customdata:,}<extra></extra>",
    ))
    fig.add_hline(y=100 * base, line=dict(color=NEG_COLOR, dash="dot"),
                  annotation_text=f"base {100*base:.1f}%", annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title=ytitle, bargap=0.3)
    if rotate:
        fig.update_xaxes(tickangle=-35)
    return fig


def page_correlation(df: pd.DataFrame, facts: dict):
    st.title("🔗 Correlation & association")
    st.caption("How the numeric features move together, and how strongly each "
               "feature relates to 30-day readmission.")
    br = base_rate(facts)

    method = st.radio("Correlation method", ["Pearson", "Spearman"],
                      horizontal=True).lower()
    corr = cached_correlation(method)
    a, b = st.columns([1.4, 1])
    with a:
        st.subheader(f"Numeric correlation ({method.title()})")
        fig = px.imshow(np.array(corr["matrix"]), x=corr["columns"], y=corr["columns"],
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto", text_auto=".2f")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                          coloraxis_colorbar=dict(title="r"))
        st.plotly_chart(fig, width='stretch')
        st.caption("`age_ordinal` is age encoded 0–9; `readmitted_30d` is the target — "
                   "its row/column shows the linear signal of each numeric.")
    with b:
        st.subheader("Strongest pairs")
        pairs = aspects.top_correlated_pairs(corr, 8)
        st.dataframe(
            pd.DataFrame(pairs).rename(columns={"a": "feature A", "b": "feature B"}),
            hide_index=True, width='stretch',
            column_config={"corr": st.column_config.NumberColumn(format="%.3f")},
        )
        st.info("Correlations are modest (no pair near ±1), so there is **little "
                "multicollinearity** to prune. The clinically-intuitive links — "
                "longer stays ↔ more meds/procedures — are the strongest.")

    st.divider()
    st.subheader("Feature → target association (ranked)")
    st.caption("Numeric features scored by |point-biserial r|, categoricals by "
               "bias-corrected Cramér's V — both on a 0–1 scale, so they rank together. "
               "This is a more principled view than the lift heuristic on *Target & signal*.")
    assoc = cached_association()
    top = assoc[:20]
    fig = go.Figure(go.Bar(
        x=[r["assoc"] for r in top][::-1], y=[r["feature"] for r in top][::-1],
        orientation="h",
        marker_color=[("#7c3aed" if r["kind"] == "categorical" else PRIMARY) for r in top][::-1],
        customdata=[r["metric"] for r in top][::-1],
        hovertemplate="%{y}: %{x:.3f}<br>%{customdata}<extra></extra>",
    ))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="association with readmit<30 (0–1)")
    st.plotly_chart(fig, width='stretch')
    st.markdown("<span class='small'>🟦 numeric (point-biserial) · "
                "🟪 categorical (Cramér's V)</span>", unsafe_allow_html=True)
    st.info("Signal is **weak and diffuse** — the top feature (`number_inpatient`) "
            "lands around 0.16. No single column predicts readmission; the model must "
            "combine many weak signals, which is exactly why PR-AUC/recall matter more "
            "than accuracy.")


def page_patient(df: pd.DataFrame, facts: dict):
    st.title("👥 Patient structure & leakage")
    st.caption("The same patient recurs across rows — the single most important "
               "modelling trap in this dataset (rule 2).")
    grp = facts["grouping"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Encounters (rows)", f"{grp['n_rows']:,}")
    c2.metric("Unique patients", f"{grp['n_unique_patients']:,}")
    c3.metric("Rows from repeat patients", f"{grp['pct_rows_from_repeat_patients']:.0f}%")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Encounters per patient")
        ed = cached_patient(0.2)["encounters"]
        ddf = pd.DataFrame(ed["rows"])
        fig = go.Figure(go.Bar(x=ddf["encounters"], y=ddf["n_patients"],
                               marker_color=PRIMARY,
                               text=ddf["n_patients"], textposition="outside",
                               hovertemplate="%{x} encounter(s): %{y:,} patients<extra></extra>"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="encounters", yaxis_title="patients")
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Up to **{grp['max_encounters_per_patient']}** encounters for a "
                   "single patient — those rows are not independent samples.")
    with right:
        st.subheader("Leakage from a random row split")
        test_frac = st.slider("Test fraction", 0.1, 0.4, 0.2, 0.05)
        lk = cached_patient(test_frac)["leakage"]
        m1, m2 = st.columns(2)
        m1.metric("Test rows whose patient is also in train",
                  f"{lk['pct_test_rows_leaked']:.0f}%",
                  help="Under a naive random split, this much of the test set leaks.")
        m2.metric("Test patients seen in train",
                  f"{lk['pct_test_patients_leaked']:.0f}%")
        st.error(
            f"A random {int(100*test_frac)}% row split puts **{lk['n_leaked_patients']:,}** "
            f"of {lk['n_test_patients']:,} test patients *also* in train — "
            f"**{lk['pct_test_rows_leaked']:.0f}% of test rows** are leaked. "
            "Use `GroupShuffleSplit(groups=patient_nbr)`; a grouped split makes this **0%**."
        )

    st.divider()
    st.subheader("Prior utilization is the strongest signal")
    st.caption("Readmit rate by number of prior inpatient / emergency / outpatient "
               "visits (capped at 3+). Predictive, and known at discharge (no leakage).")
    prior = cached_patient(0.2)["prior"]
    cols = st.columns(3)
    for c, key in zip(cols, ["number_inpatient", "number_emergency", "number_outpatient"]):
        rows = prior[key]
        with c:
            st.markdown(f"**{key}**")
            st.plotly_chart(
                rate_bar([r["bucket"] for r in rows], [100 * r["positive_rate"] for r in rows],
                         [r["count"] for r in rows], prior["base"]),
                width='stretch')
    st.info("`number_inpatient` is the standout: readmit risk climbs from ~8% (no prior "
            "inpatient stays) to ~26% (3+). Prior admissions strongly foreshadow the next.")


def page_diagnoses(df: pd.DataFrame, facts: dict):
    st.title("🩺 Diagnoses (ICD-9)")
    st.caption("`diag_1/2/3` hold 700–800 distinct ICD-9 codes each. We bucket them "
               "into Strack-2014 chapters — the scheme the model will use.")

    which = st.radio("Diagnosis position", mappings.DIAGNOSIS_COLS, horizontal=True,
                     help="diag_1 = primary, diag_2/3 = secondary diagnoses.")
    data = cached_diagnoses(which)
    chapters = data["chapters"]
    br = base_rate(facts)

    st.subheader(f"Chapter frequency & readmit rate — {which}")
    cdf = pd.DataFrame(chapters)
    fig = go.Figure()
    fig.add_bar(x=cdf["chapter"], y=cdf["count"], marker_color=PRIMARY, name="count",
                hovertemplate="%{x}<br>count=%{y:,}<extra></extra>")
    fig.add_trace(go.Scatter(
        x=cdf["chapter"], y=100 * cdf["positive_rate"], name="readmit<30 %",
        mode="markers+lines", yaxis="y2",
        marker=dict(color=POS_COLOR, size=10),
        hovertemplate="%{x}<br>readmit<30 = %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=100 * br, yref="y2", line=dict(color=POS_COLOR, dash="dot"))
    fig.update_layout(
        height=400, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="count"),
        yaxis2=dict(title="readmit<30 %", overlaying="y", side="right",
                    showgrid=False, color=POS_COLOR, rangemode="tozero"),
        legend=dict(orientation="h", y=1.08, x=0), xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')

    a, b = st.columns([1, 1])
    with a:
        st.subheader("Diabetes coded in any position")
        d = data["diabetes_any"]
        fig2 = rate_bar(["has diabetes dx", "no diabetes dx"],
                        [100 * d["with"]["rate"], 100 * d["without"]["rate"]],
                        [d["with"]["count"], d["without"]["count"]], d["base"])
        st.plotly_chart(fig2, width='stretch')
        st.caption("Surprisingly flat — a diabetes diagnosis is near-universal here, so "
                   "its *presence* barely separates the class.")
    with b:
        st.subheader("Comorbidity load")
        com = data["comorbidity"]
        comdf = pd.DataFrame(com)
        fig3 = go.Figure(go.Scatter(
            x=comdf["n_diagnoses"], y=100 * comdf["positive_rate"],
            mode="lines+markers", line=dict(color=POS_COLOR, width=2),
            hovertemplate="%{x} diagnoses<br>readmit<30 = %{y:.1f}%<extra></extra>"))
        fig3.add_hline(y=100 * br, line=dict(color=NEG_COLOR, dash="dot"))
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="number_diagnoses", yaxis_title="readmit<30 %")
        st.plotly_chart(fig3, width='stretch')
        st.caption("More distinct diagnoses ⇒ sicker patient ⇒ higher readmit risk.")


def page_medications(df: pd.DataFrame, facts: dict):
    st.title("💊 Medications")
    st.caption("23 drug columns record the dose-change at this encounter "
               "(No / Down / Steady / Up). Most are barely prescribed.")
    data = cached_medications()
    land = data["landscape"]
    sig = data["signal"]

    st.subheader("Prescribing landscape")
    ldf = pd.DataFrame(land)
    fig = go.Figure(go.Bar(
        y=ldf["drug"][::-1], x=ldf["pct_active"][::-1], orientation="h",
        marker_color=[("#9ca3af" if nc else PRIMARY) for nc in ldf["near_constant"]][::-1],
        hovertemplate="%{y}: active in %{x:.2f}% of encounters<extra></extra>"))
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="% of encounters with a non-'No' value")
    st.plotly_chart(fig, width='stretch')
    n_const = int(ldf["near_constant"].sum())
    st.info(f"**{n_const} of 23** drugs are near-constant (active in < 0.1% of "
            "encounters, grey bars) — including the literal constants `examide` & "
            "`citoglipton`. These are drop candidates. **`insulin`** dominates "
            "(~53% active) and carries real signal.")
    with st.expander("Full dose-change table"):
        st.dataframe(
            ldf[["drug", "pct_active", "No", "Down", "Steady", "Up", "rate_active", "near_constant"]],
            hide_index=True, width='stretch', height=420,
            column_config={"pct_active": st.column_config.NumberColumn("% active", format="%.2f"),
                           "rate_active": st.column_config.NumberColumn("readmit% (active)", format="%.3f")})

    st.divider()
    st.subheader("Signal from the high-coverage indicators")
    cols = st.columns(3)
    for c, key in zip(cols, ["insulin", "change", "diabetesMed"]):
        rows = sig[key]
        with c:
            st.markdown(f"**{key}**")
            st.plotly_chart(
                rate_bar([r["level"] for r in rows], [100 * r["positive_rate"] for r in rows],
                         [r["count"] for r in rows], sig["base"]),
                width='stretch')
    st.subheader("Number of active diabetes meds vs readmit")
    nm = pd.DataFrame(sig["n_active_meds"])
    st.plotly_chart(
        rate_bar([str(int(x)) for x in nm["n"]], [100 * r for r in nm["positive_rate"]],
                 list(nm["count"]), sig["base"]),
        width='stretch')
    st.caption("`insulin` up/down changes and a medication change (`change=Ch`) both "
               "track higher readmission — markers of an unstable regimen.")


def page_subgroups(df: pd.DataFrame, facts: dict):
    st.title("⚖️ Demographic subgroups")
    st.caption("Readmit rate across protected attributes, with 95% Wilson confidence "
               "intervals. A preview of the Fairlearn fairness audit.")

    col = st.radio("Attribute", ["race", "gender", "age"], horizontal=True)
    sg = cached_subgroups(col)
    rows = sg["rows"]
    base = sg["base"]
    rdf = pd.DataFrame(rows)
    rate = 100 * rdf["positive_rate"]
    up = 100 * rdf["ci_high"] - rate
    dn = rate - 100 * rdf["ci_low"]
    fig = go.Figure(go.Bar(
        x=rdf["level"], y=rate, marker_color=PRIMARY,
        error_y=dict(type="data", symmetric=False, array=up, arrayminus=dn,
                     color="#475569", thickness=1.3),
        customdata=rdf["count"],
        hovertemplate="%{x}<br>readmit<30 = %{y:.2f}%<br>n=%{customdata:,}<extra></extra>"))
    fig.add_hline(y=100 * base, line=dict(color=POS_COLOR, dash="dot"),
                  annotation_text=f"overall {100*base:.1f}%")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10),
                      yaxis_title="readmit<30 %", xaxis_title=col)
    st.plotly_chart(fig, width='stretch')

    st.dataframe(
        rdf.assign(**{"readmit %": (100 * rdf["positive_rate"]).round(2)})[
            ["level", "count", "readmit %"]],
        hide_index=True, width='stretch',
        column_config={"count": st.column_config.NumberColumn(format="%d")})
    st.info("Wide intervals on small groups (e.g. `Asian`, young age bands) mean apparent "
            "gaps there are **statistical noise**, not real disparities — the fairness "
            "audit must weight by group size. Note `gender`/`race` rates sit close to the "
            "overall base rate; `age` shows the clearest gradient.")


# --------------------------------------------------------------------------- #
# Data Dictionary — plain-language column guide + feature interdependency map
# --------------------------------------------------------------------------- #

# One colour per analytical family, used to tint the interdependency-graph nodes
# and the family chips. Kept local to the page so the rest of the dashboard's
# palette is untouched.
DD_FAMILY_COLORS: dict[str, str] = {
    "Identifier": "#64748b",
    "Demographic": "#0ea5e9",
    "Administrative ID (coded)": "#6366f1",
    "Administrative text": "#8b5cf6",
    "Utilization (numeric)": "#2563eb",
    "Diagnosis (ICD-9)": "#0d9488",
    "Lab result": "#db2777",
    "Medication (dose change)": "#9333ea",
    "Indicator": "#ca8a04",
    "Target": "#dc2626",
}

# Order the clusters are placed around the ring (target sits at the centre, since
# almost every edge points at it).
DD_RING_ORDER = [
    "patient-identity", "demographics", "encounter-context",
    "utilization-intensity", "diagnoses", "glucose-management",
]


def _dd_node_cluster(node_id: str) -> str:
    """Cluster a graph node belongs to (the collapsed drug node → glucose mgmt)."""
    if node_id == "medications":
        return "glucose-management"
    c = dd.cluster_of(node_id)
    return c["id"] if c else "target"


def _dd_layout(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    """Deterministic cluster layout (no networkx): each cluster gets a slot on a
    ring, its members sit on a small circle around that slot, target at centre."""
    by_cluster: dict[str, list[str]] = {}
    for n in nodes:
        by_cluster.setdefault(_dd_node_cluster(n["id"]), []).append(n["id"])

    pos: dict[str, tuple[float, float]] = {}
    ring = [c for c in DD_RING_ORDER if c in by_cluster]
    R = 1.18
    for i, cid in enumerate(ring):
        ang = 2 * np.pi * i / len(ring) - np.pi / 2
        cx, cy = R * np.cos(ang), R * np.sin(ang)
        members = by_cluster[cid]
        m = len(members)
        if m == 1:
            pos[members[0]] = (float(cx), float(cy))
            continue
        r = 0.16 + 0.024 * m
        for j, nid in enumerate(members):
            a = 2 * np.pi * j / m + ang
            pos[nid] = (float(cx + r * np.cos(a)), float(cy + r * np.sin(a)))
    # target (and any stragglers) at the hub
    for cid, members in by_cluster.items():
        if cid in ring:
            continue
        for nid in members:
            pos[nid] = (0.0, 0.0)
    return pos


def _dd_node_meta(node: dict) -> tuple[int, str, str]:
    """(marker size, symbol, hovertext) for a graph node."""
    nid = node["id"]
    if nid == "readmitted":
        return 30, "star", "<b>readmitted &lt;30d</b><br>the target — every column is judged against it"
    if node["kind"] == "group":
        return 30, "circle", ("<b>Medications (23 drugs)</b><br>the 23 antidiabetic dose-change "
                              "columns, collapsed.<br>insulin is shown separately (it carries the signal)")
    meaning = dd.COLUMNS.get(nid, {}).get("meaning", "")
    return 20, "circle", f"<b>{nid}</b><br>{_wrap(meaning, 46)}"


def _wrap(text: str, width: int) -> str:
    """Soft-wrap for plotly hover (<br> every ~width chars at a space)."""
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "<br>".join(out)


def interdependency_graph(focus: str | None = None) -> go.Figure:
    """Network of the curated feature interdependencies. ``focus`` (a node id)
    dims everything not directly connected to it, turning the full map into an
    ego-graph for one feature."""
    nodes = dd.GRAPH_NODES
    edges = dd.GRAPH_EDGES
    pos = _dd_layout(nodes)

    # Resolve focus to a node present in the graph (collapsed drugs → 'medications').
    if focus is not None and focus not in pos:
        focus = "medications" if dd.COLUMNS.get(focus, {}).get("family") == \
            "Medication (dose change)" else None
    neighbors: set[str] = set()
    if focus is not None:
        neighbors = {focus}
        for e in edges:
            if e["source"] == focus:
                neighbors.add(e["target"])
            elif e["target"] == focus:
                neighbors.add(e["source"])

    fig = go.Figure()

    # --- edges, grouped by strength so each gets its own legend entry / style ---
    drawn_legend: set[str] = set()
    inactive_xs, inactive_ys = [], []
    for e in edges:
        s, t = e["source"], e["target"]
        if s not in pos or t not in pos:
            continue
        x0, y0 = pos[s]; x1, y1 = pos[t]
        active = focus is None or (s in neighbors and t in neighbors)
        if not active:
            inactive_xs += [x0, x1, None]; inactive_ys += [y0, y1, None]
            continue
        meta = dd.STRENGTH_META[e["strength"]]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(width=meta["width"], color=meta["color"],
                      dash=meta["dash"]),
            name=meta["label"], legendgroup=e["strength"],
            showlegend=e["strength"] not in drawn_legend,
            hoverinfo="skip",
        ))
        drawn_legend.add(e["strength"])
    if inactive_xs:
        fig.add_trace(go.Scatter(
            x=inactive_xs, y=inactive_ys, mode="lines",
            line=dict(width=1, color="#e2e8f0"), hoverinfo="skip",
            showlegend=False))

    # --- invisible edge-midpoint markers carrying the link label on hover ---
    mx, my, mtext = [], [], []
    for e in edges:
        s, t = e["source"], e["target"]
        if s not in pos or t not in pos:
            continue
        if focus is not None and not (s in neighbors and t in neighbors):
            continue
        x0, y0 = pos[s]; x1, y1 = pos[t]
        mx.append((x0 + x1) / 2); my.append((y0 + y1) / 2)
        mtext.append(f"{s} — {t}<br><i>{e['label']}</i> ({e['strength']})")
    if mx:
        fig.add_trace(go.Scatter(
            x=mx, y=my, mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)"),
            hovertext=mtext, hoverinfo="text", showlegend=False))

    # --- nodes ---
    nx_, ny_, ncolor, nsize, nsym, ntext, nhover, nline = [], [], [], [], [], [], [], []
    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y = pos[nid]
        size, sym, hover = _dd_node_meta(n)
        active = focus is None or nid in neighbors
        base = DD_FAMILY_COLORS.get(n["family"], "#64748b")
        nx_.append(x); ny_.append(y)
        ncolor.append(base if active else "#e2e8f0")
        nsize.append(size); nsym.append(sym)
        ntext.append(n["label"] if active else "")
        nhover.append(hover)
        nline.append("#1e293b" if nid == focus else "#ffffff")
    fig.add_trace(go.Scatter(
        x=nx_, y=ny_, mode="markers+text",
        marker=dict(size=nsize, color=ncolor, symbol=nsym,
                    line=dict(width=[2.5 if c == "#1e293b" else 1.2 for c in nline],
                              color=nline)),
        text=ntext, textposition="bottom center",
        textfont=dict(size=10, color="#0f172a"),
        hovertext=nhover, hoverinfo="text", showlegend=False))

    fig.update_layout(
        height=620, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="white",
        xaxis=dict(visible=False, range=[-1.75, 1.75]),
        yaxis=dict(visible=False, range=[-1.6, 1.6], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    title="edge strength", font=dict(size=11)),
    )
    return fig


def _dd_family_legend() -> str:
    chips = []
    for fam, color in DD_FAMILY_COLORS.items():
        chips.append(
            f"<span class='pill' style='background:{color}22;color:{color};"
            f"border:1px solid {color}55'>{fam}</span>")
    return " ".join(chips)


def _dd_link_line(kind: str, target: str, why: str, *, reverse: bool = False) -> str:
    meta = dd.KIND_META[kind]
    arrow = "←" if reverse else "→"
    return (f"<div class='small' style='margin:2px 0'>{arrow} "
            f"<span title='{meta['label']}'>{meta['icon']}</span> "
            f"<code>{target}</code> — {why}</div>")


def _dd_column_card(col: str, facts: dict, *, compact: bool = True):
    entry = dd.COLUMNS[col]
    fam = entry["family"]
    color = DD_FAMILY_COLORS.get(fam, "#64748b")
    miss = facts["columns"].get(col, {}).get("pct_missing", 0.0)
    miss_pill = (f" <span class='pill pill-drop'>{miss:.0f}% missing</span>"
                 if miss >= 5 else "")
    st.markdown(
        f"<b><code>{col}</code></b> "
        f"<span class='pill' style='background:{color}22;color:{color}'>{fam}</span>"
        f"{miss_pill}<br>{entry['meaning']}<br>"
        f"<span class='small'>📐 {entry['reads_as']}</span>",
        unsafe_allow_html=True,
    )
    links = entry["links"]
    if links:
        st.markdown(
            "".join(_dd_link_line(l["kind"], l["to"], l["why"]) for l in links),
            unsafe_allow_html=True)


def _dd_guide(facts: dict):
    st.markdown(
        "Every raw column, grouped into the seven conceptual blocks the data falls "
        "into. Each entry: what it **is**, how to **read its values** (📐), and the "
        "**links** it has to other columns — "
        f"{dd.KIND_META['empirical']['icon']} measured · "
        f"{dd.KIND_META['mechanical']['icon']} computed-from · "
        f"{dd.KIND_META['clinical']['icon']} clinical · "
        f"{dd.KIND_META['governance']['icon']} pipeline-rule."
    )
    for ci, cluster in enumerate(dd.CLUSTERS):
        members = [c for c in cluster["members"] if c in dd.COLUMNS]
        with st.expander(f"{cluster['label']}  ·  {len(members)} columns",
                         expanded=(ci < 2)):
            st.caption(cluster["summary"])
            for k, col in enumerate(members):
                if k:
                    st.divider()
                _dd_column_card(col, facts)


def _dd_map():
    st.markdown(
        "How the features hang together. Nodes are coloured by family; the 23 "
        "sparse drug columns are collapsed into one **Medications** node (with "
        "`insulin` kept separate). Solid edges are **measured correlations** "
        "(thicker = stronger); dotted edges are **conceptual** links "
        "(computed-from / clinical / governance). The target sits at the centre."
    )
    st.plotly_chart(interdependency_graph(), width="stretch")
    st.markdown(_dd_family_legend(), unsafe_allow_html=True)
    st.divider()
    st.subheader("The seven blocks")
    cols = st.columns(2)
    for i, cluster in enumerate(dd.CLUSTERS):
        with cols[i % 2]:
            st.markdown(f"**{cluster['label']}** — <span class='small'>"
                        f"{cluster['summary']}</span>", unsafe_allow_html=True)


def _dd_numeric_neighbors(col: str, k: int = 6) -> list[tuple[str, float]]:
    """Top live |Pearson r| partners for a numeric column (age → age_ordinal)."""
    corr = cached_correlation("pearson")
    cols, mat = corr["columns"], corr["matrix"]
    name = "age_ordinal" if col == "age" else col
    if name not in cols:
        return []
    i = cols.index(name)
    out = [(c, mat[i][j]) for j, c in enumerate(cols) if j != i]
    out.sort(key=lambda x: -abs(x[1]))
    return out[:k]


def _dd_explorer(facts: dict):
    st.markdown("Pick a column to see its meaning, the links it declares to other "
                "columns, which columns point **back** at it, and its live numbers.")
    options = list(dd.COLUMNS.keys())
    col = st.selectbox(
        "Column", options, index=options.index("number_inpatient"),
        format_func=lambda c: f"{c}  ·  {dd.COLUMNS[c]['family']}")

    entry = dd.COLUMNS[col]
    cluster = dd.cluster_of(col)
    left, right = st.columns([1.15, 1])

    with left:
        _dd_column_card(col, facts)
        if cluster:
            st.caption(f"🧩 Block: **{cluster['label']}** — {cluster['summary']}")
        st.markdown("**Referenced by** (columns that link *to* this one)")
        incoming = dd.links_in(col)
        if incoming:
            st.markdown(
                "".join(_dd_link_line(l["kind"], l["from"], l["why"], reverse=True)
                        for l in incoming),
                unsafe_allow_html=True)
        else:
            st.caption("— nothing points here.")

    with right:
        assoc = facts["columns"].get(col, {})
        st.markdown("**Live numbers**")
        m1, m2 = st.columns(2)
        m1.metric("Missing", f"{assoc.get('pct_missing', 0.0):.1f}%")
        lift = assoc.get("target_lift")
        m2.metric("Target lift", f"{lift:.2f}×" if lift else "—")

        neigh = _dd_numeric_neighbors(col)
        if neigh:
            st.caption("Strongest measured correlations (Pearson r):")
            ndf = pd.DataFrame(neigh, columns=["column", "r"])
            ndf["column"] = ndf["column"].replace(
                {"age_ordinal": "age", "readmitted_30d": "readmitted<30d"})
            st.dataframe(
                ndf, hide_index=True, width="stretch",
                column_config={"r": st.column_config.NumberColumn(format="%.3f")})
        else:
            ta = next((r for r in cached_association() if r["feature"] == col), None)
            if ta:
                st.caption("Association with the target (this is a categorical "
                           "feature, so we use Cramér's V):")
                st.metric(f"{col} ↔ readmit<30d", f"{ta['assoc']:.3f}",
                          help=ta["metric"])

    st.divider()
    st.subheader("This feature in the map")
    st.caption("The full interdependency graph, dimmed to this feature and its "
               "direct neighbours.")
    st.plotly_chart(interdependency_graph(focus=col), width="stretch")


def page_dictionary(df: pd.DataFrame, facts: dict):
    st.title("📖 Data Dictionary & Feature Map")
    st.caption(
        "A plain-language tour of all 49 raw columns — what each field means, how "
        "to read its values, and how the features depend on one another. The "
        "meanings and links were drafted per family and fact-checked against the "
        "live correlation numbers."
    )
    tab_guide, tab_map, tab_explore = st.tabs(
        ["📋 Column guide", "🕸️ Interdependency map", "🔎 Relationship explorer"])
    with tab_guide:
        _dd_guide(facts)
    with tab_map:
        _dd_map()
    with tab_explore:
        _dd_explorer(facts)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

def main():
    df = load_df()
    facts = load_facts()
    analysis = load_analysis()

    st.sidebar.title("🏥 Readmission EDA")
    st.sidebar.caption("Diabetes 130-US hospitals · 1999–2008")
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Data Dictionary", "Column X-ray", "Correlation", "Missingness",
         "Target & signal", "Patient & leakage", "Diagnoses (ICD-9)", "Medications",
         "Subgroups", "Imputation plan"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    if analysis:
        st.sidebar.success(f"Analysis loaded · {analysis.get('n_columns', '?')} columns")
    else:
        st.sidebar.warning("Column-analysis artifact not found")
    st.sidebar.caption(
        f"{facts['target']['n_rows']:,} rows · "
        f"{facts['schema']['n_columns_raw']-1} features · "
        f"base rate {100*base_rate(facts):.1f}%"
    )

    if page == "Overview":
        page_overview(df, facts)
    elif page == "Data Dictionary":
        page_dictionary(df, facts)
    elif page == "Column X-ray":
        page_column_xray(df, facts, analysis)
    elif page == "Correlation":
        page_correlation(df, facts)
    elif page == "Missingness":
        page_missingness(df, facts, analysis)
    elif page == "Target & signal":
        page_target(df, facts)
    elif page == "Patient & leakage":
        page_patient(df, facts)
    elif page == "Diagnoses (ICD-9)":
        page_diagnoses(df, facts)
    elif page == "Medications":
        page_medications(df, facts)
    elif page == "Subgroups":
        page_subgroups(df, facts)
    elif page == "Imputation plan":
        page_imputation(df, facts, analysis)


if __name__ == "__main__":
    main()

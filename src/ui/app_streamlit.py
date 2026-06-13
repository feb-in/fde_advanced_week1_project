"""app_streamlit.py — clinical decision-support front-end for the readmission API.

THIN CLIENT. This UI computes nothing — no features, no score, no threshold logic. It
collects raw patient fields (form derived from src/contracts/input_contract.json so it
matches the API schema exactly), POSTs them to the running API's /predict over HTTP
(src/ui/api_client.py), and renders the response. All intelligence stays server-side;
doing any featurization here would reintroduce train/serve skew.

Run (API must be running separately):
    READMISSION_API_URL=http://localhost:8000 uv run --group ui \
        streamlit run src/ui/app_streamlit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo/src
from ui import api_client  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / "src/contracts/input_contract.json").read_text())
FIELDS = CONTRACT["fields"]
DEFAULTS = json.loads((ROOT / "tests/sample_request.json").read_text())  # realistic sample patient

THRESHOLD_HINT = 0.091046  # only for the illustrative band; the real one comes from the API response

# Human labels for the high-signal fields (drugs fall back to their raw name).
LABELS = {
    "race": "Race", "gender": "Sex", "age": "Age band",
    "number_inpatient": "Inpatient visits (prior yr)",
    "number_emergency": "Emergency visits (prior yr)",
    "number_outpatient": "Outpatient visits (prior yr)",
    "admission_type_id": "Admission type (code 1–8)",
    "admission_source_id": "Admission source (code 1–26)",
    "discharge_disposition_id": "Discharge disposition (code 1–30)",
    "time_in_hospital": "Length of stay (days)", "num_medications": "Distinct medications",
    "num_lab_procedures": "Lab procedures", "num_procedures": "Other procedures",
    "number_diagnoses": "Diagnoses recorded", "A1Cresult": "A1C result",
    "max_glu_serum": "Max glucose serum", "diabetesMed": "On diabetes medication",
    "change": "Medication changed this stay", "insulin": "Insulin", "metformin": "Metformin",
    "diag_1": "Primary diagnosis (ICD-9)", "diag_2": "Secondary diagnosis (ICD-9)",
    "diag_3": "Additional diagnosis (ICD-9)", "payer_code": "Payer code",
    "medical_specialty": "Admitting specialty",
}
DRUG_FIELDS = [f for f, s in FIELDS.items() if s.get("allowed_values") == ["No", "Down", "Steady", "Up"]]
MAIN_MEDS = ["insulin", "metformin"]
ADVANCED_MEDS = [d for d in DRUG_FIELDS if d not in MAIN_MEDS]

C_BLUE, C_RED, C_AMBER, C_GREEN, C_GREY = "#2c5f8a", "#c0392b", "#e67e22", "#2e8b57", "#6b7280"


# ---------------------------------------------------------------------------
# Widgets (one per contract field — no invented fields)
# ---------------------------------------------------------------------------
def _cat(name, options=None, fmt=None):
    spec = FIELDS[name]
    opts = options or spec["allowed_values"]
    d = DEFAULTS.get(name)
    idx = opts.index(d) if d in opts else 0
    return st.selectbox(LABELS.get(name, name), opts, index=idx,
                        format_func=fmt or (lambda x: x))


def _int(name):
    spec = FIELDS[name]
    return int(st.number_input(LABELS.get(name, name), min_value=int(spec["min"]),
                               max_value=int(spec["max"]), value=int(DEFAULTS.get(name, spec["min"])),
                               step=1))


def _str(name):
    d = DEFAULTS.get(name)
    return st.text_input(LABELS.get(name, name), value="" if d is None else str(d))


def render_form():
    """Render every input field and return (payload, submitted)."""
    with st.form("patient", border=False):
        vals = dict(DEFAULTS)  # guarantees all 44 fields are present even if untouched

        st.markdown("##### Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            race = _cat("race", options=FIELDS["race"]["allowed_values"] + ["Unknown"])
            vals["race"] = "?" if race == "Unknown" else race
        with c2:
            vals["gender"] = _cat("gender", options=["Female", "Male"])  # drop Unknown/Invalid (API rejects it)
        with c3:
            vals["age"] = _cat("age")

        st.markdown("##### Prior utilization (last 12 months)")
        c1, c2, c3 = st.columns(3)
        with c1:
            vals["number_inpatient"] = _int("number_inpatient")
        with c2:
            vals["number_emergency"] = _int("number_emergency")
        with c3:
            vals["number_outpatient"] = _int("number_outpatient")

        st.markdown("##### This admission")
        c1, c2, c3 = st.columns(3)
        with c1:
            vals["admission_type_id"] = _int("admission_type_id")
            vals["time_in_hospital"] = _int("time_in_hospital")
            vals["number_diagnoses"] = _int("number_diagnoses")
        with c2:
            vals["admission_source_id"] = _int("admission_source_id")
            vals["num_medications"] = _int("num_medications")
        with c3:
            vals["discharge_disposition_id"] = _int("discharge_disposition_id")
            vals["num_lab_procedures"] = _int("num_lab_procedures")
            vals["num_procedures"] = _int("num_procedures")

        st.markdown("##### Labs & medications")
        c1, c2, c3 = st.columns(3)
        with c1:
            vals["A1Cresult"] = _cat("A1Cresult")
            vals["max_glu_serum"] = _cat("max_glu_serum")
        with c2:
            vals["diabetesMed"] = _cat("diabetesMed")
            vals["change"] = _cat("change")
        with c3:
            vals["insulin"] = _cat("insulin")
            vals["metformin"] = _cat("metformin")

        st.markdown("##### Diagnoses (ICD-9)")
        c1, c2, c3 = st.columns(3)
        with c1:
            d1 = _str("diag_1")
        with c2:
            d2 = _str("diag_2")
        with c3:
            d3 = _str("diag_3")
        vals["diag_1"], vals["diag_2"], vals["diag_3"] = (d1 or "?", d2 or "?", d3 or "?")

        with st.expander("Advanced fields (other medications & administrative)"):
            pc = _str("payer_code")
            ms = _str("medical_specialty")
            vals["payer_code"] = pc or None
            vals["medical_specialty"] = ms or None
            st.caption("Other diabetes medications (default **No** = not prescribed):")
            cols = st.columns(3)
            for i, drug in enumerate(ADVANCED_MEDS):
                with cols[i % 3]:
                    spec = FIELDS[drug]
                    d = DEFAULTS.get(drug, "No")
                    vals[drug] = st.selectbox(drug, spec["allowed_values"],
                                              index=spec["allowed_values"].index(d) if d in spec["allowed_values"] else 0,
                                              key=f"drug_{drug}")

        submitted = st.form_submit_button("Assess risk", type="primary", width="stretch")
    return vals, submitted


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def _band(prob, threshold):
    if prob >= threshold:
        return "HIGH", C_RED
    if prob >= 0.05:
        return "MODERATE", C_AMBER
    return "LOW", C_GREEN


def factor_chart(top_factors):
    df = pd.DataFrame(top_factors)
    df["value"] = df["value"].astype(str)  # feature values are mixed str/int → unify for Arrow
    df["abs"] = df["contribution"].abs()
    df["Effect"] = df["direction"].map({"increases": "raises risk", "decreases": "lowers risk"})
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("contribution:Q", title="Signed contribution (log-odds)"),
            y=alt.Y("feature:N", sort=alt.EncodingSortField(field="abs", order="descending"), title=None),
            color=alt.Color("Effect:N",
                            scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[C_RED, C_GREEN]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("feature:N", title="Feature"),
                     alt.Tooltip("value:N", title="Value"),
                     alt.Tooltip("contribution:Q", title="Contribution", format=".4f")],
        )
        .properties(height=28 * len(df) + 40)
    )


def render_results():
    state = st.session_state
    if state.get("error"):
        kind, info = state["error"]
        if kind == "connection":
            st.error(f"**API not reachable** at `{info}`. Start it "
                     "(`uv run uvicorn src.app.app:app --port 8000` or `podman compose up -d`) "
                     "and set `READMISSION_API_URL` if it lives elsewhere.")
        else:
            fields = sorted({".".join(str(p) for p in e.get("loc", [])[1:]) for e in info}) if isinstance(info, list) else []
            st.error("**The record was rejected — please check these fields:** "
                     + (", ".join(f"`{f}`" for f in fields) if fields else str(info)))
        return

    res = state.get("result")
    if not res:
        st.info("Enter a patient on the left and press **Assess risk**. "
                "The form is pre-filled with a sample patient — one click runs the demo.")
        return

    prob, threshold, flag = res["readmission_probability"], res["threshold"], res["flag"]
    label, color = _band(prob, threshold)

    st.markdown(f"<div class='riskcard'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.metric("30-day readmission risk", f"{prob * 100:.1f}%")
    with c2:
        st.markdown(
            f"<div class='band' style='background:{color}'>{label} RISK</div>",
            unsafe_allow_html=True)
    st.caption("Risk bands (Low / Moderate / High) are an illustrative reading aid — "
               "the model output is the probability above.")

    if flag:
        st.markdown(f"<div class='flag flagon'>⚑ FLAG for 30-day follow-up</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='flag flagoff'>No flag — routine discharge</div>",
                    unsafe_allow_html=True)
    st.caption(f"Decision rule (applied by the API, shown for transparency): flag when "
               f"probability ≥ **{threshold:.4f}**. This patient: {prob:.4f}.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("##### Top contributing factors")
    st.altair_chart(factor_chart(res["top_factors"]), width="stretch")
    st.caption("Directions are reliable; magnitudes are on the model's uncalibrated "
               "(log-odds) scale — read these as *what pushed this patient's risk up or down*, "
               "not as exact probability changes.")

    st.divider()
    st.caption(f"Model: **{res['model_name']}** v{res['model_version']} @ "
               f"`{res['model_alias']}` · screening aid, not a diagnosis.")


# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="30-Day Readmission Risk — Decision Support",
                       page_icon="🏥", layout="wide")
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2rem;}
          .riskcard {background:#f6f8fa; border:1px solid #e3e8ee; border-radius:12px;
                     padding:1.1rem 1.3rem; margin-bottom:0.4rem;}
          .band {color:white; text-align:center; font-weight:700; letter-spacing:.04em;
                 padding:.55rem; border-radius:8px; margin-top:1.1rem;}
          .flag {font-weight:700; padding:.6rem .8rem; border-radius:8px; margin:.4rem 0;}
          .flagon {background:#fdecea; color:#c0392b; border:1px solid #f5c6c0;}
          .flagoff {background:#eef6ee; color:#2e8b57; border:1px solid #cfe6cf;}
          h5 {color:#2c5f8a; margin-bottom:.2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("30-Day Readmission Risk — Decision Support")
    st.markdown("<p style='color:#6b7280; margin-top:-0.6rem;'>A screening aid that ranks "
                "discharged diabetic patients by 30-day readmission risk — not a diagnosis, "
                "and not a substitute for clinical judgement.</p>", unsafe_allow_html=True)

    api_url = api_client.DEFAULT_BASE_URL
    h = api_client.health(api_url)
    status = (f"connected · {h['model_name']} v{h['model_version']} @ {h['model_alias']}"
              if h else "not reachable")
    st.caption(f"API: `{api_url}` — {status}")

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.subheader("Patient")
        vals, submitted = render_form()
    if submitted:
        try:
            st.session_state.result = api_client.predict(vals, api_url)
            st.session_state.error = None
        except api_client.APIValidationError as exc:
            st.session_state.error = ("validation", exc.detail)
            st.session_state.result = None
        except api_client.APIConnectionError as exc:
            st.session_state.error = ("connection", str(exc))
            st.session_state.result = None
    with right:
        st.subheader("Assessment")
        render_results()


if __name__ == "__main__":
    main()

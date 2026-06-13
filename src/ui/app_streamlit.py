"""app_streamlit.py — clinical decision-support front-end for the readmission API.

THIN CLIENT. This UI computes nothing — no features, no score, no threshold logic. It
collects raw patient fields (form derived from src/contracts/input_contract.json so it
matches the API schema exactly), POSTs them to the running API's /predict over HTTP
(src/ui/api_client.py), and renders the response. All intelligence stays server-side;
doing any featurization here would reintroduce train/serve skew.

The "Load random patient" feature reads local data files (src/ui/sample_data.py) ONLY to
fill the form with a real held-out-test patient and reveal its true outcome — it still
scores through the API, never locally.

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
from ui import api_client, sample_data  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT / "src/contracts/input_contract.json").read_text())
FIELDS = CONTRACT["fields"]
SAMPLE = json.loads((ROOT / "tests/sample_request.json").read_text())  # default pre-fill patient
API_URL = api_client.DEFAULT_BASE_URL

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
RACE_OPTS = FIELDS["race"]["allowed_values"] + ["Unknown"]

C_BLUE, C_RED, C_AMBER, C_GREEN, C_GREY = "#2c5f8a", "#c0392b", "#e67e22", "#2e8b57", "#6b7280"


# ---------------------------------------------------------------------------
# State <-> widgets <-> API payload (the UI never scores; it only shuffles fields)
# ---------------------------------------------------------------------------
def fkey(name):
    return f"f_{name}"


def raw_to_widget(name, raw):
    """A raw record value → the value stored in the widget's session_state."""
    spec = FIELDS[name]
    if name == "race":
        return raw if raw in FIELDS["race"]["allowed_values"] else "Unknown"
    if name == "gender":
        return raw if raw in ("Female", "Male") else "Female"
    if spec["type"] == "integer":
        return int(raw) if raw is not None else int(spec["min"])
    if name in ("diag_1", "diag_2", "diag_3", "payer_code", "medical_specialty"):
        return "" if raw in (None, "?") else str(raw)
    return raw  # other categoricals (A1Cresult, max_glu_serum, change, diabetesMed, drugs)


def widget_to_api(name, v):
    """A widget value → the field value POSTed to /predict (mirrors training's missingness)."""
    spec = FIELDS[name]
    if name == "race":
        return "?" if v == "Unknown" else v
    if name in ("diag_1", "diag_2", "diag_3"):
        return v.strip() if (v and v.strip()) else "?"
    if name in ("payer_code", "medical_specialty"):
        return v.strip() if (v and v.strip()) else None
    if spec["type"] == "integer":
        return int(v)
    return v


def set_patient(record):
    for name in FIELDS:
        st.session_state[fkey(name)] = raw_to_widget(name, record.get(name))


def collect_payload():
    return {name: widget_to_api(name, st.session_state[fkey(name)]) for name in FIELDS}


def do_predict(payload):
    try:
        st.session_state["result"] = api_client.predict(payload, API_URL)
        st.session_state["error"] = None
    except api_client.APIValidationError as exc:
        st.session_state["error"] = ("validation", exc.detail)
        st.session_state["result"] = None
    except api_client.APIConnectionError as exc:
        st.session_state["error"] = ("connection", str(exc))
        st.session_state["result"] = None


def load_random_cb():
    """Button callback: real held-out patient → fill form, reveal truth, score via API."""
    try:
        record, truth = sample_data.random_test_patient()
    except Exception as exc:  # noqa: BLE001 — surface data-file issues, don't crash the app
        st.session_state["error"] = ("data", str(exc))
        return
    set_patient(record)
    st.session_state["truth"] = truth
    do_predict(collect_payload())


def init_state():
    if st.session_state.get("_init"):
        return
    set_patient(SAMPLE)
    st.session_state.update(result=None, truth=None, error=None, _init=True)


# ---------------------------------------------------------------------------
# Form (one keyed widget per contract field — no invented fields)
# ---------------------------------------------------------------------------
def _cat(name, options=None):
    return st.selectbox(LABELS.get(name, name), options or FIELDS[name]["allowed_values"], key=fkey(name))


def _int(name):
    spec = FIELDS[name]
    return st.number_input(LABELS.get(name, name), min_value=int(spec["min"]),
                           max_value=int(spec["max"]), step=1, key=fkey(name))


def _str(name):
    return st.text_input(LABELS.get(name, name), key=fkey(name))


def render_form():
    with st.form("patient", border=False):
        st.markdown("##### Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            _cat("race", RACE_OPTS)
        with c2:
            _cat("gender", ["Female", "Male"])  # API rejects "Unknown/Invalid"
        with c3:
            _cat("age")

        st.markdown("##### Prior utilization (last 12 months)")
        c1, c2, c3 = st.columns(3)
        with c1:
            _int("number_inpatient")
        with c2:
            _int("number_emergency")
        with c3:
            _int("number_outpatient")

        st.markdown("##### This admission")
        c1, c2, c3 = st.columns(3)
        with c1:
            _int("admission_type_id"); _int("time_in_hospital"); _int("number_diagnoses")
        with c2:
            _int("admission_source_id"); _int("num_medications")
        with c3:
            _int("discharge_disposition_id"); _int("num_lab_procedures"); _int("num_procedures")

        st.markdown("##### Labs & medications")
        c1, c2, c3 = st.columns(3)
        with c1:
            _cat("A1Cresult"); _cat("max_glu_serum")
        with c2:
            _cat("diabetesMed"); _cat("change")
        with c3:
            _cat("insulin"); _cat("metformin")

        st.markdown("##### Diagnoses (ICD-9)")
        c1, c2, c3 = st.columns(3)
        with c1:
            _str("diag_1")
        with c2:
            _str("diag_2")
        with c3:
            _str("diag_3")

        with st.expander("Advanced fields (other medications & administrative)"):
            _str("payer_code")
            _str("medical_specialty")
            st.caption("Other diabetes medications (default **No** = not prescribed):")
            cols = st.columns(3)
            for i, drug in enumerate(ADVANCED_MEDS):
                with cols[i % 3]:
                    st.selectbox(drug, FIELDS[drug]["allowed_values"], key=fkey(drug))

        return st.form_submit_button("Assess risk", type="primary", width="stretch")


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


def render_truth_vs_prediction(res, truth):
    """Ground truth vs the model — shows right AND wrong predictions honestly."""
    flag, prob = res["flag"], res["readmission_probability"]
    was = truth["was_readmitted_30d"]
    correct = flag == was
    if flag and was:
        kind = "True positive — correctly flagged"
    elif (not flag) and (not was):
        kind = "True negative — correctly not flagged"
    elif flag and (not was):
        kind = "False positive — flagged, but was not readmitted"
    else:
        kind = "False negative — missed a real readmission"
    mark, tone = ("✓", C_GREEN) if correct else ("✗", C_RED)

    with st.container(border=True):
        st.caption(f"Held-out test patient · encounter {truth['encounter_id']} "
                   "(the model never trained on this row)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Model prediction**")
            st.markdown(f"{'⚑ FLAG' if flag else 'No flag'} · {prob * 100:.1f}%")
        with c2:
            st.markdown("**Actual outcome**")
            st.markdown(f"{'WAS' if was else 'was NOT'} readmitted within 30 days")
        st.markdown(f"<div class='verdict' style='color:{tone}'>{mark} {kind}</div>",
                    unsafe_allow_html=True)


def render_results():
    state = st.session_state
    if state.get("error"):
        kind, info = state["error"]
        if kind == "connection":
            st.error(f"**API not reachable** at `{info}`. Start it "
                     "(`uv run uvicorn src.app.app:app --port 8000` or `podman compose up -d`) "
                     "and set `READMISSION_API_URL` if it lives elsewhere.")
        elif kind == "data":
            st.error(f"**Could not load a test patient:** {info}. The held-out data may need "
                     "`dvc pull` (`data/featurized/…parquet`, `data/raw/diabetic_data.csv`).")
        else:
            fields = sorted({".".join(str(p) for p in e.get("loc", [])[1:]) for e in info}) if isinstance(info, list) else []
            st.error("**The record was rejected — please check these fields:** "
                     + (", ".join(f"`{f}`" for f in fields) if fields else str(info)))
        return

    res = state.get("result")
    if not res:
        st.info("Enter a patient on the left and press **Assess risk**, or **Load random "
                "patient** to score a real held-out case and see the true outcome. The form "
                "is pre-filled — one click runs the demo.")
        return

    if state.get("truth"):
        render_truth_vs_prediction(res, state["truth"])

    prob, threshold, flag = res["readmission_probability"], res["threshold"], res["flag"]
    label, color = _band(prob, threshold)

    with st.container(border=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("30-day readmission risk", f"{prob * 100:.1f}%")
        with c2:
            st.markdown(f"<div class='band' style='background:{color}'>{label} RISK</div>",
                        unsafe_allow_html=True)
        st.caption("Risk bands (Low / Moderate / High) are an illustrative reading aid — "
                   "the model output is the probability above.")
        if flag:
            st.markdown("<div class='flag flagon'>⚑ FLAG for 30-day follow-up</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<div class='flag flagoff'>No flag — routine discharge</div>",
                        unsafe_allow_html=True)
        st.caption(f"Decision rule (applied by the API, shown for transparency): flag when "
                   f"probability ≥ **{threshold:.4f}**. This patient: {prob:.4f}.")

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
          /* Tuned for the dark theme (launch with --theme.base dark). Colour cards use
             translucent tints so they sit on the dark background and still degrade
             acceptably if the app is ever run on a light theme. */
          .block-container {padding-top: 1.4rem;}
          .beta {background:rgba(230,170,40,0.12); border:1px solid rgba(230,170,40,0.40);
                 color:#e3c265; border-radius:8px; padding:.6rem .9rem; margin-bottom:1rem;
                 font-size:.92rem;}
          .verdict {font-weight:700; font-size:1.02rem; margin-top:.6rem;}
          .band {color:white; text-align:center; font-weight:700; letter-spacing:.04em;
                 padding:.55rem; border-radius:8px; margin-top:1.1rem;}
          .flag {font-weight:700; padding:.6rem .8rem; border-radius:8px; margin:.4rem 0;}
          .flagon {background:rgba(220,70,55,0.16); color:#ff8d7c;
                   border:1px solid rgba(220,70,55,0.45);}
          .flagoff {background:rgba(46,160,90,0.16); color:#74d199;
                    border:1px solid rgba(46,160,90,0.45);}
          h5 {color:#6fa8dc; margin-bottom:.2rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    init_state()

    st.title("30-Day Readmission Risk — Decision Support")
    st.markdown(
        "<div class='beta'>⚠ <b>DEMONSTRATION / BETA</b> — an interface for <b>evaluating "
        "the model</b>, <b>not a deployed medical device</b> and <b>not for real clinical "
        "decisions</b>. The model <i>informs</i>; a clinician reviews every case and decides "
        "(human-in-the-loop). Use only with de-identified or synthetic data.</div>",
        unsafe_allow_html=True,
    )

    h = api_client.health(API_URL)
    status = (f"connected · {h['model_name']} v{h['model_version']} @ {h['model_alias']}"
              if h else "not reachable")
    st.caption(f"API: `{API_URL}` — {status}")

    left, right = st.columns([5, 4], gap="large")
    with left:
        st.subheader("Patient")
        st.button("🎲 Load random patient (held-out test set)", on_click=load_random_cb,
                  width="stretch")
        st.caption("Pulls a real patient the model never trained on, scores it via the API, "
                   "and reveals the true outcome — so you can see the model right *and* wrong.")
        submitted = render_form()
    if submitted:
        st.session_state["truth"] = None  # manual entry has no ground-truth label
        do_predict(collect_payload())
    with right:
        st.subheader("Assessment")
        render_results()


if __name__ == "__main__":
    main()

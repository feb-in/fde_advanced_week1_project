# EDA — EDA dashboard

Self-contained exploratory data analysis for the diabetes 30-day readmission
dataset. Interactive Streamlit dashboard over all 49 columns: a plain-language
data dictionary + feature-interdependency map, distributions, missingness,
target signal, and a per-column fill strategy.

## Run it

```bash
# from the repo root
uv run streamlit run EDA/dashboard.py        # → http://localhost:8501
```

No `uv`? Use plain Python (needs `streamlit` + `plotly` + `pandas`):

```bash
pip install -r ../requirements.txt
streamlit run EDA/dashboard.py
```

Or the one-shot launcher (rebuilds the fact pack if missing, then serves):

```bash
./EDA/run.sh
```

Then open **http://localhost:8501** and use the sidebar to switch pages:
**Overview · Data Dictionary · Column X-ray · Correlation · Missingness ·
Target & signal · Patient & leakage · Diagnoses (ICD-9) · Medications ·
Subgroups · Imputation plan**.

The **Data Dictionary** page is the place to start if you're new to the data:
a column-by-column guide (what each field means + how to read its values),
an interactive interdependency map (how the features hang together, edges
weighted by real correlation strength), and a per-column relationship explorer.

To stop it: `Ctrl-C` in the terminal (or `pkill -f "streamlit run EDA/dashboard.py"`).
Run on a different port: `uv run streamlit run EDA/dashboard.py --server.port 8600`.

## Regenerate the artifacts (optional)

```bash
python EDA/profile.py          # rebuild artifacts/eda_facts.json from data/raw/
python EDA/make_findings.py    # rebuild EDA_FINDINGS.md from the artifacts
```

## Files

| File | Role |
|---|---|
| `dashboard.py` | the Streamlit app (run this) — 11 pages |
| `data_dictionary.py` | plain-language meanings, value guides + curated feature-interdependency graph (powers the **Data Dictionary** page) |
| `profile.py` | per-column analytical engine — all stats, computed live from `../data/raw/diabetic_data.csv` |
| `aspects.py` | cross-cutting analyses (correlation, association, ICD-9, medications, subgroups, leakage) |
| `mappings.py` | decodes the coded `*_id` columns, ICD-9 chapter grouping + graded-rule constants |
| `make_findings.py` | generates `EDA_FINDINGS.md` |
| `EDA_FINDINGS.md` | written per-column write-up |
| `artifacts/eda_facts.json` | cached fact pack |
| `artifacts/column_analysis.json` | verified per-column fill strategies (powers the Imputation pages) |

Reads the raw CSV read-only; nothing here mutates the shared dataset.

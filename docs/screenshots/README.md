# Screenshots / Evidence

Images embedded by the root `README.md` "Screenshots / Evidence" section.

| File | Content | Status |
|---|---|---|
| `ui-truth-vs-prediction.png` | Streamlit UI (dark) after **Load random patient** — the ✓/✗ truth-vs-prediction card, risk %, flag, top-factors chart. | ✅ present |
| `mlflow-experiments.png` | MLflow Runs view for `readmission-30d` — runs + PR-AUC/ROC-AUC metric charts. | ✅ present |
| `ci-green.png` | A green run of the **CI — test, build, push to ECR** workflow (all steps ✓). | ✅ present |
| `evidently-drift.png` | An Evidently data-drift report (currently the *control* / no-drift view). | ✅ present |
| `grafana-dashboard.png` | The Grafana *"Readmission API — Observability"* dashboard with panels populated. | ⬜ optional, not yet added |

To add the Grafana shot: `podman compose up -d`, send a little `/predict` traffic, open
`http://localhost:3000` (admin/admin), capture the dashboard, save as
`grafana-dashboard.png` here, then uncomment/add its embed in the root README.

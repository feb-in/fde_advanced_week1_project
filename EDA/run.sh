#!/usr/bin/env bash
# Launch the EDA dashboard. Run from anywhere; paths resolve relative to this file.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# (Re)build the cached fact pack if missing.
[ -f "$HERE/artifacts/eda_facts.json" ] || python3 "$HERE/profile.py"
exec python3 -m streamlit run "$HERE/dashboard.py" "$@"

"""Data contracts shared across the pipeline, serving, and monitoring.

`data_contract.py` is the single source of truth for the input data rules. It is
read three ways:
  1. GX batch suites (src/data/validate.py) — pipeline checkpoints.
  2. The serving Pydantic schema (next gate) — generated from input_contract.json.
  3. Stage 6 Evidently drift — the training reference schema.
"""

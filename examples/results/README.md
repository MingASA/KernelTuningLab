# Final Mock result summaries

These files accompany `../report/report.md` and contain synthetic infrastructure-validation results:

- `run_summary.json`: record/trial counts, status breakdown and artifact paths.
- `split_manifest.json`: held-out hardware identities and complete case/shape membership.
- `replay_summary.json`: hardware/operator/protocol-separated method summaries, budgets and seed variation.
- `l3_unavailable.json`: explicitly unavailable computations; the final full Mock run had none.

Full raw records, private Replay source tables, trace histories and SQLite journals remain under `artifacts/final-demo/`. They are reproducible with the documented Demo command and are not shipped as large duplicate example files. All performance values remain synthetic.

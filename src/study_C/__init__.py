"""Study C: licit/illicit node classification on the Elliptic Bitcoin dataset.

Nodes are Bitcoin transactions (not users or wallets); directed edges are
transaction-flow relations. Raw files under ``datasets/`` are read-only;
every derived artifact is written below ``reports/study_C/``.

Module map (in pipeline execution order):

    config.py    reads study_C.toml into typed dataclasses
    data.py      loads the three CSVs and VALIDATES them (data profile)
    splits.py    chronological train/val/test split on the time steps
    features.py  feature sets (local / all), scaler and class weights fit on train ONLY
    graph.py     builds the PyG graph visible to each phase (no edges from the future)
    models.py    Random Forest, MLP, GraphSAGE, GAT
    train.py     training loops with early stopping on validation
    evaluate.py  threshold chosen on validation, metrics on the illicit class
    reporting.py saves the run's tables and metadata
    figures.py   figures from the saved CSVs
    pipeline.py  orchestrates everything above
    main.py      entry point (no CLI: everything from study_C.toml)
"""

# Increment this number whenever you change how the raw data are transformed
# (columns, filters, split): it is stored in every run's metadata.
PREPROCESSING_VERSION = "0.1"

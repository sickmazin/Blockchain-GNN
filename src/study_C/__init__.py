"""Study C: licit/illicit node classification on the Elliptic Bitcoin dataset.

Nodes are Bitcoin transactions (not users or wallets); directed edges are
transaction-flow relations. Raw files under ``datasets/`` are read-only;
every derived artifact is written below ``reports/study_C/``.

Mappa dei moduli (in ordine di esecuzione nella pipeline):

    config.py    legge study_C.toml in dataclass tipizzate
    data.py      carica i tre CSV e li VALIDA (profilo dei dati)
    splits.py    split cronologico train/val/test sui time step
    features.py  insiemi di feature (local / all), scaler e pesi di classe fit SOLO sul train
    graph.py     costruisce il grafo PyG visibile a ogni fase (niente archi dal futuro)
    models.py    Random Forest, MLP, GraphSAGE, GAT
    train.py     cicli di addestramento con early stopping sulla validazione
    evaluate.py  soglia scelta sulla validazione, metriche sulla classe illecita
    reporting.py salva tabelle e metadati del run
    figures.py   figure dai CSV salvati
    pipeline.py  orchestra tutto quanto sopra
    main.py      punto di ingresso (nessuna CLI: tutto da study_C.toml)
"""

# Incrementa questo numero ogni volta che cambi il modo in cui i dati grezzi vengono
# trasformati (colonne, filtri, split): finisce nei metadati di ogni run.
PREPROCESSING_VERSION = "0.1"

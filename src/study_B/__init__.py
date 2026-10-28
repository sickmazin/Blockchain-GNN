"""Study B — temporal link prediction of NEW directed transfer edges on one ERC-20 network.

Flow (one module per step, orchestrated by ``pipeline.run``):

    data.py       deduplicated events of the token (Study A cache) -> remove mint/burn and self-loops
    splits.py     disjoint calendar windows -> phase (history-only / train / val / test) per window
    samples.py    for each target window t: history graph (< t), positives (new pairs in t), negatives
    features.py   historical-only node features for each target window, scaler fit on train
    baselines.py  preferential attachment, common neighbours, Node2Vec (dot-product scorer)
    gnn.py        GraphSAGE / GAT with sent_to + received_from message passing, MLP pair decoder
    evaluate.py   AP (primary), ROC-AUC, MRR, Hits@K
    figures.py    report figures, built only from the saved CSV files

Specification: assistant.md §6.
"""

# Study C figures

## fig01_labels_by_step: Labelled transactions per time step

Top: licit and illicit labelled transactions per time step (unknown nodes omitted). Bottom: illicit share of labelled transactions. Dashed lines mark the chronological split (train 1-34, validation 35-39, test after).

- Split: all steps
- Files: fig01_labels_by_step.png
- Sources: data/label_counts_by_step.csv

## fig02_test_metrics: Test performance by model and feature set

Illicit-class PR-AUC and F1 on the test steps, mean over seeds; error bars = one standard deviation (none for the single-seed random forest). The F1 threshold is chosen on validation and applied unchanged. Dashed: prevalence (PR-AUC of a random ranking). Dotted: degree-only baseline (score = node degree in the visible graph).

- Split: test
- Files: fig02_test_metrics.png
- Sources: summary/metrics_summary.csv

## fig03_per_step_f1: Illicit F1 per test time step

Illicit F1 on each test time step with the validation-chosen threshold, mean over seeds (band = ± one standard deviation). Top: number of illicit transactions per step; F1 is undefined (gap) where a step has no illicit transaction and no alarm.

- Split: test
- Files: fig03_per_step_f1.png
- Sources: summary/per_step_summary.csv; data/label_counts_by_step.csv

## fig04_confusion_test: Test confusion matrices

Confusion matrices on the test steps at the validation-chosen threshold. Cells show counts and the share of the actual class (row); shading = that share. For multi-seed models the seed with the median validation PR-AUC is shown.

- Split: test
- Files: fig04_confusion_test.png
- Sources: summary/metrics_all.csv; models/random_forest__local__seed42/metrics.json; models/mlp__local__seed3/metrics.json; models/graphsage__local__seed6/metrics.json; models/gat__local__seed3/metrics.json; models/random_forest__all__seed42/metrics.json; models/mlp__all__seed42/metrics.json; models/graphsage__all__seed6/metrics.json; models/gat__all__seed6/metrics.json

## fig06_homophily_by_phase: Label homophily along edges, per phase

Share of illicit neighbours for illicit and for licit transactions, on edges whose two endpoints are both labelled (read as undirected; unknown endpoints excluded). The dashed segment is the share expected without homophily; r is Newman's label assortativity (0 = random, 1 = only same-class edges). Only the training phase informs design choices; validation and test are descriptive.

- Split: train / validation / test (descriptive)
- Files: fig06_homophily_by_phase.png
- Sources: data/edge_homophily.json

## fig07_homophily_by_step: Label assortativity per time step

Top: Newman's label assortativity of the edges inside each time step, on edges with both endpoints labelled (gap = step without illicit endpoints). Bottom: number of such edges with at least one illicit endpoint, i.e. the evidence behind each value; steps with few edges give noisy r (hollow markers: fewer than 20). Dashed lines mark the chronological split; values outside the training steps are descriptive only.

- Split: all steps (descriptive)
- Files: fig07_homophily_by_step.png
- Sources: data/edge_homophily_by_step.csv

## fig05_learning_mlp: MLP learning curves

Full-batch training: one optimiser step per epoch. Top: class-weighted BCE on labelled training nodes. Bottom: validation PR-AUC used for early stopping; the marker is the best epoch, whose weights are kept. One line per seed.

- Split: train (loss) / validation (PR-AUC)
- Files: fig05_learning_mlp.png
- Sources: models/mlp__local__seed3/history.csv; models/mlp__local__seed6/history.csv; models/mlp__local__seed12/history.csv; models/mlp__local__seed24/history.csv; models/mlp__local__seed42/history.csv; models/mlp__all__seed3/history.csv; models/mlp__all__seed6/history.csv; models/mlp__all__seed12/history.csv; models/mlp__all__seed24/history.csv; models/mlp__all__seed42/history.csv

## fig05_learning_graphsage: GraphSAGE learning curves

Full-batch training: one optimiser step per epoch. Top: class-weighted BCE on labelled training nodes. Bottom: validation PR-AUC used for early stopping; the marker is the best epoch, whose weights are kept. One line per seed.

- Split: train (loss) / validation (PR-AUC)
- Files: fig05_learning_graphsage.png
- Sources: models/graphsage__local__seed3/history.csv; models/graphsage__local__seed6/history.csv; models/graphsage__local__seed12/history.csv; models/graphsage__local__seed24/history.csv; models/graphsage__local__seed42/history.csv; models/graphsage__all__seed3/history.csv; models/graphsage__all__seed6/history.csv; models/graphsage__all__seed12/history.csv; models/graphsage__all__seed24/history.csv; models/graphsage__all__seed42/history.csv

## fig05_learning_gat: GAT learning curves

Full-batch training: one optimiser step per epoch. Top: class-weighted BCE on labelled training nodes. Bottom: validation PR-AUC used for early stopping; the marker is the best epoch, whose weights are kept. One line per seed.

- Split: train (loss) / validation (PR-AUC)
- Files: fig05_learning_gat.png
- Sources: models/gat__local__seed3/history.csv; models/gat__local__seed6/history.csv; models/gat__local__seed12/history.csv; models/gat__local__seed24/history.csv; models/gat__local__seed42/history.csv; models/gat__all__seed3/history.csv; models/gat__all__seed6/history.csv; models/gat__all__seed12/history.csv; models/gat__all__seed24/history.csv; models/gat__all__seed42/history.csv

# MISTIC validation studies

This directory contains the consolidated synthetic-data validation notebook,
its supporting helper scripts, and generated result tables and figures.

``Synthetic100_svmSet_benefit.ipynb`` is the canonical validation notebook. It
compares forward- and backward-knee MISTIC selection across ensemble sizes and
includes the necessary reference models: an all-feature RBF SVM, an SVM with
recursive feature elimination, and SVMs fitted to MISTIC's unified features.
It also reports blind-set performance, known-signal recovery, member
disagreement, and stability across inner seeds.

Exploratory and intermediate validation notebooks live under the locally
ignored ``experimentation`` directory and are intentionally not distributed or
tracked in the release repository. Their associated result tables are archived
there as well. Only six CSV files remain here: four inputs required by the
consolidated notebook and two inputs required by the documentation's measured
comparison figures.

These studies are kept separate from `mistic/examples`, which contains the
smaller user-facing package examples. Run the notebook from the repository root
so its references to ``validation/*.csv`` resolve correctly.

# OOD Generalization on Biological Sequences

A study of whether unsupervised domain-adaptation and semi-supervised methods
help a regression model predict far-OOD biological sequences when only labelled
in-distribution and unlabelled near-OOD data are available — and whether such a
model can be selected without using far-OOD labels.


## Research question

Two coupled questions:

1. **Q1.** Given oracle access to far-OOD labels for model selection, how much
   do adaptation methods improve over a tuned ERM baseline?
2. **Q2.** Using only labels actually available at deployment (ID or near-OOD),
   how much of that improvement survives?

The headline number is **Q2 − Q1**, the *model-selection penalty*.

## Datasets

- **GFP** — protein fluorescence (ProteinGym)
- **AAV** — capsid fitness (ProteinGym)
- **TFBind8** — DNA transcription-factor binding (Design-Bench)

Each dataset is split along Hamming distance to the wild-type sequence into
disjoint bands: `train` (ID), `val_id` (ID holdout), `val_ood` (near-OOD),
`target_close` (mid-OOD, unlabelled adaptation), `target_test` (far-OOD oracle
selection), and `test` (far-OOD report only). `target_test` and `test` are
disjoint, enforced by a test.

## Methods implemented

All built on top of six encoders (MLP, CNN, LSTM, CNN-LSTM, LSTM-CNN,
Transformer).

| Method | Paper |
|---|---|
| ERM | source-only baseline |
| AdaBN | Li et al. 2018 |
| CMD | Zellinger et al. 2017 |
| Pseudo-Labelling | Lee 2013, with the uncertainty-gated variant from Rizve et al. 2021 (UPS) |
| Mean Teacher | Tarvainen & Valpola 2017 |
| FixMatch | Sohn et al. 2020 |

## Pipeline

The notebook `pipeline.ipynb` runs the whole study in five phases:

| Phase | What it does |
|---|---|
| 0 | Build mutation-distance splits + preprocess |
| 1 | ERM baseline search per architecture + normalization check |
| 2 | Per-method hyperparameter search → matched-oracle scored runs → oracle prune |
| 3 | Survivors run extrapolatively across the three selection modes × seeds |
| 4 | Compute Q1 (oracle MAE), Q2 (val_ood MAE), and the penalty Q2 − Q1 |
| 5 | Headline table + per-dataset penalty chart |

Every method is built on the same tuned encoder per (model, dataset), so each
method-vs-ERM comparison is on identical backbones.

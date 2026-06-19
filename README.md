# OOD Generalization on Biological Sequences

An empirical study of whether unsupervised domain-adaptation (UDA) and
semi-supervised methods help a regression model predict the fitness of
out-of-distribution biological sequences, when the only labels available are
in-distribution, and when the method and model must be chosen **without** looking
at OOD labels.

## The question

Sequences drift away from a training distribution by mutation. Given labelled
in-distribution data and *unlabelled* shifted data, two things are unclear:

1. Do adaptation methods actually beat a well-tuned ERM baseline as the test shift
   grows, or does the gain vanish far from the training distribution?
2. Can the better method/model be picked using only an in-distribution validation
   set (the honest, deployable setting), rather than peeking at OOD labels?

Every result here is produced under **honest model selection**: early stopping and
checkpoint selection always monitor in-distribution validation MAE (`val_id`),
following the DomainBed rule. OOD labels are used only to *report* final numbers,
never to select.

## Quickstart

```bash
uv sync --extra dev                 # build the environment (torch + lightning, pinned via uv.lock)
uv run python pipeline_phases.py    # run the pipeline in the mode set at the top of the file
```

The run mode is a single switch (`MODE`) near the top of `pipeline_phases.py`:
`tutorial` (fast, debug data) · `gfp_sanity` (full gfp, real search) · `real` (full study).
Outputs land in `results_phases/` (see [Outputs](#outputs)).

## Datasets and the shift axis

| Dataset | Signal | Source |
|---|---|---|
| GFP | protein fluorescence | ProteinGym |
| AAV | capsid fitness | ProteinGym |
| TFBind8 | DNA TF-binding | Design-Bench |

Each dataset is partitioned by Hamming distance to the wild-type into three bands,
then into the pools the protocols consume:

- **ID:** `train` (labelled, supervised) and `val_id` (labelled, selection only).
- **close** (mid-shift): split into `U_close` (unlabelled adaptation) and `T_close` (test).
- **far** (large shift): split into `U_far` (unlabelled adaptation) and `T_far` (test).

## The five protocols

Each protocol is an `(adaptation pool, test pool)` pair. All share the same labelled
`train` and the same `val_id` selection signal, so they differ only in *what shifted
data the method adapts on* and *what it is tested on*.

| Protocol | Adapt on | Test on | Isolates |
|---|---|---|---|
| E1 | `T_close` | `T_close` | transductive, close |
| E2 | `U_close` | `T_close` | inductive, close |
| E3 | `U_close` | `T_far`  | **extrapolation** (adapt close, test far) |
| E4 | `U_far`  | `T_far`  | inductive, far |
| E5 | `T_far`  | `T_far`  | transductive, far |

Phase G turns these into the headline quantities, each averaged over seeds:

- `lift_close = ERM_close − E2`, `lift_far = ERM_far − E4` — improvement over the ERM baseline.
- `G_extrap = E3 − E4` — the extrapolation gap (adapting on close vs far for a far test).
- `D_shift = E3 − E2` — degradation from a close to a far test under close adaptation.
- `delta_close = E2 − E1`, `delta_far = E4 − E5` — the transductive/inductive gap.

## Methods

Each method is trained on the **same tuned encoder per (architecture, dataset)**, so
every method-vs-ERM comparison is on identical backbones. Encoders: MLP, CNN, LSTM,
CNN-LSTM, LSTM-CNN, Transformer.

| Method | Reference | Adaptation signal |
|---|---|---|
| ERM | source-only baseline | none |
| AdaBN | Li et al. 2018 | recompute BatchNorm statistics on the target |
| CMD | Zellinger et al. 2017 | match central moments of source/target features |
| Pseudo-Labelling | Lee 2013; UPS gate, Rizve et al. 2021 | MC-dropout-confident pseudo-labels |
| Mean Teacher | Tarvainen & Valpola 2017 | EMA-teacher consistency |
| FixMatch | Sohn et al. 2020 | weak-view pseudo-label, strong-view consistency |

The pseudo-label and FixMatch methods are regression adaptations: there is no softmax
confidence, so MC-dropout predictive variance (Gal & Ghahramani 2016) provides the
confidence signal, and Gaussian input noise replaces image augmentation.

## Pipeline

`pipeline_phases.py` is the sole orchestrator. Each phase reads the committed outputs
of the previous one, so the run is resumable.

| Phase | Function | What it does |
|---|---|---|
| A | `phase_A` | build the band splits (`T_close`/`U_close`/`T_far`/`U_far`) and preprocess |
| B | `phase_B` | per-architecture ERM encoder search + a normalization run-off, one tuned backbone per (architecture, dataset) |
| C | `phase_C` | ERM baseline reference (close + far) |
| D | `phase_D` | top-K architecture cut per dataset |
| E | `phase_E` | per-method hyperparameter search (on E2), winners written to `config/tuned/` |
| F | `phase_F` | run the five protocols with the committed hyperparameters, over seeds |
| G | `phase_G` | aggregate over seeds and compute the headline quantities |

Search winners are committed to `config/tuned/` (`models.yaml`, `methods.yaml`) for
reproducibility. Authored configuration lives in `config/` (see `config/README.md`).

## Outputs

| File | Contents |
|---|---|
| `phaseB_encoder_search.csv` / `phaseB_all_trials.csv` | per-(architecture, dataset) winner / every search trial |
| `phaseC_baseline_reference.csv` | ERM baseline MAE, close and far |
| `phaseE_method_search.csv` | per-method search log |
| `phaseF_all_protocols.csv` | one row per (model, method, protocol, seed) |
| `phaseF_by_mut_dist.csv` | the same, broken down by mutation distance |
| `phaseG_analysis.csv` | seed-aggregated headline quantities |
| `phaseG_by_mut_dist.csv` | per-mutation-distance curves with across-seed error bars |

## Repository layout

```
pipeline_phases.py         orchestrator (phases A-G)
config/                    authored configs + tuned/ search winners
src/bioseq_ood/
  data/                    dataset loading, band splits, preprocessing
  models/                  encoders + normalization
  methods/                 method specs and hyperparameter sampling
  training/                Lightning module, trainer, search, selection
  evaluation/              metrics, per-mutation-distance breakdown
tests/                     unit tests (uv run pytest)
```

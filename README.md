# HyperSPEAR

Codebase for the hypergraph backdoor attack experiments used in the paper.

## Overview

This project trains a backdoor trigger generator on hypergraph data and evaluates attack success rate (ASR) and clean accuracy (CA) under different defenses and HGNN backbones.

Main entry points:

- `main.py`: standard attack evaluation with optional `none`, `prune`, or `reconstruct` defense modes
- `main_rigbd.py`: evaluation pipeline for the RIGBD setting

## Environment

The project was tested with:

- Python 3.10+
- PyTorch 2.2.1
- CUDA 11.8
- `torch-geometric`
- `dhg`
- `ogb`
- `deeprobust`

You can install the main dependencies with:

```bash
bash install.sh
```

## Data Format

Each dataset should be placed under `data/<dataset_name>/` and contain:

- `H.pt`: hyperedge incidence pairs as a `torch.Tensor` of shape `[2, num_pairs]`
- `X.pt`: node feature matrix
- `Y.pt`: node labels

Example:

```text
data/
  cora_cite/
    H.pt
    X.pt
    Y.pt
```

At the moment, the repository includes an example dataset folder: `data/cora_cite/`.

## Quick Start

Run the main attack pipeline:

```bash
python main.py --dataset cora_cite --model HyperSAGE --test_model HyperSAGE --defense_mode none --device_id 0
```

Run the RIGBD evaluation pipeline:

```bash
python main_rigbd.py --dataset cora_cite --model HyperGCN --test_model HyperGCN --device_id 0
```

Useful arguments:

- `--dataset`: dataset name, e.g. `cora_cite`
- `--model`: surrogate/backdoor training model
- `--test_model`: victim HGNN used for evaluation
- `--defense_mode`: `none`, `prune`, or `reconstruct`
- `--vs_number`: number of poisoned nodes
- `--alpha_int`: number of perturbed feature dimensions
- `--device_id`: CUDA device id

## Output

The scripts print the main metrics to the console:

- `ASR`: attack success rate on triggered target nodes
- `CA`: clean accuracy on clean test nodes

`main.py` also writes timestamped logs to `logs/`.

## Project Structure

```text
HyperSPEAR/
  data/                 input hypergraph datasets
  models/               HGNN backbones and backdoor modules
  sage_modified/        local SAGE utilities
  main.py               main attack pipeline
  main_rigbd.py         RIGBD evaluation
  select_feature.py     feature selection
  select_sample.py      poison node selection
  utils.py              data loading and split utilities
  help_funcs.py         defense and pruning helpers
```

## Notes

- Dataset names passed to the scripts should match the folder names under `data/`.
- For reproducible runs, keep the default random seed unless you want to study variance across runs.

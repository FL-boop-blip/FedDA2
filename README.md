**FedDA2: Federated Learning with Dual Adaptive Mechanisms for Accelerated Primal–Dual Optimization**

FedDA2 implements a federated learning simulator with the FedDA2 algorithm and a set of commonly used baselines. The code supports IID and non‑IID data partitions, multiple datasets, and a variety of CNN/text models. Results (accuracy, loss, divergence, time) are saved for each communication round.

Key components:
- Dual adaptive server updates with local dual correction and diversity-aware scaling (FedDA2 logic in `server/FedDA2.py:1`).
- Sharpness-aware local optimization via ESAM (an SAM variant) in clients (`optimizer/ESAM.py:1`, `client/fedda2.py:1`).
- Optional late-phase distillation variant `fedda2_t` with KL loss (`client/fedda2_t.py:1`, `utils.py:1`).
- Baseline algorithms and utilities organized under `server/`, `client/`, `optimizer/`.

This repository is intended for reproducible simulation of FL algorithms with configurable client populations and data heterogeneity.

**Repository Structure**
- `train.py:1` — Main entry point; parses CLI args, builds datasets/models, selects a server, runs training.
- `server/` — Server-side algorithms and training loop:
  - `server/server.py:1` base server with orchestration, logging, evaluation, and IO.
  - `server/FedDA2.py:1` FedDA2 implementation (dual variables, gradient diversity, ESAM clients).
  - Additional baselines: `FedAvg.py`, `FedDyn.py`, `FedSMOO.py`, `FedSpeed.py`, `FedTOGA.py`, `FedVRA.py`.
- `client/` — Client-side local training implementations:
  - `client/client.py:1` base client with SGD local updates.
  - `client/fedda2.py:1` FedDA2 client using ESAM; `client/fedda2_t.py:1` with distillation.
- `optimizer/` — Optimizers and SAM-family variants used by clients (e.g., `ESAM.py:1`, `SAM.py`, `LESAM*.py`).
- `dataset.py:1` — Dataset preparation (MNIST, CIFAR10/100) and IID/non‑IID partitioning (Dirichlet, Pathological).
- `models.py:1` — Model zoo (e.g., LeNet, ResNet18 variants).
- `run.sh:1`, `run.bat:1` — Example launchers.

**Installation**
- Python 3.8+ recommended.
- Install the required dependencies:
  - `pip install -r requirements.txt`
- A CUDA-enabled PyTorch build is recommended for speed; CPU is supported but slower.

**Datasets**
- MNIST, CIFAR10, CIFAR100: downloaded via `torchvision` on first run to `./Data/Raw` by default (`dataset.py:240+`).
- Non‑IID splits:
  - Dirichlet: `--non-iid --split-rule Dirichlet --split-coef 0.1|0.3|0.6|1.0`.
  - Pathological: `--non-iid --split-rule Pathological --split-coef 3|5`.

**Quick Start**
- Install dependencies and run the provided script:
  - `bash run.sh`
  - or on Windows: `run.bat`

**Direct Usage**
- Typical CIFAR‑10 (IID) baseline:
  - `python train.py --method FedAvg --dataset CIFAR10 --model ResNet18 --comm-rounds 300 --local-epochs 5 --batchsize 50`
- CIFAR‑10 (non‑IID Dirichlet 0.6, 100 clients, 10% participate):
  - `python train.py --method FedAvg --dataset CIFAR10 --model ResNet18 --non-iid --split-rule Dirichlet --split-coef 0.6 --total-client 100 --active-ratio 0.1 --comm-rounds 300 --local-epochs 5 --batchsize 50`

**Running FedDA2**
- FedDA2 is implemented in `server/FedDA2.py:1` and clients `client/fedda2*.py`. To run it, set `--method FedDA2`.
- If `FedDA2` is not yet listed in the `--method` choices in your snapshot of `train.py`, add it to the parser choices and dispatch (search for `--method` and the method mapping in `train.py:120+`).

**Important Arguments (subset)**
- `--dataset` one of `mnist|CIFAR10|CIFAR100` (`train.py:23`).
- `--model` one of `mnist_2NN|ResNet18` (`train.py:24`).
- `--method` federated algorithm; see `train.py:120+` and `server/__init__.py:1` for available servers.
- `--total-client`, `--active-ratio`, `--comm-rounds`, `--local-epochs`, `--batchsize` to shape the simulation.
- `--non-iid`, `--split-rule`, `--split-coef` to control heterogeneity.
- Optimizer/regularization knobs (used by some methods): `--weight-decay`, `--local-learning-rate`, `--rho` (SAM/ESAM), `--lamb`, etc. (see `train.py:40+`).

**Outputs and Logging**
- Results are saved under `out/<METHOD>/T=<ROUNDS>/<DATASET>-<SPLIT>-<CLIENTS>/active-<RATIO>/` (`server/server.py:24`).
  - Performance: `Performance/tst-<METHOD>.npy` (loss, acc per round).
  - Divergence: `Divergence/divergence-<METHOD>.npy` (consistency metric).
  - Time: `Time/time-<METHOD>.npy` (seconds per round).

**Notes**
- Mixed precision TF32 is enabled on A100 by default (`train.py:10`).
- Some optional client variants import `thop` for profiling; this is included in requirements.



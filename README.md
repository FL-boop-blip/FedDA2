**FedDA2: Federated Learning with Dual Adaptive Mechanisms for Accelerated Primal–Dual Optimization**

FedDA2 implements a federated learning simulator with the FedDA2 algorithm and a set of commonly used baselines. The code supports IID and non‑IID data partitions, multiple datasets, and a variety of CNN/text models. Results (accuracy, loss, divergence, time) are saved for each communication round.



**Installation**
- Python 3.8+ recommended.
- Install the required dependencies:
  - `pip install -r requirements.txt`
- A CUDA-enabled PyTorch build is recommended for speed; CPU is supported but slower.

**Datasets**
- MNIST, CIFAR10, CIFAR100: downloaded via `torchvision` on first run to `./Data/Raw` by default.
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
- `--dataset` one of `mnist|CIFAR10|CIFAR100`.
- `--model` one of `mnist_2NN|ResNet18` .
- `--method` federated algorithm; see `train.py:120+` and `server/__init__.py:1` for available servers.
- `--total-client`, `--active-ratio`, `--comm-rounds`, `--local-epochs`, `--batchsize` to shape the simulation.
- `--non-iid`, `--split-rule`, `--split-coef` to control heterogeneity.
- Optimizer/regularization knobs (used by some methods): `--weight-decay`, `--local-learning-rate`, `--rho` (SAM/ESAM), `--lamb`, etc.

**Outputs and Logging**
- Results are saved under `out/<METHOD>/T=<ROUNDS>/<DATASET>-<SPLIT>-<CLIENTS>/active-<RATIO>/` (`server/server.py:24`).
  - Performance: `Performance/tst-<METHOD>.npy` (loss, acc per round).
  - Divergence: `Divergence/divergence-<METHOD>.npy` (consistency metric).
  - Time: `Time/time-<METHOD>.npy` (seconds per round).

**Notes**
- Mixed precision TF32 is enabled on A100 by default (`train.py:10`).
- Some optional client variants import `thop` for profiling; this is included in requirements.



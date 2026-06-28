# INFL-Spatial Transcriptomics 🧬🗺️🔐

## Overview 📚

This task adds the spatial transcriptomics workflow from PPML-Omics into the INFL-Pytorch layout. It supports federated histology-to-transcriptomics regression with normal, DP, PPML, and INR variants, plus visualization and gradient inversion attack analysis.

- Task: Spatial transcriptomics gene-expression prediction from histology patches 🧫➡️📈
- Backbone options: LeNet and ST-Net-style DenseNet backbones 🧠
- Privacy settings: federated learning, DP/PPML noise calibration, INR protection 🔐
- Main outputs: prediction `.npz` files, FASN packed heatmaps, iDLG/GIA reconstruction metrics 📊

---

## Installation 🧰

Follow the PPML-Omics environment setup for the spatial transcriptomics workflow:

```bash
conda create -n ppmlomics python=3.9
conda activate ppmlomics
conda install mamba -y
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
mamba install seaborn -y
mamba install matplotlib -y
mamba install tqdm -y
mamba install scikit-learn -y
mamba install -c conda-forge scipy -y
mamba install numpy=1.20.3 -y
mamba install -c conda-forge umap-learn -y
mamba install -c conda-forge openslide-python -y
pip install tenseal
```

Verify the task scripts:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n ppmlomics python -m py_compile src/train.py
```

Core Python packages are also listed in `requirements.txt`. `openslide-python` requires the system OpenSlide shared library to be installed. ✅

---

## Data 📦

The scripts expect the following task-local data layout:

```text
spatial_transcriptomics/data/hist2tscript
spatial_transcriptomics/data/hist2tscript-patch
```

If the PPML-Omics data already exists, symlink or copy these directories from:

```text
/deltadisk/zhangrongyu/PPML-Omics-master/Application/SpatialTranscriptomics/data
```

Fixed test patients used by the reference experiments:

```text
BT23269,BT23277,BT23288,BT23377,BT23901,BT23944
```

---

## Tutorial ▶️

Open the notebooks under `tutorial/`:

```bash
cd tutorial
```

- `training.ipynb`: INR federated training for normal, DP, PPML, and INR methods.
- `analysis.ipynb`: prediction visualization, LeNet iDLG, and checkpoint-based GIA commands.
- `attack_visualization.ipynb`: activation-FiLM INR-only GIA reproduction and visualization for the 9000-iteration progress-grid run.

---

## Quick Start 🚀

Run activation-dependent FiLM INR training:

```python
from src.pipeline_utils import TrainingConfig, run_training

config = TrainingConfig(
    out_dir="spatial_fed_e5_c30_fixedsplit_stable",
    methods="normal,dp,ppml,inr",
    epochs=5,
    clients=30,
    batch_size=16,
    eval_batch_size=64,
    num_workers=4,
    lr=1e-5,
    epsilon=10,
    delta=1e-3,
    l2_clip=1,
    key_strength=1.0,
)
run_training(config)
```

Visualize predictions and FASN packed heatmaps:

```python
from src.pipeline_utils import VisualizationConfig, run_visualization

config = VisualizationConfig(
    root="spatial_fed_e5_c30_fixedsplit_stable",
    outdir="evaluation_spatial_fed_e5_c30_fixedsplit_stable",
    epoch="5",
    gene="FASN",
    methods="normal,dp,ppml,inr",
)
run_visualization(config)
```

Run LeNet iDLG on selected samples:

```python
from src.pipeline_utils import AttackConfig, run_attack

config = AttackConfig(
    expname="iDLG_attack_spatial",
    device="cuda:0",
    samples="0,50,100",
    methods="baseline,correct,wrong,without",
    iterations=901,
    key_strength=1.0,
)
run_attack(config)
```

Run checkpoint-based GIA:

```python
from src.pipeline_utils import CheckpointGIAConfig, run_checkpoint_gia

config = CheckpointGIAConfig(
    outdir="evaluation_example_fixedsplit/checkpoint_gia_s50",
    samples="0,50,100",
    iterations=300,
    param_scope="classifier",
    methods="all",
)
run_checkpoint_gia(config)
```

Run the activation-FiLM INR-only GIA visualization run:

```python
from src.pipeline_utils import AttackVisualizationConfig, run_attack_visualization

config = AttackVisualizationConfig(
    out_dir="evaluation_example_fixedsplit/gia_fed_st_demo_leaky_lenet_e5_train_sample0_key10_dpeps236_ppmleps129_i9000_progress100_fairinit_lastlinear_activation_film_inr_only",
    key_strength=10.0,
    iters=9000,
    tv_weight=3e-4,
    param_scope="last_linear",
    restarts=3,
    save_intermediate_every=100,
)
run_attack_visualization(config)
```

---

## Important Files 🧩

| file | purpose |
|---|---|
| `src/pipeline_utils.py` | Notebook-friendly configuration objects and run functions for training, visualization, iDLG, and GIA. |
| `src/train.py` | Low-level activation-dependent FiLM INR federated training CLI. |
| `src/simulation_core.py` | Shared ST dataset/model utilities used by training and checkpoint GIA. |
| `src/attack.py` | Low-level LeNet definitions and iDLG attack CLI. |
| `src/attack_core.py` | Shared original ST attack/data utilities used by the attack workflow. |
| `src/visualize.py` | Low-level prediction metrics and packed FASN heatmap visualization CLI. |
| `src/checkpoint_gia.py` | Low-level checkpoint-based GIA CLI with clipping and released-gradient noise support. |
| `src/inr_gia.py` | Low-level activation-FiLM INR-only GIA reproduction CLI. |
| `src/gia_core.py` | Shared GIA helpers adapted to this task layout. |

---

## Tips & Troubleshooting 🛠️

- Missing data errors usually mean `data/hist2tscript-patch` or `data/hist2tscript` is absent from this task directory.
- CUDA import errors are environment-specific; use `ppmlomics` rather than the base environment.
- `openslide` import errors require installing both `openslide-python` and the system OpenSlide shared library.
- Very large DP/PPML noise can make spatial prediction MSE much worse; check `summary.json` and `metrics.md` before reporting a run.

# INFL-Cell Perturbation 🧬⚙️🚀

## Overview 📚
This section focuses on perturbation prediction — a regression task modeling the mapping from control gene expression and perturbation conditions to post-perturbation gene expression. This repository provides the implementation for training INFL on triplet datasets to predict gene expression under unseen perturbations. We evaluate the framework on widely used Adamson and Norman datasets, plus an in-house private dataset, representing diverse experimental conditions. ✅

- Task: Single-cell perturbation outcome prediction (regression) 🧫➡️📈  
- Datasets: Norman, Adamson, and in-house custom data 📦  
- Federated INR: Communication-efficient training with Implicit Neural Representation module for stronger privacy preservation 🔐

---

### Installation 🧰

Install PyG, then install GEARS API:

- PyG minimal install (PyTorch already installed):
  - pip install torch_geometric

- Install additional accelerated ops (optional but recommended for speed):  
  Please match your Torch and CUDA versions accordingly:
  - Check Torch:
    - python -c "import torch; print(torch.__version__)"
  - Check CUDA for Torch:
    - python -c "import torch; print(torch.version.cuda)"
  - Install wheels (example for Torch 2.6.0 + CUDA 12.6):
    - pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html

Official installation guide: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Then install GEARS (API used for perturbation data I/O and training loop utilities):
- pip install cell-gears

Tips:
- Start with minimal PyG (torch_geometric) and only add extras if needed. 🧩  
- For CPU-only or mismatched CUDA, pick the appropriate wheels per the PyG docs. 💡

---

### Start training ▶️

```bash
cd tutorial/
```

Set the hyperparameters:

```python
args = FLArgs(
    gpu=0,                # Set your GPU id (int) or use CPU if unavailable
    n_client=10,          # Number of federated clients
    dataset='norman',     # 'norman' or 'adamson'
    global_round=200,     # Communication rounds for FL
    local_epoch=2,        # Local epochs per round
    frac=0.5,             # Fraction of clients participating each round
    lr=1e-3,              # Learning rate
    inr=True,             # Enable INR module
    alpha=0.3,            # Weight of INR module in the objective
    data_dir='./data',    # Data directory (GEARS-compatible)
    batch_size=32,        # Mini-batch size
    seed=1,               # Random seed
    out_dir_prefix='INFL' # Output directory prefix
)
```

---

### Use your own dataset 📦

Prepare a Scanpy AnnData object with:
- adata.var['gene_name']  
- adata.obs['condition'], adata.obs['cell_type']

Then process and load:

```python
pert_data.new_data_process(dataset_name='XXX', adata=adata)
# to load the processed data
pert_data.load(data_path='./data/XXX')
```

We also provide an example using our preprocessed in-house data.

---

### Customizing pipeline loading 🔧

Substitute the following code in `src/pipeline_utils.py`:

```python
pdata.load(data_name=args.dataset)
pdata.prepare_split(split="simulation", seed=args.seed)
pdata.get_dataloader(batch_size=args.batch_size)
```

with:

```python
pdata.load(data_path='/path/to/your/data')  # directory of your processed GEARS data
pdata.prepare_split(split='custom', split_dict_path='custom_split.pkl')  # use your own split
pdata.get_dataloader(batch_size=32, test_batch_size=128)  # prepare data loaders
```

Notes:
- custom_split.pkl should define train/val/test indices per GEARS format.  
- Ensure gene_name/condition/cell_type fields are present and consistent. ✅

---

## Tips & Troubleshooting 🛠️

- PyG install errors (undefined symbol: make_function_schema):
  - Likely Torch/CUDA wheels mismatch. Reinstall with the exact versions per PyG docs.
  - Use pip --force-reinstall --no-cache-dir and verify versions via:
    - python -c "import torch; print(torch.__version__)"
    - python -c "import torch; print(torch.version.cuda)"
    - nvcc --version (if using system CUDA)

- Dataloader splits:
  - For reproducibility, fix the seed and cache split_dict (custom_split.pkl). 📌

- Performance:
  - Increase local_epoch for better on-client convergence.
  - Tune frac and n_client to simulate participation variability.
  - INR on (inr=True) can stabilize across heterogeneous clients. 🌐

---

Made with ❤️ for privacy-preserving single-cell analysis. 🔐🧫📈
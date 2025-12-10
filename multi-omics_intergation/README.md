# SpaMosaic: mosaic integration of spatial multi-omics
## Overview
With the advent of spatial multi-omics, we can mosaic integrate such datasets with partially overlapping modalities to construct higher dimensional views of the source tissue. SpaMosaic is a spatial multi-omics mosaic integration tool that employs contrastive learning and graph neural networks to construct a modality-agnostic and batch-corrected latent space suited for analyses like spatial domain identification and imputing missing omes. 

## Installation
We tested our code on a server running Ubuntu 18.04.5 LTS, equipped with 4 NVIDIA A6000 GPUs. The installation process typically takes 10–15 minutes.
```
git clone https://github.com/JinmiaoChenLab/SpaMosaic.git
cd SpaMosaic
conda create -n SpaMosaic python=3.8.8
conda activate SpaMosaic
pip install -r requirements.txt

# install torch
pip install torch==2.1.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html
# install torch_geometrics
pip install torch_geometric==2.4.0 pyg_lib==0.3.1+pt21cu121 torch_scatter==2.1.2+pt21cu121 torch_sparse==0.6.18+pt21cu121 torch_cluster==1.6.3+pt21cu121 torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.1+cu121.html

python setup.py install
```
R package `mclust` is needed to perform clustering and make sure it installed in a R environment.  

## Tutorial
```bash
cd tutorial/{horizontal/mosaic}
```

Set the hyperparameters:

```python
result = run_and_save_mm(
    data_dir='./data/integration/Human_tonsil',
    device='cuda:0',
    num_rounds=50,
    local_epochs=2,
    learning_rate=1e-3,
    inr=True,              # Change to True when using INFL
    inr_hidden_size=8,
    run_name=None,
    seed=1234,
)
```

## Data Availability  
All data used in the experiments are publicly available on [Zenodo](https://zenodo.org/records/16925549). 




# INFL-Cancer Subtyping 🧬🔐🚀

## Overview 📚
We utilized a large-cohort bulk proteomics dataset released by Cai et al., containing 1,207 samples with measurements for 9,105 proteins and corresponding cancer type labels across 14 classes. This dataset was used to evaluate the performance of our federated learning framework under a simulated setup of 5 clients, where the data was partitioned into five equal shards and distributed to each client. The framework demonstrated robust performance, effectively preserving data privacy while achieving results comparable to centralised training. ✅

## Installation 🧰

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/INFL-Pytorch.git
cd cancer_subtyping
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio
```

### 4. Verify installation ✅
```bash
cd src
python config.py
```
This will check if all required files are present and show the configuration summary. 🧪

## Quick Start: Training models with our demo tutorial 🚀

### Configuration ⚙️
All paths and hyperparameters are centralised in `config.py`. Key settings based on the published model:
```python
# Directories
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Cancer types (14 subtypes for ProCan Compendium)
DEFAULT_CANCER_TYPES = [...]  # List of 14 cancer types
```
You can modify these settings by editing `config.py` or by overriding them in your training scripts. ✍️

### Data format 📄
Required files (place in `data/` directory):
- `all_protein_list_mapping.csv`: Protein ID to gene name mapping
- `P10/E0008_P10_protein_averaged_log2_transformed_EB.csv`: ProCan training data
- `P10/replicate_corr_protein.csv`: Sample quality metrics
- `sample_info/sample_metadata_path_noHek_merged_replicates_Adel_EB_2.0_no_Mucinous.xlsx`: Sample metadata

### Tutorial ▶️
```bash
cd tutorial/
```

Set the hyperparameters:

```python
args = Args(
    use_inr=True,              # Enable INR module
    device='cuda:0',           # Set your GPU id (int) or use CPU if unavailable
)

consts = RunConstants(
    N_clients=5,               # Number of federated clients
    N_included_clients=5,      # Number of clients participating each round
    N_iters=100,               # Communication rounds for FL
    base_model_dir='./INFL',   # Output directory prefix
)
```

**Data Availability 📦:**  
The raw DIA-MS data and processed data of cohort 1 (ProCan Compendium pan-cancer cohort) and the corresponding spectral library have been deposited in the Proteomics Identification database (PRIDE) under dataset identifier **PXD056810**.

---

## Tips & Troubleshooting 🛠️

### Common Issues

**1) Missing data files 📂❓**  
Error:
```
FileNotFoundError: Protein mapping file not found
```
Fix:
- ✅ Ensure `data/all_protein_list_mapping.csv` exists (and other listed data files).  
- 🧪 Run `python config.py` to validate paths and environment.  
- 📁 Verify relative paths if launching from a different working directory.

---

**2) Feature dimension mismatch 📏⚠️**  
Error:
```
RuntimeError: size mismatch
```
Fix:
- 🧬 Align your input columns to the expected feature list (see `data/data_format.csv`).  
- 🚫 Do not use `--no_align` unless your column order exactly matches the model.  
- 🧹 Ensure the same preprocessing (log2 transform, normalization, missing=0).  
- 🧷 Confirm there are ~8k protein features with the correct UniProt IDs.

---

**3) Reproducibility of splits 🎲📌**  
Issue:
- Different results across runs due to random splits.  
Fix:
- 🧷 Set and document `seed` in config/CLI.  
- 💾 Save your split indices and reuse them across experiments.

---


Made with ❤️ for privacy-preserving bioinformatics. 🧪🔐🌍
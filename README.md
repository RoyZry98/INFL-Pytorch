# Implicit Neural Federated Learning for Privacy-Preserved Biological Analysis ✨🧬🔐

Code and data to reproduce the **INFL** on four **Cancer Subtyping**, **Cell Perturbation**, **Multi-Omics Integration**, and **Spatial Transcriptomics** benchmarking results in our manuscript.

---

## Overview 📚
INFL is a lightweight federated learning framework based on Implicit Neural Representations that addresses heterogeneity, privacy, and scalability in biomedical data integration. By embedding a secret key into its architecture and using coordinate-conditioned modules, INFL ensures strong privacy and seamless aggregation across diverse client models. It demonstrates broad applicability across biomedical omics tasks, including classification in bulk proteomics, regression in single-cell transcriptomics, and clustering in spatial and multi-omics data, while preserving performance for downstream scientific and clinical applications.

- Privacy-preserving: 🔒 Federated + implicit representation with secret key
- Heterogeneity-robust: 🌐 Handles modality/site variation
- Scalable: ⚡ Lightweight model aggregation
- Bio tasks: 🧪 Proteomics (classification), 🧫 scRNA-seq (regression), 🧭 spatial/multi-omics (clustering)

---

## Pseudo-code to Code Mapping Table

| Algorithm 1 step | Code mapping | Notes |
|---|---|---|
| Inputs: clients, local datasets, $\Theta_0$, $T$, $E$, sampling rule, $\lambda$ | `cancer_subtyping/src/pipeline_utils.py::run_federated_loop`, `cell_perturbation/src/pipeline_utils.py::FLArgs` and `run_federated_demo`, `spatial_transcriptomics/src/train.py::main` and `train_method` | `N_iters`, `global_round`, `epochs`/`local_epoch`, `frac`, `clients`, and `key_strength` provide the runtime controls. |
| Generate versioned private coordinate key $\pi_v$ | `cancer_subtyping/src/FedTrain.py::make_coordinates`, `cell_perturbation/src/pipeline_utils.py::make_coordinates`, `multi-omics_intergation/src/framework.py::make_coordinates`, `spatial_transcriptomics/src/attack.py::make_key_inr_coordinates` | Code generates coordinates from seeds/modes; versioning and KMS governance are deployment responsibilities. |
| Instantiate $\pi=\{c_m\}_{m=1}^M$, $M=128$, $d=2$ | `CoordinateKeyEncoder` in `cancer_subtyping/src/FedTrain.py`, `cell_perturbation/src/pipeline_utils.py`, `multi-omics_intergation/src/framework.py`, `spatial_transcriptomics/src/attack.py` | Defaults use 128 coordinate points and 2-D coordinates in the keyed-INR paths. |
| Provision key only to authorized environments; server does not receive key | `coord_seed`, `key_coord_seed`, `wrong_key_coord_seed`, and `set_coordinates(...)` usage in task scripts | The repo models this by local coordinate construction; explicit authentication/key provisioning is not implemented inside PyTorch modules. |
| Server samples participating clients $S_t$ | `cell_perturbation/src/pipeline_utils.py::run_federated_demo`, `cell_perturbation/demo/main_inr.py`, `cancer_subtyping/src/pipeline_utils.py::run_federated_loop`, `spatial_transcriptomics/src/train.py::train_method` | Cell perturbation samples by `frac`; cancer uses KFold simulated clients; spatial iterates the prepared client loaders. |
| Server sends $\Theta_{t-1}$ to clients | `global_model_path` load in `cancer_subtyping/src/pipeline_utils.py`, `gears_client.model.load_state_dict(...)` in `cell_perturbation/src/pipeline_utils.py`, `clone_model(...)` in `spatial_transcriptomics/src/train.py` | Broadcast is represented by loading the latest global checkpoint/state dict into each client model. |
| Authenticate client for key version $v$ | External to current code; related hooks are CLI key seed args and `set_coordinates(...)` methods | Add this in production around job launch/update submission; the current scripts assume authorized local execution. |
| Replace selected linear layers with INRLinear modules | `cancer_subtyping/src/FedTrain.py::FedProtNet_KeyedINR`, `cell_perturbation/src/pipeline_utils.py::replace_linear_with_inr`, `multi-omics_intergation/src/framework.py::replace_linear_with_inr`, `spatial_transcriptomics/src/train.py::replace_final_linear_with_keyed_inr` | Replacement copies unkeyed `nn.Linear` weights/biases into key-conditioned layers where applicable. |
| Train locally using $f_{\pi_v}$ | `cancer_subtyping/src/FedTrain.py::TrainFedProtNet.train_epoch`, `cell_perturbation/src/pipeline_utils.py::run_federated_demo`, `spatial_transcriptomics/src/train.py::local_train`, `multi-omics_intergation/src/framework.py::SpaMosaic.train` | Keyed forward paths call `CoordinateKeyEncoder`, `KeyControlledLinear`, and optional FiLM modules. |
| Return $\Theta_t^k$, $n_k$, key-version metadata $v$ | `local_model_*.pt` files in cancer, `local_w.append(state_dict)` in cell perturbation, `compute_grad_update(...)` in spatial transcriptomics | Current code returns weights/updates; sample-count metadata and key-version tags would need to be added around these return points. |
| Accept only updates with key-version metadata equal to $v$ | Not implemented as a standalone server filter | Practical insertion point is immediately before `WeightsAggregation.fed_avg()`, `fed_avg(local_w)`, or spatial update averaging. |
| Aggregate accepted updates by FedAvg | `cancer_subtyping/src/FedAggregateWeights.py::WeightsAggregation.fed_avg`, `cell_perturbation/src/pipeline_utils.py::fed_avg`, `cell_perturbation/demo/main_inr.py::fed_avg`, `spatial_transcriptomics/src/train.py` update averaging in `train_method` | Existing demos mostly use equal client averaging; add $n_k$ weights here for sample-count-weighted FedAvg. |
| Broadcast $\Theta_t$ without storing/distributing $\pi_v$ | `WeightsAggregation.save_model(...)`, `gears_global.model.load_state_dict(new_glob_w)`, `add_update_to_model(server, aggregated, lr)` | Aggregated parameters are stored or reused; key coordinates remain local model buffers/CLI-derived values, not server-side secrets. |
| Authorized inference with final $\Theta_T$ and $\pi_v$ | `cancer_subtyping/src/FedTrain.py::predict`/`predict_custom`, `cell_perturbation/src/pipeline_utils.py::evaluate_and_log`, `multi-omics_intergation/src/framework.py::infer_emb`/`impute`, `spatial_transcriptomics/src/train.py::evaluate_and_save` | Load the final model with the same approved coordinates to reproduce the authorized effective parameters. |
| No-key or wrong-key path differs from $f_{\pi_v}$ | `forward_without_inr(...)` in keyed modules, `set_coordinates(...)`, `cancer_subtyping/src/run_gia_visualizations.py`, `spatial_transcriptomics/src/inr_gia.py` | These scripts/functions are used for privacy/GIA and wrong-coordinate ablations. |
| Key rotation or revocation | Coordinate seed args such as `--key-coord-seed`, `--wrong-key-coord-seed`, plus `set_coordinates(...)` and the same training scripts | Generate a new coordinate tensor/version, provision it externally, then fine-tune/retrain and reject old-version updates outside the model code. |

## Contents 📁

- `cancer_subtyping/`
  - `src/`: Source code required for cancer subtyping using INFL.
  - `tutorial/`: Jupyter notebooks for a quick start guide and biological analysis of cancer subtyping with INFL.
- `cell_perturbation/`
  - `src/`: Source code required for cell perturbation analysis using INFL.
  - `tutorial/`: Jupyter notebooks for a quick start guide and biological analysis of cell perturbation with INFL.
- `multi-omics_integration/`
  - `src/`: Source code required for multi-omics integration using INFL.
  - `tutorial/`: Jupyter notebooks for a quick start guide and biological analysis of multi-omics integration with INFL.
- `spatial_transcriptomics/`
  - `src/`: Source code required for spatial transcriptomics using INFL and keyed-INR privacy analysis.
  - `tutorial/`: Jupyter notebooks for training, visualization, and gradient inversion analysis.

---

## Quick Start 🚀

1. **Environment**  
   Set up the task-specific environment as required for each method. 🛠️

2. **Run a Method**  
   Open the `{}/tutorial/training.ipynb` notebook corresponding to different dataset and follow the instructions in the README.md file for the specific task. ▶️

3. **Evaluation and Biological Analysis**  
   Use `{}/tutorial/analysis.ipynb` for comprehensive biological analysis as described in our manuscript. 📊🧠

---

## Resources 📦

### Data Availability 🗂️  
The in-house dataset used in our cell perturbation experiments is publicly available on [figshare](https://doi.org/10.6084/m9.figshare.30763670).

- Contents: HUVEC scRNA-seq, YWHAB/YWHAE/YWHAH knockdowns (Sh-B/Sh-E/Sh-H) and control (SCR) 🧬  
- License: CC BY 4.0 ✅

### Codebase (Official Repositories and Tutorials) 🧰  
> Below, we provide the **codebases** used for implementing INFL.

- **ProCanFDL** — [GitHub](https://github.com/CMRI-ProCan/ProCanFDL) 🧪🖥️  
- **GEARS** — [GitHub](https://github.com/snap-stanford/GEARS) 🧠🔧  
- **SpaMosaic** — [GitHub](https://github.com/JinmiaoChenLab/SpaMosaic) 🗺️🧩  

> We also provide the **baseline** implementation used for comparison.

- **PPML** — [GitHub](https://github.com/JoshuaChou2018/PPML-Omics) 🛡️📡

---

## Citing 📖
If you use INFL or related components in your research, please also refer to the original repositories:

- ProCanFDL (Cancer proteomics, federated DL) — “Federated Deep Learning Enables Cancer Subtyping by Proteomics” (Cancer Discovery, 2025) 🧬  
- GEARS (Gene perturbation modeling) — “Predicting transcriptional outcomes of novel multigene perturbations with GEARS” (Nature Biotechnology, 2023) 🧪  
- SpaMosaic (Spatial multi-omics integration) — Spatial integration with contrastive learning + GNNs 🗺️
- PPML-Omics (Privacy-preserving federated omics learning) — “PPML-Omics: a Privacy-Preserving federated Machine Learning method protects patients’ privacy from omic data” 🛡️

---

## Contributing 🤝
- Issues and PRs are welcome! 📨  
- Please follow conventional commit messages and create minimal reproducible examples in bug reports. 🧩

## License 📄
This project is licensed under the MIT License. See the LICENSE file for details.

Made with ❤️ for privacy-preserving bioinformatics. 🧪🔐🌍

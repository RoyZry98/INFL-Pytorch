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

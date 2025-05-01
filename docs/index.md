# CRISGI

Welcome to the official documentation of Charting CRItical tranSient Gene Interactions in Disease Progression from Multi-modal Transcriptomics (**CRISGI**)!

## 🗺️ Overview

![Figure1](./figure/Figure1.png)

Critical transitions (CTs) in gene regulatory networks (GRNs) drive pivotal shifts in disease progression. While CT theory holds great promise for early disease detection, existing computational frameworks face major limitations. They rely on unsupervised ranking of CT signals at individual gene or gene-module level, apply unranked gene set enrichment analyses, and depend on manual inspection of signal trends to infer CT presence and onset within a single cohort. Additionally, multimodal transcriptomic data remain underutilized. These approaches limit mechanistic resolution and hinder clinical translation.

We present CRISGI, a novel CT framework designed to overcome these challenges. CRISGI enables phenotype-specific CT gene-gene interaction modeling, CT-rank enrichment analyses, automated CT presence and onset prediction, and supports bulk, single-cell, and spatial transcriptomic (ST) data.

In viral infections, CRISGI identified 128 CT interactions (128-TER) predictive of symptom status and onset, validated across six external bulk RNA-seq cohorts and COVID-19 scRNA-seq datasets. It further revealed cell-type-specific early immune exhaustion signals associated with severe COVID-19 outcomes.

In cancer, CRISGI uncovered stage-specific CT interactions linked to survival across five TCGA bulk RNA-seq data, highlighted FoxO/p53 signaling in scRNA-seq colorectal cancer progression, and identified LUM-driven invasive transitions in ST breast tumors.

Together, CRISGI bridges mechanistic insight and translational potential, enabling early and context-specific detection of disease transitions.

## 🚀 Getting Started

Want to start using it immediately? Check out the following sections:

### 📥 Installation

CRISGI is a Python package that can be installed via pip. You can install it from PyPI or from the source.

Details on how to install CRISGI can be found in the [Installation](installation.md) section.

### 🔧 Usage

CRISGI is designed to be user-friendly and easy to use. The package provides a set of functions that can be used to perform various tasks related to critical transitions in gene interactions.

### 📖 Tutorials

Moreover, you can find some tutorials in the [Tutorial](tutorial.md) section. These tutorials will guide you through the process of using CRISGI for your own data analysis.

## 📚 API Reference

For more detailed information, please refer to the [API Reference](api_reference.md).

## 📑 Citation

If you use CRISGI in your research, please cite the following paper:

APA format:

```
Lyu, C., Jiang, A., Ng, K. H., Liu, X., & Chen, L. (2025). Predicting Early Transitions in Respiratory Virus Infections via Critical Transient Gene Interactions. bioRxiv. https://doi.org/10.1101/2025.04.18.649619
```


BibTeX format:

```bibtex
@article{crisgi,
  title={Predicting Early Transitions in Respiratory Virus Infections via Critical Transient Gene Interactions},
  author={Lyu, Chengshang and Jiang, Anna and Ng, Ka Ho and Liu, Xiaoyu and Chen, Lingxi},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.04.18.649619},
  publisher={Cold Spring Harbor Laboratory}
}
```

<script type='text/javascript' id='clustrmaps' src='//cdn.clustrmaps.com/map_v2.js?cl=ffffff&w=a&t=n&d=aehrmM5iUrNweGVMgYWen6iLrg21PDipO7L_XllJ1lo&co=2d78ad&cmo=3acc3a&cmn=ff5353&ct=ffffff'></script>

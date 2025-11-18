# RareFold
Structure prediction and design of proteins with noncanonical amino acids.

**RareFold** predicts single-chain protein structures containing **rare** noncanonical amino acids and enables the design of novel peptide binders through the EvoBindRare framework.

RareFold supports 49 different amino acid types.\
The 20 regular ones, and 29 **rare** ones:
MSE, TPO, MLY, CME, PTR, SEP,SAH, CSO, PCA, KCX, CAS, CSD, MLZ, OCS, ALY, CSS, CSX, HIC, HYP, YCM, YOF, M3L, PFF, CGU,FTR, LLP, CAF, CMH, MHO

**EvoBindRare** designs both linear and cyclic binders directly from a protein target sequence, no prior knowledge of binding sites is required. The framework enables rapid and flexible design with expanded chemical diversity through the incorporation of noncanonical amino acids. EvoBindRare has been experimentally validated for both linear and cyclic designs, achieving high-affinity binding in each case.

[Read more here](https://www.biorxiv.org/content/10.1101/2025.05.19.654846v1)

<img src="./data/RareFold.svg"/>


# Table of Contents
1.  [RareFold](#rarefold)
2.  [LICENSE](#license)
3.  [Colab](#colab)
4.  [Installation](#installation)
5.  [Predict using RareFold](#predict-using-rarefold)
6.  [Design using EvoBindRare](#design-using-evobindrare)
    * [Design linear or cyclic peptide binders incorporating noncanonical amino acids](#design-linear-or-cyclic-peptide-binders-incorporating-noncanonical-amino-acids)
    * [Run the test case](#run-the-test-case)
        * [Step-by-Step Time Savings](#step-by-step-time-savings)
7.  [Citation](#citation)
8.  [Data](#data)
9. [The EvoBind ecosystem](#the-evobind-ecosystem)

---

# LICENSE
RareFold is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  \
The RareFold parameters for prediction are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode). \
The design protocol EvoBindRare and the parameters for design are made available under the terms of the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).

**You may not use these files except in compliance with the licenses.**

# Colab
It is possible to run EvoBindRare online in the [Google colab here](https://colab.research.google.com/github/patrickbryant1/RareFold/blob/main/rarefold.ipynb)


# Installation
The entire installation takes <1 hour on a standard computer. \
We assume you have CUDA12. For CUDA11, you will have to change the installation of some packages. \
The runtime will depend on the GPU you have available and the size of the protein you are predicting. \
On an NVIDIA A100 GPU, the prediction time is a few minutes on average.

First install miniconda, see: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html or https://docs.conda.io/projects/miniconda/en/latest/miniconda-other-installer-links.html


```
bash install_dependencies.sh
```

1. Install the RareFold environment
2. Get the RareFold parameters for single-chain structure prediction
3. Get the EvoBindRare parameters for binder design
4. Get Uniclust for MSA search
5. Install HHblits for MSA search

# Predict using RareFold
Run the test case (a few minutes)
```
conda activate rarefold
bash predict.sh
```


# Design using EvoBindRare
## Design linear or cyclic peptide binders incorporating noncanonical amino acids
EvoBindRare designs novel peptide binders based only on a protein target sequence.
It is not necessary to specify any target residues within the protein sequence - let RareFold choose!
EvoBindRare is the only protocol with experimentally verified cyclic-noncanonical amino acid design capacity.

<p align="center">
  <img alt="Linear" src="./data/linear.gif" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Cyclic" src="./data/cyclic.gif" width="45%">
</p>

The target structure is shown in green, with canonical peptide residues in blue and noncanonical residues in magenta.

For the original version of EvoBind with regular amino acids, see: https://github.com/patrickbryant1/EvoBind

### Run the test case

```
conda activate rarefold
bash design_eff.sh
```
- This runs lengths 10-15 using 5 initialisations per length (on a single GPU!).
Compare this with using 6 lengths x 5 = 30 GPUs.
(with 40 GB RAM this fits, if you have less, reduce the number of lengths/initialisations)

The design_eff.sh script also calls many CPU instances to handle data processing, mutation and feature updates efficiently and simultaneously. This reduces GPU off-time from minutes to seconds for each iteration, resulting in many hours saved.
- For your own design, take your time to optimise the GPU utilisation before starting the design. Try to set different number of lengths and initalisations and check the GPU utilisation (until you run out of RAM).

By consolidating all 30 (6 lengths x 5 initializations) design threads onto a single device, we not only slash the GPU requirement but also eliminate a substantial amount of redundant CPU overhead **per** design step.

#### Step-by-Step Time Savings
| Design Step Component | Old Time (30x Multiplier) | New Time (Consolidated) | Time Saved (per step) |
| :--- | :--- | :--- | :--- |
| **Prediction (GPU)** | 91.29 s (per process, on 30 separate GPUs) | 91.29 s (on 1 consolidated GPU) | **30&times; Hardware Reduction** |
| **Non-Prediction CPU Overhead** | **600.60** s | **20.02** s | **580.58** s |
| **Total Step Time** | $\approx$ 600.60 s (CPU time) + 91.29 s (GPU time) | **111.31** s | (Faster wall-clock time + massive resource savings) |

#### Breakdown of New CPU Overhead (20.02 s)
| Step | Time (s) |
| :--- | :--- |
| Mutating sequences | 3.16 |
| Loss calcs | 3.41 |
| Making new feats | 0.04 |
| Adding metrics | 0.01 |
| Saving structures | 13.40 |
| **Total** | **20.02** |

# Citation
If you use RareFold in your research, please cite

Li Q, Daumiller D, Zuo F, Marcotte H, Pan-Hammarstrom Q and Bryant P. RareFold: Structure prediction and design of proteins with noncanonical amino acids. bioRxiv. 2025. p. 2025.05.19.654846. doi:10.1101/2025.05.19.654846
[link to paper](https://www.biorxiv.org/content/10.1101/2025.05.19.654846v2.full)

# Data
https://zenodo.org/records/17071355

Unfortunately the entire train data with MSAs is too large to share through zenodo. This repo mainly contains predictions and metrics.

# The EvoBind ecosystem
[EvoBind](https://github.com/patrickbryant1/EvoBind) - designs novel [cyclic] peptide binders based **only on a protein target sequence**.
[RareFold](https://github.com/patrickbryant1/RareFold) - prediction & design with noncanonical amino acids \
[RareFoldGPCR](https://github.com/patrickbryant1/RareFoldGPCR) - GPCR agonist design with noncanonical amino acids

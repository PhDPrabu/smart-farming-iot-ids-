# Smart Farming IoT Intrusion Detection using TabTransformer–LightGBM with SHAP Explainability

## Overview

This repository contains the implementation of the hybrid TabTransformer–LightGBM framework proposed for explainable intrusion detection in smart farming IoT environments.

The framework integrates:

- **TabTransformer** for contextual representation learning from heterogeneous IoT traffic features
- **LightGBM** for scalable and efficient binary intrusion classification
- **SHAP (SHapley Additive exPlanations)** for interpretable intrusion analysis and behavioral explanation

The proposed system is designed for cloud-assisted smart farming environments characterized by:
- heterogeneous IoT devices,
- dynamic traffic behavior,
- severe class imbalance,
- and real-time intrusion detection requirements.

The framework is evaluated using:
- CIC-IoT 2023
- CIC-IoT-DIAD 2024
- UNSW-NB15

datasets under a binary classification setting (Benign vs Attack).

---

# Repository Structure


smart-farming-iot-ids/
│
├── README.md
├── requirements.txt
├── sample_config.yaml
│
├── preprocessing.py
├── train_tabtransformer.py
├── train_lightgbm.py
├── shap_analysis.py
├── inference.py
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── cross_dataset_evaluation.ipynb
│   └── shap_visualization.ipynb
│
├── models/
│   └── README.md
│
├── outputs/
│   └── README.md
│
└── data/
    └── README_DATA.md


----------------------------
Framework Workflow
----------------------------
The proposed intrusion detection pipeline consists of four sequential stages:
Data Preprocessing
Removal of inconsistent records
Categorical feature encoding
Numerical feature normalization
Stratified train/validation/test partitioning
Contextual Representation Learning
TabTransformer-based self-attention learning
Cross-feature interaction modeling
Contextual embedding generation
Binary Intrusion Classification
LightGBM-based attack detection
Class-weighted learning for imbalance handling
Scalable gradient boosting classification
Explainability and Behavioral Interpretation
Global SHAP feature importance
Local intrusion explanation
Cross-feature interaction analysis
False positive behavioral analysis
------------------------------------
Experimental Configuration
------------------------------------
Computational Environment
Parameter	Configuration
Platform	Google Colab Pro
Operating System	Ubuntu Linux
GPU	NVIDIA Tesla T4
CPU	2 vCPUs
RAM	16 GB
Python Version	Python 3.1
---------------------------------------------------
Datasets
---------------------------------------------------
Dataset	Description
CIC-IoT 2023	Large-scale IoT intrusion dataset
CIC-IoT-DIAD 2024	IoT distributed attack dataset
UNSW-NB15	External network intrusion benchmark dataset
-----------------------------------------------------------------------
Installation
-----------------------------------------------------------------------
Clone Repository

git clone https://github.com/PhdPrabu/smart-farming-iot-ids.git

cd smart-farming-iot-ids

Install Dependencies

pip install -r requirements.txt

-----------------------------------------------
Execution Pipeline
----------------------------------------------
Step 1 — Data Preprocessing
python preprocessing.py
Step 2 — Train TabTransformer
python train_tabtransformer.py
Step 3 — Train LightGBM Classifier
python train_lightgbm.py
Step 4 — SHAP Explainability Analysis
python shap_analysis.py
Step 5 — End-to-End Inference
python inference.py
---------------------------------------------------
Key Features
---------------------------------------------------
Context-aware intrusion representation learning
Robust binary intrusion detection
SHAP-based explainability
Cross-feature behavioral interpretation
False positive analysis support
Cross-dataset generalization evaluation
Modular cloud-oriented architecture
------------------------------------------
Evaluation Metrics
------------------------------------------
The framework is evaluated using:
Accuracy
Precision
Recall
F1-Score
AUC-ROC
Reproducibility

The repository includes:
preprocessing scripts,
hyperparameter configurations,
model training modules,
evaluation pipeline,
SHAP explainability analysis,
and deployment-oriented inference workflow

to support reproducibility and future comparative research.

-------------------------------------------------------------------------------------------------------------------------------------------
Citation
-------------------------------------------------------------------------------------------------------------------------------------------
If you use this repository in your research, please cite:

@article{smartfarming_ids_2025,
  title={Hybrid TabTransformer--LightGBM Framework with SHAP Explainability for Intrusion Detection in Smart Farming IoT Networks},
  author={Alphin Ezhil Manuel M L, S.Priya, Manikandan Thirumalaisamy, Prabu Kaliyaperumal, Balamurugan Balusamy, Francesco Benedetto},
  journal={Under Review},
  year={2025}
}
Contact

For research-related queries, please contact the corresponding author.
Dr. Francesco Benedetto


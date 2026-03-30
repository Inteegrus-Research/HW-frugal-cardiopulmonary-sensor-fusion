# Frugal Cardiopulmonary Sensor Fusion: Algorithmic Evaluation of SpO2 and ECG for
Edge-Based Sleep Apnea Triage

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: MESA](https://img.shields.io/badge/Dataset-NSRR_MESA-green)](https://sleepdata.org/datasets/mesa)

This repository contains the official feature extraction and machine learning pipeline for the paper: **"Frugal Cardiopulmonary Sensor Fusion:
Algorithmic Evaluation of SpO2 and ECG for
Edge-Based Sleep Apnea Triage"**

## Overview
Continuous processing of high-frequency physiological signals (like ECG) drains the battery of wearable Edge IoT devices. This project validates a "hierarchical wake-up" architecture for Obstructive Sleep Apnea (OSA) screening. By utilizing highly compressed features from an ultra-low-power SpO2 sensor (ODI, CT90) and selectively duty-cycling the ECG sensor (HRV, EDR), we demonstrate an 80% reduction in theoretical analog front-end power consumption with negligible loss in diagnostic accuracy (AUC ~0.869). 

The pipeline is validated on a statistically powered, unstratified epidemiological cohort of $N=800$ patients from the Multi-Ethnic Study of Atherosclerosis (MESA) dataset.

---

## Repository Structure

```text
Frugal-Edge-Apnea/
│
├── data/
│   ├── raw_edfs/                     # Directory for raw .edf files (User populated)
│   ├── cohort_list.csv               # Unstratified cohort demographic metadata
│   ├── mesa-sleep-dataset-0.8.0.csv  # To generate the dedicated cohort, downlaod it from mesa
│   └── extracted_features.csv        # Extracted AASM-compliant features (50KB)
│
├── figures/                          # Generated plots and architectural diagrams
│   ├── block_diagram.png
│   ├── ieee_calibration_plot.png
│   ├── ieee_confusion_matrix.png
│   ├── ieee_energy_tradeoff.png
│   └── ieee_shap_summary.png
│
├── src/                             # Core execution pipeline
│   ├── 01_cohort_selection.py       # Generates the download script for the cohort
│   ├── 02_feature_extraction.py     # Extracts SpO2/ECG features & applies SQI
│   └── 03_model_benchmark.py        # ML evaluation, nested CV, and duty-cycle ablation
│
├── requirements.txt
└── README.md
```

---

## Prerequisites and Installation

**1. Clone the repository:**
```bash
git clone https://github.com/Inteegrus-Research/HW-frugal-cardiopulmonary-sensor-fusion.git
cd HW-frugal-cardiopulmonary-sensor-fusion
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Install the NSRR Ruby Client:**
The raw medical data is hosted by the National Sleep Research Resource (NSRR). You must install their API tool to download the EDF files.
```bash
gem install nsrr
```

---

## Data Acquisition (MESA Dataset)

Due to file size constraints (~120GB), the raw `.edf` polysomnography files are not hosted in this repository. You must download them securely via the NSRR API.

1. Request data access for the MESA dataset at [sleepdata.org](https://sleepdata.org/datasets/mesa).
2. Once approved, retrieve your API token from your NSRR profile.
3. Authenticate your local machine:
   ```bash
   nsrr config --token YOUR_API_TOKEN_HERE
   ```
4. Ensure the `data/raw_edfs/` directory exists in your local repository. The pipeline will download the specific 800 patient records into this folder.

---

## Execution Pipeline

The methodology is split into three sequential scripts to ensure modularity and reproducibility.

### Step 1: Cohort Selection and Download
This script reads the master MESA demographic file, isolates the unstratified N=800 cohort, and generates a targeted bash script for downloading the raw waveforms.
```bash
cd src
python 01_cohort_selection.py
```
*Note: Execute the resulting `download_edfs.sh` script to pull the 120GB of data into `data/raw_edfs/`. This process is network-dependent and may take several hours.*

### Step 2: Frugal Feature Extraction
This script processes the raw `.edf` files, applying a strict Signal Quality Index (SQI) and extracting AASM-compliant features (ODI, CT90, RMSSD, LF/HF, and EDR). Corrupted records are automatically dropped.
```bash
python 02_feature_extraction.py
```

### Step 3: Machine Learning & Power Budget Simulation
Executes the Subject-Wise 5-Fold Cross-Validation, handles `NaN` values via sparsity-aware splitting in XGBoost, and generates all performance metrics, SHAP explainability plots, and the simulated hardware duty-cycle ablation.
```bash
python 03_model_benchmark.py
```
*Generated plots will be saved to the `figures/` directory.*

---

## Key Findings
* **Optimal Linear Performance:** A frugal Logistic Regression classifier achieved an AUC of 0.869, proving complex non-linear models (e.g., deep CNNs) are mathematically redundant for SpO2-dominant risk triage.
* **Explainable AI:** SHAP analysis confirmed that the ultra-low-power SpO2 features (ODI, CT90) account for the overwhelming majority of the predictive decision boundary.
* **Energy Trade-Off:** Reducing the ECG hardware duty-cycle from 100% to 0% resulted in a functionally flat AUC variation (0.850 to 0.851), theoretically reducing analog front-end power draw by 80%.

---

## License
This project is licensed under the MIT License. MESA dataset usage is strictly governed by the National Sleep Research Resource (NSRR) Data Use Agreements.
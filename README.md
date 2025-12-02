# Robust BAN Authentication Dataset Curation and Analyses Framework

A comprehensive Python framework for analyzing Bluetooth Low Energy (BLE) based Body Area Network (BAN) authentication using IQ (In-phase and Quadrature) signal data. This framework provides end-to-end capabilities for dataset curation, feature engineering, machine learning/deep learning model training, and robust security analysis.

## Overview

This project implements a complete analysis pipeline for BLE-based physical layer authentication in Body Area Networks. It supports:

- **Multi-scenario Analysis**: On-body and off-body authentication scenarios
- **Advanced Signal Processing**: IQ signal processing, feature extraction, and device fingerprint mitigation
- **Machine Learning & Deep Learning**: Multiple ML models (Random Forest, SVM, Gradient Boosting, etc.) and DL architectures (MLP, CNN1D, CNN-RNN)
- **Security Analysis**: Adversarial attack detection, robustness evaluation, and verification metrics
- **Comprehensive Reporting**: Automated result aggregation with detailed visualizations and metrics

## Key Features

### 1. Data Management (`data_manager.py`)
- Unified data loading from multiple dataset formats (Parquet, CSV, Pickle)
- Support for on-body and off-body scenarios
- Automated data preprocessing and normalization
- Dataset card generation for reproducibility

### 2. Signal Processing & Feature Engineering (`feature_engineering.py`, `IQ.py`)
- IQ signal processing with device fingerprint mitigation
- Advanced feature extraction:
  - Statistical features (mean, std, percentiles, etc.)
  - Spectral features (band energy, centroid, bandwidth, rolloff)
  - Autocorrelation features
  - Phase and magnitude statistics
- Pre-feature normalization to prevent device fingerprint leakage

### 3. Machine Learning Framework (`ml_dl_framework.py`)
- **ML Models**: Random Forest, Extra Trees, Gradient Boosting, SVM, Logistic Regression, KNN, Nearest Centroid
- **DL Architectures**: MLP, 1D CNN, CNN-RNN hybrid models
- **Evaluation Metrics**: 
  - Standard metrics (accuracy, F1, precision, recall, AUC)
  - Advanced diagnostics (ECE - Expected Calibration Error, class divergence, separability)
  - Train/test split diagnostics for data leakage detection
- **Cross-validation**: Group-based and stratified splitting strategies

### 4. Analysis Framework (`analysis_framework.py`)
Comprehensive analysis pipeline supporting:
- **Device Identification**: On-body and off-body device classification
- **Position Analysis**: Body position detection and classification
- **Movement Detection**: On-body movement pattern analysis
- **Verification**: Per-device ROC/AUC/EER analysis for authentication
- **Hierarchical Classification**: Multi-stage classification (scenario → position → device)
- **Mixed Scenario Analysis**: Baseline vs robust model comparison with fingerprint mitigation
- **Adversarial Analysis**: Robustness evaluation against attacks

### 5. Orchestration (`orchestrator.py`)
- Automated execution of all analyses
- Configurable defaults for balanced sampling
- Result aggregation and manuscript-ready outputs

### 6. Training API (`training_api.py`)
- Unified interface for ML/DL model training
- Automatic model selection based on performance
- Result persistence with metadata tracking
- Support for grouped cross-validation

### 7. Results Reporting (`results_report.py`)
- Automated aggregation of analysis results
- CSV and JSON export for manuscript tables
- Combined summary generation across all analyses

## Installation

### Requirements
- Python 3.8+
- NumPy, Pandas, SciPy
- scikit-learn
- PyTorch (for deep learning models)
- Matplotlib (for visualizations)
- Optional: tqdm (for progress bars), joblib (for model serialization)

### Setup
```bash
# Clone the repository
git clone https://github.com/mavrik3/project.git
cd project

# Install dependencies
pip install numpy pandas scipy scikit-learn torch matplotlib tqdm joblib
```

## Usage

### Running Complete Analysis Pipeline

```python
from orchestrator import run_all_analyses

# Run all analyses with default configuration
run_all_analyses()
```

### Running Specific Analyses

```python
from analysis_framework import AnalysisFramework

# Initialize framework
framework = AnalysisFramework(
    data_root="path/to/data",
    output_root="path/to/results",
    max_samples_per_class=10000,
    use_advanced_features=True,
    use_deep_learning=True,
    default_balance_by=["dvc", "pos_label", "session"]
)

# Run specific analysis
results = framework.analyze_offbody_position(source_path="path/to/dataset")

# Run verification analysis
ver_results = framework.analyze_offbody_verification(source_path="path/to/dataset")

# Run device identification
dev_results = framework.analyze_device_identification(scenario="onBody")
```

### Feature Engineering

```python
from feature_engineering import FeatureEngineering
import numpy as np

# Initialize feature engine
fe = FeatureEngineering()

# Extract features from IQ data
iq_data = np.array([...])  # Your IQ signal data
features = fe.extract_features(iq_data)
```

### Training Custom Models

```python
from training_api import train_classifier
import numpy as np

# Prepare data
X = np.array([...])  # Feature matrix
y = np.array([...])  # Labels
groups = np.array([...])  # Group labels for cross-validation

# Train models
results = train_classifier(
    X, y, groups,
    task_name="my_classification",
    output_dir="./results",
    use_grouped_split=True
)
```

## Project Structure

```
.
├── README.md                           # This file
├── ABOUT.md                           # Detailed project description
├── IQ.py                              # IQ signal processing utilities
├── data_manager.py                    # Data loading and preprocessing
├── feature_engineering.py             # Feature extraction pipeline
├── ml_dl_framework.py                 # ML/DL models and training
├── training_api.py                    # Unified training interface
├── analysis_framework.py              # Main analysis orchestration
├── orchestrator.py                    # Automated analysis runner
├── results_report.py                  # Results aggregation
├── adversarial_attacks.ipynb          # Adversarial attack analysis
├── Robust BAN Authentication Dataset Curation and Analyses Framework/
│   ├── Code/                          # Alternative code location
│   └── Results/                       # Sample results and outputs
└── DEV_SESSION_SUMMARY.md            # Development notes
```

## Analysis Outputs

The framework generates comprehensive outputs including:
- **JSON Results**: Detailed metrics, model parameters, and metadata
- **CSV Summaries**: Tabular results for manuscript tables
- **Visualizations**: ROC curves, confusion matrices, feature importance plots
- **Model Files**: Serialized trained models (via joblib)
- **Dataset Cards**: Reproducibility metadata with train/test splits

## Key Analyses

1. **On-Body Device Identification**: Classify which device generated the signal
2. **Off-Body Position Detection**: Determine body position from off-body measurements
3. **Verification Analysis**: Per-device authentication with ROC/AUC/EER metrics
4. **Hierarchical Classification**: Multi-level classification (scenario → position → device)
5. **Mixed Scenario Analysis**: Evaluate baseline vs robust models with fingerprint mitigation
6. **Movement Detection**: Classify movement patterns in on-body scenarios
7. **Adversarial Robustness**: Evaluate model resilience to attacks

## Performance Metrics

The framework tracks comprehensive metrics:
- **Classification**: Accuracy, F1-score (macro/weighted), Precision, Recall
- **Verification**: AUC, EER (Equal Error Rate), ROC curves
- **Calibration**: Expected Calibration Error (ECE)
- **Diagnostics**: Class divergence, separability (d', Bhattacharyya distance)
- **Cross-validation**: Per-fold metrics with train/test indices

## Security Features

- **Device Fingerprint Mitigation**: Pre-feature normalization to prevent hardware ID leakage
- **Adversarial Analysis**: Attack detection and robustness evaluation
- **Data Leakage Prevention**: Group-based splitting, train/test divergence monitoring
- **Reproducibility**: Seed management, split tracking, manifest generation

## Citation

If you use this framework in your research, please cite:
```
[Your publication details here]
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## Acknowledgments

This framework was developed as part of research on Body Area Network authentication using physical layer characteristics of BLE signals.

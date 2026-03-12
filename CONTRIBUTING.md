# Contributing to Robust BAN Authentication Framework

Thank you for your interest in contributing! This document outlines guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/mavrik3/project/issues) to report bugs or request features.
- Search existing issues before opening a new one.
- Include a clear title, description, and steps to reproduce for bug reports.

### Submitting Pull Requests

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure your code follows the style conventions described below.
4. Add or update docstrings for any new or modified functions and classes.
5. Open a pull request with a descriptive title and summary of your changes.

## Code Style

### Python Style

- Follow [PEP 8](https://pep8.org/) conventions.
- Use **Google-style docstrings** for all public classes, methods, and functions.

**Example:**
```python
def train_classifier(X, y, groups, task_name, output_dir=None):
    """Train and evaluate classifiers for a given task.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
        groups: Group labels for cross-validation splits.
        task_name: Descriptive name for the classification task.
        output_dir: Directory to save results. If None, results are not persisted.

    Returns:
        dict: Evaluation results including metrics, model paths, and visualizations.
    """
```

### Module Conventions

| Module | Responsibility |
|--------|---------------|
| `data_manager.py` | Data loading and preprocessing |
| `feature_engineering.py` | Feature extraction from IQ signals |
| `ml_dl_framework.py` | ML/DL model training and evaluation |
| `training_api.py` | Unified training interface |
| `analysis_framework.py` | High-level analysis orchestration |
| `orchestrator.py` | Entry points for automated analysis runs |
| `results_report.py` | Result aggregation and export |
| `IQ.py` | Low-level BLE IQ signal utilities |

## Areas for Contribution

We welcome contributions in the following areas:

- **New feature extraction methods** in `feature_engineering.py`
- **Additional ML/DL models** in `ml_dl_framework.py`
- **Novel analysis types** in `analysis_framework.py`
- **Performance optimizations** across all modules
- **Documentation improvements** (docstrings, README, guides)
- **Bug fixes** with accompanying regression tests

Planned enhancements from [ABOUT.md](ABOUT.md):
- Transformer and attention-based DL architectures
- Real-time inference optimization
- Expanded adversarial attack library
- Transfer learning capabilities
- Federated learning support

## Dependencies

Install all required dependencies before developing:

```bash
pip install -r requirements.txt
```

See [requirements.txt](requirements.txt) for the full dependency list.

## Questions

For questions, open a [GitHub Issue](https://github.com/mavrik3/project/issues) or review the existing documentation:

- [README.md](README.md) — Project overview and quick-start guide
- [ABOUT.md](ABOUT.md) — Architecture and technical details
- [DEV_SESSION_SUMMARY.md](DEV_SESSION_SUMMARY.md) — Recent implementation notes

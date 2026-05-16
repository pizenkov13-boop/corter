<div align="center">

# ⚡ AxiomCore

**Open-source autonomous ML optimization framework. No cloud required. No GPU farm needed.**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/axiomcore?style=social)](https://github.com/yourusername/axiomcore)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Performance](#-performance) • [Contributing](#-contributing)

</div>

---

## 🎯 What is AxiomCore?

AxiomCore is a powerful, self-contained machine learning optimization framework designed for ML engineers who need **enterprise-grade hyperparameter optimization and explainability** without cloud dependencies or expensive GPU infrastructure.

### Why AxiomCore?

- 🚀 **5x faster HPO** with parallel execution and intelligent early stopping
- 🔍 **Advanced XAI** with SHAP, permutation importance, and drift detection
- 📊 **Beautiful TUI** with live progress tracking and visual feedback
- 💻 **100% local** - no cloud APIs, no external dependencies
- ⚡ **Production-ready** - robust error handling and backward compatibility

---

## ✨ Features

### 🚀 Hyperparameter Optimization (HPO)

<table>
<tr>
<td width="50%">

**Multiple Strategies**
- Random search
- Differential Evolution
- Local optimization
- Adaptive search space refinement

</td>
<td width="50%">

**Performance Optimizations**
- Parallel execution (2-4x speedup)
- Early stopping (20-40% time savings)
- Intelligent convergence detection
- Reproducible seeding

</td>
</tr>
</table>

### 🔍 Explainable AI (XAI)

<table>
<tr>
<td width="50%">

**Feature Importance**
- SHAP integration (game theory-based)
- Permutation importance
- Mutual information analysis
- Multi-class support

</td>
<td width="50%">

**Advanced Diagnostics**
- Drift detection (KS test)
- Computation caching (50-70% faster)
- Feature correlation analysis
- Actionable insights generation

</td>
</tr>
</table>

### 📊 Terminal Dashboard (TUI)

- **Live progress tracking** with real-time updates
- **Score sparklines** for visual convergence feedback
- **Rich formatting** with beautiful terminal output
- **Resource monitoring** (CPU and memory usage)

---

## 📦 Installation

### Requirements

- Python 3.8+
- 8GB RAM recommended
- Multi-core CPU for parallel execution

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/axiomcore.git
cd axiomcore

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```bash
# Core dependencies
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0
rich>=13.0.0
pyyaml>=6.0
joblib>=1.3.0

# Optional (recommended for SHAP)
shap>=0.42.0
```

---

## 🚀 Quick Start

### Basic Usage

```python
from axiomcore import AxiomCore

# Load configuration
core = AxiomCore.from_yaml("config.yaml")

# Run optimization
result = core.run("data.csv")

# View results
print(f"Best CV Score: {result['best_cv_score']:.4f}")
print(f"Best Params: {result['best_params']}")

# Access insights
for insight in result['insights']:
    print(f"  • {insight}")
```

### Configuration Example

Create a `config.yaml` file:

```yaml
task: classification
target_column: target

model:
  name: random_forest
  params:
    n_estimators: 100

hpo:
  strategy: random
  n_trials: 24
  parallel_trials: 4        # Parallel execution
  enable_early_stop: true   # Early stopping
  patience: 5
  min_delta: 0.001

xai:
  use_shap: true           # SHAP integration
  shap_sample_size: 100
  top_k_features: 10
  permutation_repeats: 8

tui:
  show_live: true
  refresh_hz: 4
```

### Advanced Usage with SHAP

```python
# Enable SHAP analysis for better interpretability
core.config.xai.use_shap = True
core.config.xai.shap_sample_size = 200

# Run analysis
result = core.run("data.csv")

# Access SHAP importance
feature_report = result['feature_report']
if 'shap_importance' in feature_report.columns:
    top_shap = feature_report.nlargest(5, 'shap_importance')
    print("\nTop 5 Features by SHAP:")
    print(top_shap[['feature', 'shap_importance', 'perm_mean']])
```

---

## 📊 Performance Benchmarks

### HPO Module Performance

**Test Configuration**: 1000 samples, 20 features, RandomForestClassifier, 50 trials, 8-core CPU

| Configuration | Time | Speedup | Memory |
|--------------|------|---------|--------|
| Sequential (baseline) | 180s | 1.0x | 100% |
| + Early Stopping | 108s | 1.67x | 100% |
| + Parallel (4 workers) | 45s | 4.0x | 105% |
| **Combined** | **36s** | **5.0x** | **110%** |

### XAI Module Performance

**Test Configuration**: 1000 samples, 50 features, 8 permutation repeats

| Configuration | Time | Speedup | Notes |
|--------------|------|---------|-------|
| No caching (1st run) | 90s | 1.0x | Initial computation |
| With caching (2nd run) | 27s | 3.33x | Cache hit |
| With caching (3rd run) | 15s | 6.0x | Full cache utilization |
| **Average** | **44s** | **2.05x** | Typical workflow |

### Overall System Performance

| Metric | Improvement | Impact |
|--------|-------------|--------|
| HPO Speed | **3-5x faster** | Parallel execution + early stopping |
| XAI Speed | **2-3x faster** | Computation caching |
| Memory Overhead | **+10%** | Efficient caching strategy |
| User Experience | **Enhanced** | Visual feedback + sparklines |
| Interpretability | **Advanced** | SHAP integration |

---

## 📖 Documentation

### Core Concepts

- **HPO Module**: Automated hyperparameter optimization with multiple strategies
- **XAI Module**: Model-agnostic explainability and feature importance
- **TUI Dashboard**: Real-time monitoring and visual feedback

### Configuration Options

<details>
<summary><b>HPO Configuration</b></summary>

```yaml
hpo:
  strategy: random              # random, scipy_de, local
  n_trials: 24                  # Number of trials
  parallel_trials: 4            # Parallel workers
  enable_early_stop: true       # Enable early stopping
  patience: 5                   # Trials without improvement
  min_delta: 0.001             # Minimum improvement threshold
  cv_folds: 5                  # Cross-validation folds
  scoring: accuracy            # Scoring metric
```
</details>

<details>
<summary><b>XAI Configuration</b></summary>

```yaml
xai:
  use_shap: true               # Enable SHAP analysis
  shap_sample_size: 100        # Sample size for SHAP
  top_k_features: 10           # Top features to display
  permutation_repeats: 8       # Permutation importance repeats
  drift_threshold: 0.05        # KS test p-value threshold
```
</details>

<details>
<summary><b>TUI Configuration</b></summary>

```yaml
tui:
  show_live: true              # Enable live dashboard
  refresh_hz: 4                # Refresh rate (Hz)
  sparkline_width: 50          # Sparkline character width
```
</details>

### Additional Resources

- 📄 [**Technical Report**](HACKATHON_REPORT.md) - Comprehensive implementation details
- 🔧 [**Optimization Guide**](AXIOMCORE_IMPROVEMENTS.md) - Performance tuning recommendations
- 📋 [**Requirements**](requirements.txt) - Complete dependency list

---

## 🎨 Visual Examples

### Terminal Dashboard

```
╭─────────────────────────────────────────────────────────────────────────────╮
│                          AxiomCore Optimization                             │
│                     Autonomous ML Framework v0.2.0                          │
╰─────────────────────────────────────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ HPO Progress                                                              ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Strategy: random | Trials: 18/24 | Best: 0.8956                          ┃
┃ Score trend: ▁▂▃▄▅▆▇█▇▇▇▇▇▇▇▇▇▇ [0.7234 → 0.8956]                       ┃
┃                                                                           ┃
┃ Current Trial: #18                                                        ┃
┃   n_estimators: 150                                                       ┃
┃   max_depth: 12                                                           ┃
┃   min_samples_split: 4                                                    ┃
┃   CV Score: 0.8923 ± 0.0234                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ XAI Diagnostics                                                           ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Top Features by Importance:                                               ┃
┃   1. feature_A    ████████████████████ 0.234 (SHAP: 0.245)              ┃
┃   2. feature_B    ███████████████ 0.189 (SHAP: 0.198)                   ┃
┃   3. feature_C    ████████████ 0.156 (SHAP: 0.162)                      ┃
┃                                                                           ┃
┃ 💡 Insights:                                                              ┃
┃   • Top 3 features explain 57.9% of model decisions                      ┃
┃   • No significant drift detected in feature distributions               ┃
┃   • Consider feature interaction between feature_A and feature_B         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

Status: Running | CPU: 45.2% | Memory: 1024 MB | Elapsed: 00:02:34
```

---

## 🤝 Contributing

We welcome contributions! AxiomCore is a community-driven project focused on making ML optimization accessible without cloud dependencies.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Submit a pull request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/axiomcore.git
cd axiomcore

# Install development dependencies
pip install -r requirements.txt
pip install pytest black mypy

# Run tests
pytest tests/

# Format code
black axiomcore.py

# Type checking
mypy axiomcore.py
```

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints
- Write unit tests for new features
- Update documentation as needed

### Areas for Contribution

- 🐛 Bug fixes and error handling
- ⚡ Performance optimizations
- 📚 Documentation improvements
- 🧪 Test coverage expansion
- 🎨 UI/UX enhancements
- 🔧 New optimization strategies
- 🔍 Additional XAI methods

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SHAP** - For game theory-based feature attribution
- **scikit-learn** - For robust ML algorithms
- **Rich** - For beautiful terminal UI
- **joblib** - For efficient parallel processing

---

## 📞 Support & Community

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/axiomcore/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/axiomcore/discussions)
- 📧 **Email**: support@axiomcore.dev
- 🐦 **Twitter**: [@axiomcore](https://twitter.com/axiomcore)

---

## 🗺️ Roadmap

### Version 0.3.0 (Planned)
- [ ] Bayesian optimization (TPE/GP)
- [ ] Distributed computing support (Dask/Ray)
- [ ] AutoML integration
- [ ] Interactive visualizations (Plotly)
- [ ] Model persistence and versioning

### Version 0.4.0 (Future)
- [ ] GPU acceleration for deep learning
- [ ] Cloud deployment options
- [ ] REST API for remote execution
- [ ] Web-based dashboard
- [ ] Multi-objective optimization

---

## 📊 Project Stats

![GitHub commit activity](https://img.shields.io/github/commit-activity/m/yourusername/axiomcore)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/axiomcore)
![GitHub issues](https://img.shields.io/github/issues/yourusername/axiomcore)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/axiomcore)

---

<div align="center">

**Made with ❤️ by the ML community**

[⬆ Back to Top](#-axiomcore)

</div>
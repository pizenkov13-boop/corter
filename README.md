<div align="center">

# ⚡ Corter

### Autonomous ML Optimization Framework

*Local hyperparameter search and explainability — no cloud required.*

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/pizenkov13-boop/Corter?style=social)](https://github.com/pizenkov13-boop/Corter)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Contributing](#-contributing)

</div>

---

## 🎯 What is Corter?

Corter runs hyperparameter search on tabular CSV data, fits a scikit-learn model with the best settings, and reports feature importance and short text insights. Everything runs on your machine via the CLI, Python API, or optional Flask web dashboard.

### Why Corter?

- **CLI workflow** — `corter init`, `corter run`, `corter web`
- **YAML configuration** — task, model, HPO, and XAI settings in one file
- **Terminal UI** — live progress with Rich during optimization
- **Web dashboard** — monitor runs at `http://localhost:5000` when using `--web` or `corter web`
- **Explainability** — permutation importance; optional SHAP when `corter-ml[xai]` is installed

---

## ✨ Features

### Hyperparameter optimization

- Random search over a YAML-defined search space
- Bayesian optimization (`bayesian`) via Optuna TPE sampler (`pip install optuna` or `corter-ml[bayesian]`)
- SciPy global (`scipy_de`) and local (`scipy_local`) strategies
- Parallel trials via joblib
- Early stopping when scores stop improving
- Resumable runs — progress saved to `corter_checkpoint.json` after each trial (set `checkpoint_path: null` to disable)
- Per-trial explanation snapshots — feature importance and insights appended to `explanation_snapshots.json` after each trial

### Models

Supported `model.name` values:

| Name | Aliases |
|------|---------|
| `random_forest` | `rf` |
| `gradient_boosting` | `gbm` |
| `xgboost` | `xgb` |
| `lightgbm` | `lgbm`, `lgb` |
| `catboost` | `cb` |
| `logistic_regression` | `logreg`, `logistic` |

XGBoost, LightGBM, and CatBoost require the boosting extra: `pip install corter-ml[boosting]`
| `ridge` | |
| `svc` | `svm` |

Classification vs regression is chosen from `task` in config (or inferred when `task: auto`).

Numeric feature columns are used automatically. If `target_column` is omitted, Corter uses a column named `target`, `label`, or `y` (case-insensitive), otherwise the **last column** in the CSV.

### Explainability

- Permutation importance (always)
- SHAP values when `shap` is installed (`pip install corter-ml[xai]`)
- Drift checks and generated insight strings

### Interfaces

- **CLI** — `corter init`, `run`, `web`, `version`
- **Python** — `Corter.from_yaml(...)` and `core.run("data.csv")`
- **Web UI** — Flask app in `web_ui.py`; `corter_web.py` pushes live updates during a run

---

## 📦 Installation

### From PyPI (when published)

```bash
pip install corter-ml
```

### From source

```bash
git clone https://github.com/pizenkov13-boop/Corter.git
cd Corter
pip install -e .
```

### Optional extras

```bash
pip install corter-ml[boosting]  # XGBoost, LightGBM, CatBoost
pip install corter-ml[xai]       # SHAP support
pip install corter-ml[bayesian]  # Optuna for strategy: bayesian
pip install corter-ml[dev]       # pytest, black, mypy
```

---

## 🚀 Quick Start

### 1. Create configuration

```bash
corter init
```

Example `config.yaml`:

```yaml
task: classification
target_column: target

model:
  name: random_forest
  params:
    n_estimators: 100

hpo:
  strategy: random          # random | bayesian | scipy_de | scipy_local
  n_trials: 24
  parallel_trials: 4
  enable_early_stop: true
  cv_folds: 5
  scoring: accuracy
  search_space:
    n_estimators:
      low: 50
      high: 200
      type: int

xai:
  use_shap: true
  top_k_features: 10

tui:
  show_live: true
  refresh_hz: 4
```

### 2. Run optimization

```bash
corter run data.csv
corter run data.csv --web          # optimization + dashboard
corter run data.csv -c other.yaml
```

Results are written to `results.json` by default. After each run, Corter also writes **`results_report.html`** (best score, parameters, holdout metrics, feature importance, insights). Optional PDF: `core.run(data, export_pdf=True)` or `pip install corter-ml[reports]`.

### 3. Python API

```python
from corter import Corter

core = Corter.from_yaml("config.yaml")
result = core.run("data.csv")

print(result["best_cv_score"])
print(result["best_params"])
print(result["insights"])
```

### 4. Web dashboard only

```bash
corter web
# open http://127.0.0.1:5000
```

During `corter run data.csv --web`, the dashboard receives live updates from the optimizer.

---

## 📖 Documentation

### Task configuration

```yaml
task: auto                  # auto | classification | regression
target_column: target       # optional: auto-detect target/label/y or last column
```

### HPO configuration

```yaml
hpo:
  strategy: random          # random | bayesian | scipy_de | scipy_local
  n_trials: 50
  parallel_trials: 4
  enable_early_stop: true
  patience: 5
  min_delta: 0.001
  cv_folds: 5
  scoring: accuracy         # or f1_weighted, neg_mean_squared_error, etc.
  search_space: { ... }
  checkpoint_path: corter_checkpoint.json   # resume if interrupted; null to disable
```

### XAI configuration

```yaml
xai:
  use_shap: true            # requires corter-ml[xai]
  shap_sample_size: 100
  top_k_features: 10
  permutation_repeats: 8
  drift_threshold: 0.15
  explanation_snapshots_path: explanation_snapshots.json   # null to disable
  snapshot_top_k: 5
  snapshot_permutation_repeats: 3
```

### CLI reference

```bash
corter init [-o config.yaml]
corter run <data.csv> [-c config.yaml] [--web] [--output results.json]
corter web [--host 127.0.0.1] [--port 5000]
corter version
```

### Direct module usage

```bash
python corter.py data.csv -c config.yaml
python corter_web.py config.yaml data.csv   # optimization with web updates
gunicorn web_ui:app                         # production-style web only (see Procfile)
```

---

## 📊 Performance notes

Internal runs on **synthetic tabular data** (~1k samples, Random Forest, 50 trials, 8-core CPU):

| HPO configuration | Time | vs sequential |
|-------------------|------|---------------|
| Sequential | 180s | 1.0× |
| + early stopping | 108s | 1.7× |
| + parallel (4 workers) | 45s | 4.0× |
| Combined | 36s | 5.0× |

XAI re-analysis on synthetic data (~1k × 50 features): caching reduced repeat runs from ~90s to ~15–27s on subsequent passes.

*Benchmarks based on internal testing on synthetic data. Results may vary.*

Full methodology: [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│                   Corter                     │
├──────────────────────────────────────────────┤
│  CLI (corter_pkg)  │  corter.py  │  web_ui  │
├────────────────────┴─────────────┴──────────┤
│  HyperparameterAutopilot  →  fit best model │
│  SemanticDiagnostics      →  insights       │
│  CorterDashboard (Rich TUI)                 │
└──────────────────────────────────────────────┘
```

---

## 🤝 Contributing

1. Fork and clone the repository
2. `pip install -e ".[dev]"`
3. Make changes and run formatters/tests as appropriate
4. Open a pull request

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) — models and metrics
- [SHAP](https://github.com/slundberg/shap) — optional explainability
- [Rich](https://github.com/Textualize/rich) — terminal UI
- [Flask](https://flask.palletsprojects.com/) — web dashboard

---

## 📞 Support

- [GitHub Issues](https://github.com/pizenkov13-boop/Corter/issues)
- [GitHub Discussions](https://github.com/pizenkov13-boop/Corter/discussions)

---

<div align="center">

**Made with ❤️ by the Corter Team**

[⬆ Back to Top](#-corter)

</div>

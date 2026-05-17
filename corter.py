"""
Corter — Autonomous ML Optimization Framework

Modules:
  - HPO: Hyperparameter Autopilot (search, score, converge)
  - XAI: Semantic Diagnostics (importance, drift, narrative insights)
  - TUI: Live Rich dashboard for runs and diagnostics
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import time
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from scipy import stats
from scipy.optimize import differential_evolution, minimize
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__version__ = "0.1.0"
__all__ = [
    "Corter",
    "HyperparameterAutopilot",
    "SemanticDiagnostics",
    "CorterDashboard",
    "load_config",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HPOConfig:
    strategy: str = "random"  # random | bayesian | scipy_de | scipy_local
    n_trials: int = 24
    cv_folds: int = 5
    scoring: Optional[str] = None  # auto from task
    random_state: int = 42
    search_space: Dict[str, Any] = field(default_factory=dict)
    # Early stopping parameters
    patience: int = 5  # Stop after N trials without improvement
    min_delta: float = 0.001  # Minimum improvement threshold
    enable_early_stop: bool = True
    # Parallel execution parameters
    parallel_trials: int = 4  # Number of trials to run in parallel (0 = sequential)
    n_jobs: int = -1  # Number of CPU cores for joblib (-1 = all cores)


@dataclass
class XAIConfig:
    top_k_features: int = 10
    permutation_repeats: int = 8
    drift_threshold: float = 0.15
    include_correlations: bool = True
    # SHAP configuration
    use_shap: bool = True  # Enable SHAP analysis (requires shap package)
    shap_sample_size: int = 100  # Subsample for SHAP computation efficiency


@dataclass
class TUIConfig:
    refresh_hz: float = 4.0
    show_live: bool = True
    width: Optional[int] = None


@dataclass
class CorterConfig:
    task: str = "auto"  # classification | regression | auto
    target_column: Optional[str] = None
    hpo: HPOConfig = field(default_factory=HPOConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    tui: TUIConfig = field(default_factory=TUIConfig)
    model: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CorterConfig":
        hpo_raw = dict(raw.get("hpo", {}))
        xai_raw = dict(raw.get("xai", {}))
        tui_raw = dict(raw.get("tui", {}))
        return cls(
            task=str(raw.get("task", "auto")),
            target_column=raw.get("target_column"),
            hpo=HPOConfig(**{k: v for k, v in hpo_raw.items() if k in HPOConfig.__dataclass_fields__}),
            xai=XAIConfig(**{k: v for k, v in xai_raw.items() if k in XAIConfig.__dataclass_fields__}),
            tui=TUIConfig(**{k: v for k, v in tui_raw.items() if k in TUIConfig.__dataclass_fields__}),
            model=dict(raw.get("model", {})),
        )


def load_config(path: Union[str, Path]) -> CorterConfig:
    """Load YAML configuration into :class:`CorterConfig`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")
    return CorterConfig.from_mapping(raw)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


_CANONICAL_TARGET_NAMES = ("target", "label", "y")


def _detect_target_column(df: pd.DataFrame, configured: Optional[str] = None) -> str:
    """
    Resolve the target column name.

    If ``configured`` is set, that column is used. Otherwise prefers columns named
    ``target``, ``label``, or ``y`` (case-insensitive), then falls back to the last column.
    """
    if configured is not None and str(configured).strip():
        name = str(configured).strip()
        if name not in df.columns:
            raise ValueError(
                f"Target column '{name}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        return name

    by_lower = {str(col).lower(): col for col in df.columns}
    for candidate in _CANONICAL_TARGET_NAMES:
        if candidate in by_lower:
            return str(by_lower[candidate])

    return str(df.columns[-1])


def _infer_task(y: np.ndarray) -> str:
    if y.dtype.kind in {"i", "u", "b", "f"} and len(np.unique(y)) <= max(20, int(0.05 * len(y))):
        return "classification"
    return "regression"


def _default_scorer(task: str) -> str:
    return "f1_weighted" if task == "classification" else "neg_mean_squared_error"


def _format_report_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _build_results_report_html(result: Mapping[str, Any], version: str = __version__) -> str:
    """Build a standalone HTML optimization report."""
    task = html_module.escape(str(result.get("task", "")))
    best_score = float(result["best_cv_score"])
    params = result.get("best_params", {})
    insights: Sequence[str] = result.get("insights", [])
    holdout = result.get("holdout_metrics") or {}
    drift = result.get("drift") or {}
    trials_df = result.get("trials")
    feature_report = result.get("feature_report")

    params_rows = "".join(
        f"<tr><td>{html_module.escape(str(k))}</td>"
        f"<td>{html_module.escape(_format_report_value(v))}</td></tr>"
        for k, v in params.items()
    )
    if not params_rows:
        params_rows = "<tr><td colspan=\"2\">No parameters recorded.</td></tr>"

    insights_html = "".join(
        f"<li>{html_module.escape(str(line))}</li>" for line in insights
    ) or "<li>No insights generated.</li>"

    metrics_rows = "".join(
        f"<tr><td>{html_module.escape(str(k))}</td>"
        f"<td>{html_module.escape(_format_report_value(v))}</td></tr>"
        for k, v in holdout.items()
    )
    if not metrics_rows:
        metrics_rows = "<tr><td colspan=\"2\">No holdout metrics.</td></tr>"

    drift_rows = "".join(
        f"<tr><td>{html_module.escape(str(k))}</td>"
        f"<td>{float(v):.4f}</td></tr>"
        for k, v in sorted(drift.items(), key=lambda x: x[1], reverse=True)[:15]
    )
    drift_section = ""
    if drift_rows:
        drift_section = f"""
        <section>
            <h2>Drift (top features)</h2>
            <table><thead><tr><th>Feature</th><th>KS statistic</th></tr></thead>
            <tbody>{drift_rows}</tbody></table>
        </section>
        """

    feature_section = ""
    if feature_report is not None and not feature_report.empty:
        fr = feature_report.head(15)
        cols = ["feature", "perm_mean"]
        if "shap_importance" in fr.columns:
            cols.append("shap_importance")
        header = "".join(f"<th>{html_module.escape(c)}</th>" for c in cols)
        body = ""
        for _, row in fr.iterrows():
            cells = "".join(
                f"<td>{html_module.escape(_format_report_value(row[c]))}</td>" for c in cols
            )
            body += f"<tr>{cells}</tr>"
        feature_section = f"""
        <section>
            <h2>Feature importance (top {len(fr)})</h2>
            <table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>
        </section>
        """

    n_trials = len(trials_df) if trials_df is not None and hasattr(trials_df, "__len__") else 0
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Corter — Optimization Report</title>
  <style>
    :root {{ --bg: #141210; --fg: #F9EFE6; --muted: rgba(249,239,230,0.55); --border: rgba(249,239,230,0.14); }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
      background: var(--bg); color: var(--fg); margin: 0; padding: 48px 24px;
      line-height: 1.5;
    }}
    .wrap {{ max-width: 720px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; font-weight: 600; letter-spacing: -0.03em; margin: 0 0 8px; }}
    .meta {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 40px; }}
    h2 {{
      font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em;
      color: var(--muted); margin: 32px 0 16px;
    }}
    .score {{
      font-size: 2.5rem; font-weight: 600; letter-spacing: -0.04em;
      margin: 8px 0 24px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9375rem; }}
    th, td {{
      text-align: left; padding: 12px 14px; border-bottom: 1px solid var(--border);
    }}
    th {{ color: var(--muted); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; }}
    ul {{ margin: 0; padding-left: 1.25rem; }}
    li {{ margin: 10px 0; color: var(--muted); }}
    li strong {{ color: var(--fg); }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Corter optimization report</h1>
    <p class="meta">Generated {html_module.escape(generated)} · v{html_module.escape(version)} · task: {task} · trials: {n_trials}</p>

    <h2>Best CV score</h2>
    <p class="score">{best_score:.4f}</p>

    <section>
      <h2>Best parameters</h2>
      <table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>
      <tbody>{params_rows}</tbody></table>
    </section>

    <section>
      <h2>Holdout metrics</h2>
      <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>{metrics_rows}</tbody></table>
    </section>

    {feature_section}

    <section>
      <h2>XAI insights</h2>
      <ul>{insights_html}</ul>
    </section>

    {drift_section}
  </div>
</body>
</html>
"""


def _write_pdf_from_html(html_content: str, pdf_path: Path) -> bool:
    """Render HTML to PDF using xhtml2pdf or weasyprint if available."""
    try:
        from xhtml2pdf import pisa

        with pdf_path.open("wb") as dest:
            status = pisa.CreatePDF(html_content, dest=dest, encoding="utf-8")
        return status.err == 0
    except ImportError:
        pass
    except Exception:
        return False

    try:
        from weasyprint import HTML

        HTML(string=html_content).write_pdf(str(pdf_path))
        return True
    except ImportError:
        return False
    except Exception:
        return False


def _task_from_params(params: Dict[str, Any]) -> str:
    return str(params.pop("_task", "classification"))


def _is_tree_estimator(est: BaseEstimator) -> bool:
    """Whether SHAP TreeExplainer is appropriate for this fitted estimator."""
    if hasattr(est, "tree_") or hasattr(est, "estimators_"):
        return True
    tree_types = (
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor",
        "CatBoostClassifier",
        "CatBoostRegressor",
        "Booster",
    )
    return type(est).__name__ in tree_types or hasattr(est, "get_booster")


def _resolve_estimator(model_cfg: Mapping[str, Any]) -> BaseEstimator:
    """Instantiate estimator from config ``{name, params}`` (sklearn or boosting libs)."""
    name = str(model_cfg.get("name", "random_forest")).lower()
    params = dict(model_cfg.get("params", {}))
    task = _task_from_params(params)
    classification = task == "classification"

    if name in {"rf", "random_forest", "randomforestclassifier", "randomforestregressor"}:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if classification else RandomForestRegressor
        return cls(**params)
    if name in {"gbm", "gradient_boosting", "gradientboostingclassifier", "gradientboostingregressor"}:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        cls = GradientBoostingClassifier if classification else GradientBoostingRegressor
        return cls(**params)
    if name in {"xgboost", "xgb", "xgbclassifier", "xgbregressor"}:
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            ) from exc
        cls = XGBClassifier if classification else XGBRegressor
        return cls(**params)
    if name in {"lightgbm", "lgbm", "lgb", "lgbmclassifier", "lgbmregressor"}:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            ) from exc
        cls = LGBMClassifier if classification else LGBMRegressor
        return cls(**params)
    if name in {"catboost", "cb", "catboostclassifier", "catboostregressor"}:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise ImportError(
                "CatBoost is not installed. Install with: pip install catboost"
            ) from exc
        cls = CatBoostClassifier if classification else CatBoostRegressor
        params.setdefault("verbose", False)
        return cls(**params)
    if name in {"logreg", "logistic", "logisticregression"}:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**params)
    if name in {"ridge", "ridgeregression"}:
        from sklearn.linear_model import Ridge

        return Ridge(**params)
    if name in {"svc", "svm"}:
        from sklearn.svm import SVC

        return SVC(**params)

    raise ValueError(f"Unknown model.name '{name}'. Extend _resolve_estimator or pass an estimator instance.")


def _holdout_metrics(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
) -> Dict[str, float]:
    pred = model.predict(X_test)
    if task == "classification":
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        }
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
    }


# ---------------------------------------------------------------------------
# HPO — Hyperparameter Autopilot
# ---------------------------------------------------------------------------


def _evaluate_trial_standalone(
    trial_id: int,
    seed: int,
    space: Mapping[str, Any],
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    scoring: Optional[str],
    task: str,
    random_state: int,
) -> Tuple[int, Dict[str, Any], float, float]:
    """
    Standalone function for parallel trial evaluation.
    Must be at module level to be picklable by joblib.
    """
    t0 = time.perf_counter()
    
    # Create trial-specific RNG for reproducibility
    trial_rng = np.random.default_rng(seed)
    
    # Sample parameters using trial-specific RNG
    params: Dict[str, Any] = {}
    for key, spec in space.items():
        if isinstance(spec, list):
            params[key] = spec[int(trial_rng.integers(0, len(spec)))]
        elif isinstance(spec, dict):
            low, high = float(spec["low"]), float(spec["high"])
            if spec.get("log", False):
                params[key] = float(np.exp(trial_rng.uniform(np.log(low), np.log(high))))
            elif spec.get("type") == "int":
                params[key] = int(trial_rng.integers(int(low), int(high) + 1))
            else:
                params[key] = float(trial_rng.uniform(low, high))
        else:
            params[key] = spec
    
    # Evaluate parameters
    est = clone(estimator)
    est.set_params(**params)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
    scoring_metric = scoring or _default_scorer(task)
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    ) if task == "classification" else cv_folds
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring_metric, n_jobs=1)
    
    score = float(np.mean(scores))
    elapsed = time.perf_counter() - t0
    
    return trial_id, params, score, elapsed


class HyperparameterAutopilot:
    """
    Autonomous hyperparameter search with trial logging and convergence tracking.

    Strategies:
      - ``random``: Random search over declared bounds
      - ``bayesian``: Model-based search via Optuna (TPE sampler)
      - ``scipy_de``: Global optimization via ``scipy.optimize.differential_evolution``
      - ``scipy_local``: Local refinement via ``scipy.optimize.minimize`` (Nelder-Mead)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        config: HPOConfig,
        task: str,
        console: Optional[Console] = None,
    ) -> None:
        self.estimator = estimator
        self.config = config
        self.task = task
        self.console = console or Console()
        self.trials: pd.DataFrame = pd.DataFrame()
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = float("-inf")
        self.best_estimator_: Optional[BaseEstimator] = None
        self._rng = np.random.default_rng(config.random_state)
        # Early stopping tracking
        self._no_improvement_count: int = 0
        self._best_score_history: List[float] = []

    def _cv_score(self, params: Mapping[str, Any], X: np.ndarray, y: np.ndarray) -> float:
        est = clone(self.estimator)
        est.set_params(**params)
        pipe = Pipeline([("scaler", StandardScaler()), ("model", est)])
        scoring = self.config.scoring or _default_scorer(self.task)
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        ) if self.task == "classification" else self.config.cv_folds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores))

    def _sample_params(self, space: Mapping[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for key, spec in space.items():
            if isinstance(spec, list):
                params[key] = spec[int(self._rng.integers(0, len(spec)))]
            elif isinstance(spec, dict):
                low, high = float(spec["low"]), float(spec["high"])
                if spec.get("log", False):
                    params[key] = float(np.exp(self._rng.uniform(np.log(low), np.log(high))))
                elif spec.get("type") == "int":
                    params[key] = int(self._rng.integers(int(low), int(high) + 1))
                else:
                    params[key] = float(self._rng.uniform(low, high))
            else:
                params[key] = spec
        return params

    def _log_trial(self, trial_id: int, params: Dict[str, Any], score: float, elapsed: float) -> None:
        row = {"trial": trial_id, "score": score, "elapsed_s": elapsed, **params}
        self.trials = pd.concat([self.trials, pd.DataFrame([row])], ignore_index=True)
        if score > self.best_score_:
            self.best_score_ = score
            self.best_params_ = dict(params)
            self.best_estimator_ = clone(self.estimator).set_params(**params)

    def _check_early_stop(self) -> bool:
        """Check if early stopping criteria met."""
        if not self.config.enable_early_stop or len(self._best_score_history) < 2:
            return False
        
        recent_improvement = self._best_score_history[-1] - self._best_score_history[-2]
        if recent_improvement < self.config.min_delta:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
        
        return self._no_improvement_count >= self.config.patience

    def fit(self, X: np.ndarray, y: np.ndarray, on_trial: Optional[Callable[[int, float, Dict], None]] = None) -> "HyperparameterAutopilot":
        space = self.config.search_space
        if not space:
            # sensible default space for tree models
            space = {
                "n_estimators": {"low": 50, "high": 300, "type": "int"},
                "max_depth": [None, 4, 8, 12, 16],
                "min_samples_leaf": {"low": 1, "high": 8, "type": "int"},
            }

        strategy = self.config.strategy.lower()
        if strategy == "random":
            self._fit_random(X, y, space, on_trial)
        elif strategy in {"bayesian", "bo", "tpe"}:
            self._fit_bayesian(X, y, space, on_trial)
        elif strategy in {"scipy_de", "de"}:
            self._fit_scipy_de(X, y, space, on_trial)
        elif strategy in {"scipy_local", "local"}:
            self._fit_scipy_local(X, y, space, on_trial)
        else:
            raise ValueError(f"Unknown HPO strategy '{self.config.strategy}'")

        if self.best_estimator_ is not None:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", self.best_estimator_)])
            pipe.fit(X, y)
            self.best_estimator_ = pipe
        return self

    def _fit_random(
        self,
        X: np.ndarray,
        y: np.ndarray,
        space: Mapping[str, Any],
        on_trial: Optional[Callable[[int, float, Dict], None]],
    ) -> None:
        # Use parallel execution if configured
        if self.config.parallel_trials > 1:
            self._fit_random_parallel(X, y, space, on_trial)
        else:
            self._fit_random_sequential(X, y, space, on_trial)

    def _fit_random_sequential(
        self,
        X: np.ndarray,
        y: np.ndarray,
        space: Mapping[str, Any],
        on_trial: Optional[Callable[[int, float, Dict], None]],
    ) -> None:
        """Sequential random search (original implementation)."""
        for trial in range(1, self.config.n_trials + 1):
            t0 = time.perf_counter()
            params = self._sample_params(space)
            score = self._cv_score(params, X, y)
            self._log_trial(trial, params, score, time.perf_counter() - t0)
            self._best_score_history.append(self.best_score_)
            if on_trial:
                on_trial(trial, score, params)
            
            # Check early stopping
            if self._check_early_stop():
                self.console.print(f"[yellow]Early stopping at trial {trial}/{self.config.n_trials} (no improvement for {self.config.patience} trials)[/]")
                break

    def _fit_random_parallel(
        self,
        X: np.ndarray,
        y: np.ndarray,
        space: Mapping[str, Any],
        on_trial: Optional[Callable[[int, float, Dict], None]],
    ) -> None:
        """Parallel random search with batch evaluation."""
        try:
            batch_size = self.config.parallel_trials
            
            # Process trials in batches
            for batch_start in range(1, self.config.n_trials + 1, batch_size):
                batch_end = min(batch_start + batch_size, self.config.n_trials + 1)
                trial_ids = list(range(batch_start, batch_end))
                
                # Generate seeds for reproducibility
                seeds = [self.config.random_state + tid for tid in trial_ids]
                
                # Parallel evaluation using standalone function
                self.console.print(f"[dim]Evaluating trials {batch_start}-{batch_end-1} in parallel...[/]")
                results = Parallel(n_jobs=self.config.n_jobs, backend='loky')(
                    delayed(_evaluate_trial_standalone)(
                        tid, seed, space, self.estimator, X, y,
                        self.config.cv_folds, self.config.scoring,
                        self.task, self.config.random_state
                    ) for tid, seed in zip(trial_ids, seeds)
                )
                
                # Log results in order
                for trial_id, params, score, elapsed in sorted(results, key=lambda x: x[0]):
                    self._log_trial(trial_id, params, score, elapsed)
                    self._best_score_history.append(self.best_score_)
                    if on_trial:
                        on_trial(trial_id, score, params)
                
                # Check early stopping after each batch
                if self._check_early_stop():
                    self.console.print(f"[yellow]Early stopping at trial {trial_id}/{self.config.n_trials} (no improvement for {self.config.patience} trials)[/]")
                    break
        
        except Exception as e:
            # Fallback to sequential execution on any error
            self.console.print(f"[yellow]Parallel execution failed ({type(e).__name__}), falling back to sequential mode[/]")
            self._fit_random_sequential(X, y, space, on_trial)

    def _suggest_param(self, trial: Any, name: str, spec: Any) -> Any:
        """Map a search-space entry to an Optuna trial suggestion."""
        if isinstance(spec, list):
            return trial.suggest_categorical(name, list(spec))
        if isinstance(spec, dict) and "low" in spec and "high" in spec:
            low, high = spec["low"], spec["high"]
            if spec.get("type") == "int":
                return trial.suggest_int(name, int(low), int(high))
            return trial.suggest_float(
                name,
                float(low),
                float(high),
                log=bool(spec.get("log", False)),
            )
        return spec

    def _fit_bayesian(
        self,
        X: np.ndarray,
        y: np.ndarray,
        space: Mapping[str, Any],
        on_trial: Optional[Callable[[int, float, Dict], None]],
    ) -> None:
        """Bayesian-style optimization using Optuna's TPE sampler."""
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Bayesian HPO requires Optuna. Install with: pip install optuna"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            t0 = time.perf_counter()
            params = {key: self._suggest_param(trial, key, spec) for key, spec in space.items()}
            score = self._cv_score(params, X, y)
            trial_id = trial.number + 1
            self._log_trial(trial_id, params, score, time.perf_counter() - t0)
            self._best_score_history.append(self.best_score_)
            if on_trial:
                on_trial(trial_id, score, params)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.config.random_state),
        )

        def _early_stop_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if self._check_early_stop():
                study.stop()
                self.console.print(
                    f"[yellow]Early stopping at trial {trial.number + 1}/{self.config.n_trials} "
                    f"(no improvement for {self.config.patience} trials)[/]"
                )

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            callbacks=[_early_stop_callback],
            show_progress_bar=False,
        )

    def _continuous_bounds(self, space: Mapping[str, Any]) -> Tuple[List[Tuple[float, float]], List[str], List[Mapping]]:
        bounds: List[Tuple[float, float]] = []
        keys: List[str] = []
        meta: List[Mapping] = []
        for key, spec in space.items():
            if isinstance(spec, dict) and "low" in spec and "high" in spec:
                bounds.append((float(spec["low"]), float(spec["high"])))
                keys.append(key)
                meta.append(spec)
        return bounds, keys, meta

    def _vector_to_params(self, vector: np.ndarray, keys: Sequence[str], meta: Sequence[Mapping], space: Mapping) -> Dict[str, Any]:
        params = self._sample_params({k: v for k, v in space.items() if k not in keys})
        for i, key in enumerate(keys):
            spec = meta[i]
            val = float(vector[i])
            if spec.get("type") == "int":
                params[key] = int(round(val))
            else:
                params[key] = val
        return params

    def _fit_scipy_de(
        self,
        X: np.ndarray,
        y: np.ndarray,
        space: Mapping[str, Any],
        on_trial: Optional[Callable[[int, float, Dict], None]],
    ) -> None:
        bounds, keys, meta = self._continuous_bounds(space)
        if not bounds:
            return self._fit_random(X, y, space, on_trial)

        trial_counter = {"n": 0}

        def objective(vec: np.ndarray) -> float:
            trial_counter["n"] += 1
            t0 = time.perf_counter()
            params = self._vector_to_params(vec, keys, meta, space)
            score = self._cv_score(params, X, y)
            self._log_trial(trial_counter["n"], params, score, time.perf_counter() - t0)
            if on_trial:
                on_trial(trial_counter["n"], score, params)
            return -score

        maxiter = max(1, self.config.n_trials // max(1, len(bounds)))
        differential_evolution(
            objective,
            bounds=bounds,
            maxiter=maxiter,
            seed=self.config.random_state,
            polish=True,
            updating="deferred",
            workers=1,
        )

    def _fit_scipy_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        space: Mapping[str, Any],
        on_trial: Optional[Callable[[int, float, Dict], None]],
    ) -> None:
        bounds, keys, meta = self._continuous_bounds(space)
        if not bounds:
            return self._fit_random(X, y, space, on_trial)

        x0 = np.array([(lo + hi) / 2 for lo, hi in bounds])
        trial_counter = {"n": 0}

        def objective(vec: np.ndarray) -> float:
            trial_counter["n"] += 1
            t0 = time.perf_counter()
            params = self._vector_to_params(vec, keys, meta, space)
            score = self._cv_score(params, X, y)
            self._log_trial(trial_counter["n"], params, score, time.perf_counter() - t0)
            if on_trial:
                on_trial(trial_counter["n"], score, params)
            return -score

        for _ in range(self.config.n_trials):
            minimize(objective, x0=x0 + self._rng.normal(0, 0.05, size=x0.shape), method="Nelder-Mead", options={"maxiter": 40})


# ---------------------------------------------------------------------------
# XAI — Semantic Diagnostics
# ---------------------------------------------------------------------------


class SemanticDiagnostics:
    """
    Model-agnostic diagnostics with human-readable semantic summaries.

    Computes permutation importance, mutual information, optional drift vs. reference,
    and emits narrative insight strings suitable for the TUI or logs.
    """

    def __init__(self, config: XAIConfig, task: str, console: Optional[Console] = None) -> None:
        self.config = config
        self.task = task
        self.console = console or Console()
        self.feature_report_: pd.DataFrame = pd.DataFrame()
        self.insights_: List[str] = []
        self.drift_scores_: Dict[str, float] = {}
        # Computation cache for expensive operations
        self._perm_cache: Dict[str, Any] = {}
        self._mi_cache: Dict[str, np.ndarray] = {}

    def _cache_key(self, X: np.ndarray, y: np.ndarray) -> str:
        """Generate cache key from data hash."""
        data_hash = hashlib.md5(X.tobytes() + y.tobytes()).hexdigest()
        return f"{data_hash}_{self.config.permutation_repeats}"

    def analyze(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
        X_reference: Optional[np.ndarray] = None,
    ) -> "SemanticDiagnostics":
        n_features = X.shape[1]
        names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(n_features)]

        est = model.named_steps["model"] if isinstance(model, Pipeline) else model
        fitted = model if isinstance(model, Pipeline) else model

        # Check cache for permutation importance
        cache_key = self._cache_key(X, y)
        if cache_key in self._perm_cache:
            perm = self._perm_cache[cache_key]
            self.console.print("[dim]Using cached permutation importance[/]")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                perm = permutation_importance(
                    fitted,
                    X,
                    y,
                    n_repeats=self.config.permutation_repeats,
                    random_state=0,
                    n_jobs=-1,
                )
            self._perm_cache[cache_key] = perm

        # Check cache for mutual information
        if cache_key in self._mi_cache:
            mi = self._mi_cache[cache_key]
            self.console.print("[dim]Using cached mutual information[/]")
        else:
            mi_func = mutual_info_classif if self.task == "classification" else mutual_info_regression
            mi = mi_func(X, y, random_state=0)
            self._mi_cache[cache_key] = mi

        # Build initial report
        report_data = {
            "feature": names,
            "perm_mean": perm.importances_mean,
            "perm_std": perm.importances_std,
            "mutual_info": mi,
        }

        # Add SHAP analysis if enabled
        if self.config.use_shap:
            try:
                import shap
                
                # Subsample for efficiency
                sample_size = min(self.config.shap_sample_size, len(X))
                sample_idx = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[sample_idx]
                
                self.console.print(f"[dim]Computing SHAP values (sample size: {sample_size})...[/]")
                
                # Create appropriate explainer based on model type
                if _is_tree_estimator(est):
                    explainer = shap.TreeExplainer(est)
                    shap_values = explainer.shap_values(X_sample)
                else:
                    # Other models - use KernelExplainer (slower but model-agnostic)
                    background = shap.sample(X_sample, min(50, len(X_sample)))
                    explainer = shap.KernelExplainer(est.predict, background)
                    shap_values = explainer.shap_values(X_sample)
                
                # Handle multi-class classification (shap_values is a list)
                if isinstance(shap_values, list):
                    # For multi-class, average absolute SHAP values across classes
                    shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    # For binary classification or regression
                    shap_importance = np.abs(shap_values).mean(axis=0)
                
                report_data["shap_importance"] = shap_importance
                self.console.print("[dim]SHAP analysis complete[/]")
                
            except ImportError:
                self.console.print("[yellow]SHAP package not available, skipping SHAP analysis[/]")
            except Exception as e:
                self.console.print(f"[yellow]SHAP analysis failed: {type(e).__name__}, skipping[/]")

        report = pd.DataFrame(report_data).sort_values("perm_mean", ascending=False)

        if self.config.include_correlations and n_features <= 64:
            corr_target = []
            for j in range(n_features):
                if self.task == "classification":
                    corr_target.append(stats.pointbiserialr(X[:, j], y)[0])
                else:
                    corr_target.append(float(np.corrcoef(X[:, j], y)[0, 1]))
            report = report.merge(
                pd.DataFrame({"feature": names, "target_corr": corr_target}),
                on="feature",
                how="left",
            )

        self.feature_report_ = report.reset_index(drop=True)
        self.insights_ = self._build_insights(report, est)
        if X_reference is not None:
            self.drift_scores_ = self._population_drift(X_reference, X, names)
            self.insights_.extend(self._drift_narrative(self.drift_scores_))
        return self

    def _population_drift(
        self,
        ref: np.ndarray,
        cur: np.ndarray,
        names: Sequence[str],
    ) -> Dict[str, float]:
        drift: Dict[str, float] = {}
        for j, name in enumerate(names):
            ks = stats.ks_2samp(ref[:, j], cur[:, j]).statistic
            drift[name] = float(ks)
        return drift

    def _build_insights(self, report: pd.DataFrame, estimator: BaseEstimator) -> List[str]:
        insights: List[str] = []
        top = report.head(self.config.top_k_features)
        if top.empty:
            return ["No features available for diagnostic analysis."]

        leader = top.iloc[0]
        insights.append(
            f"Primary driver: '{leader['feature']}' "
            f"(permutation Δ={leader['perm_mean']:.4f}, MI={leader['mutual_info']:.4f})."
        )

        if len(top) >= 2:
            runner = top.iloc[1]
            gap = leader["perm_mean"] - runner["perm_mean"]
            if gap < 0.01:
                insights.append(
                    f"'{leader['feature']}' and '{runner['feature']}' are co-dominant; "
                    "consider interaction terms or dimensionality reduction."
                )
            else:
                insights.append(f"Secondary signal from '{runner['feature']}' (Δ={runner['perm_mean']:.4f}).")

        weak = report.tail(3)
        low_imp = weak[weak["perm_mean"] <= 0]
        if not low_imp.empty:
            cols = ", ".join(f"'{f}'" for f in low_imp["feature"].tolist())
            insights.append(f"Low-impact features ({cols}) are candidates for pruning.")

        if hasattr(estimator, "feature_importances_"):
            fi = np.asarray(estimator.feature_importances_)
            if fi.std() > 0:
                concentration = float(fi.max() / fi.sum())
                if concentration > 0.5:
                    insights.append(
                        f"Tree ensemble is highly concentrated ({concentration:.0%} mass in top split feature)."
                    )

        if "target_corr" in report.columns:
            signed = report.assign(abs_corr=report["target_corr"].abs()).sort_values("abs_corr", ascending=False)
            best = signed.iloc[0]
            direction = "positively" if best["target_corr"] >= 0 else "negatively"
            insights.append(
                f"Strongest linear association: '{best['feature']}' is {direction} "
                f"correlated with the target (r={best['target_corr']:.3f})."
            )

        return insights

    def _drift_narrative(self, drift: Mapping[str, float]) -> List[str]:
        lines: List[str] = []
        flagged = [(k, v) for k, v in drift.items() if v >= self.config.drift_threshold]
        flagged.sort(key=lambda kv: kv[1], reverse=True)
        if not flagged:
            lines.append("Population drift: all monitored features stable vs. reference.")
        else:
            names = ", ".join(f"{k} (KS={v:.2f})" for k, v in flagged[:5])
            lines.append(f"Population drift detected: {names}.")
        return lines


# ---------------------------------------------------------------------------
# TUI — Dashboard
# ---------------------------------------------------------------------------


class CorterDashboard:
    """Rich terminal dashboard for live HPO progress and XAI summaries."""

    def __init__(self, config: TUIConfig, console: Optional[Console] = None) -> None:
        self.config = config
        self.console = console or Console(width=config.width)
        self._layout = self._build_layout()
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]HPO[/]"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )
        self._trial_task = self._progress.add_task("trials", total=1)
        self._status = "Idle"
        self._best_score: Optional[float] = None
        self._metrics: Dict[str, float] = {}
        self._insights: List[str] = []
        self._trials: pd.DataFrame = pd.DataFrame()
        self._score_history: List[float] = []

    def _create_sparkline(self, values: List[float], width: int = 40) -> str:
        """Create ASCII sparkline from values using Unicode block characters."""
        if not values or len(values) < 2:
            return "─" * width
        
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return "─" * width
        
        # Unicode block characters for sparkline
        blocks = "▁▂▃▄▅▆▇█"
        normalized = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Resample to fit width
        step = len(values) / width
        sparkline = ""
        for i in range(width):
            idx = min(int(i * step), len(values) - 1)
            block_idx = min(int(normalized[idx] * (len(blocks) - 1)), len(blocks) - 1)
            sparkline += blocks[block_idx]
        
        return sparkline

    def _build_layout(self) -> Layout:
        layout = Layout(name="root")
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="hpo", ratio=3),
            Layout(name="xai", ratio=2),
        )
        return layout

    def _header_panel(self) -> Panel:
        title = Text("Corter", style="bold magenta")
        subtitle = Text(" Autonomous ML Optimization", style="dim")
        status = Text(f"\n{self._status}", style="bold green" if "complete" in self._status.lower() else "yellow")
        return Panel(Align.center(Group(title, subtitle, status)), border_style="magenta", box=box.HEAVY)

    def _hpo_table(self) -> Table:
        table = Table(title="Hyperparameter Trials", box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Score", justify="right", style="green")
        table.add_column("Δt (s)", justify="right")
        table.add_column("Params", overflow="fold")

        if self._trials.empty:
            table.add_row("—", "—", "—", "awaiting trials…")
            return table

        # Add sparkline caption if we have score history
        if len(self._score_history) > 1:
            sparkline = self._create_sparkline(self._score_history, width=50)
            min_score = min(self._score_history)
            max_score = max(self._score_history)
            table.caption = f"Score trend: {sparkline} [{min_score:.4f} → {max_score:.4f}]"

        tail = self._trials.tail(8)
        param_cols = [c for c in tail.columns if c not in {"trial", "score", "elapsed_s"}]
        for _, row in tail.iterrows():
            params = ", ".join(f"{c}={row[c]}" for c in param_cols[:4])
            table.add_row(
                str(int(row["trial"])),
                f"{row['score']:.4f}",
                f"{row.get('elapsed_s', 0):.2f}",
                params,
            )
        return table

    def _xai_panel(self) -> Panel:
        table = Table(box=box.MINIMAL, expand=True, show_header=True)
        table.add_column("Insight", style="white")
        if not self._insights:
            table.add_row("Run XAI diagnostics to populate semantic insights.")
        else:
            for line in self._insights[:12]:
                table.add_row(line)

        if self._metrics:
            metric_line = "  |  ".join(f"{k}: {v:.4f}" for k, v in self._metrics.items())
            table.add_row(f"[bold]Hold-out[/] {metric_line}")

        return Panel(table, title="Semantic Diagnostics (XAI)", border_style="blue")

    def _footer_panel(self) -> Panel:
        best = f"{self._best_score:.4f}" if self._best_score is not None else "—"
        return Panel(
            f"Best CV score: [bold green]{best}[/]  •  Refresh: {self.config.refresh_hz:.1f} Hz  •  corter v{__version__}",
            style="dim",
        )

    def render(self) -> Layout:
        self._layout["header"].update(self._header_panel())
        self._layout["hpo"].update(
            Panel(
                Group(self._progress, self._hpo_table()),
                title="Hyperparameter Autopilot (HPO)",
                border_style="cyan",
            )
        )
        self._layout["xai"].update(self._xai_panel())
        self._layout["footer"].update(self._footer_panel())
        return self._layout

    def set_phase(self, status: str, total_trials: int) -> None:
        self._status = status
        self._progress.reset(self._trial_task, total=max(1, total_trials), completed=0)

    def on_trial(self, trial: int, score: float, trials: pd.DataFrame) -> None:
        self._best_score = float(trials["score"].max()) if not trials.empty else score
        self._trials = trials.copy()
        self._score_history.append(score)
        self._progress.update(self._trial_task, completed=trial)

    def set_insights(self, insights: Sequence[str], metrics: Optional[Mapping[str, float]] = None) -> None:
        self._insights = list(insights)
        if metrics:
            self._metrics = dict(metrics)

    @property
    def live(self) -> Live:
        return Live(
            self.render(),
            console=self.console,
            refresh_per_second=self.config.refresh_hz,
            screen=False,
        )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Corter:
    """
    End-to-end autonomous ML run: configure → HPO → fit → XAI → dashboard.

    Example::

        core = Corter.from_yaml("config.yaml")
        result = core.run("data.csv")
    """

    def __init__(
        self,
        config: CorterConfig,
        estimator: Optional[BaseEstimator] = None,
        console: Optional[Console] = None,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self.estimator = estimator or self._build_estimator()
        self.hpo: Optional[HyperparameterAutopilot] = None
        self.xai: Optional[SemanticDiagnostics] = None
        self.dashboard = CorterDashboard(config.tui, self.console)
        self.result_: Dict[str, Any] = {}

    @classmethod
    def from_yaml(cls, path: Union[str, Path], **kwargs: Any) -> "Corter":
        return cls(load_config(path), **kwargs)

    def _build_estimator(self) -> BaseEstimator:
        cfg = dict(self.config.model)
        task = self.config.task
        if task == "auto":
            task = "classification"
        params = dict(cfg.get("params", {}))
        params["_task"] = task
        cfg["params"] = params
        return _resolve_estimator(cfg)

    def _load_xy(
        self,
        data: Union[str, Path, pd.DataFrame],
        test_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], str]:
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        target = _detect_target_column(df, self.config.target_column)
        if not self.config.target_column:
            self.config.target_column = target

        feature_names = [c for c in df.columns if c != target]
        X = df[feature_names].select_dtypes(include=[np.number]).to_numpy(dtype=float)
        y = df[target].to_numpy()

        if len(feature_names) != X.shape[1]:
            feature_names = [f"f{i}" for i in range(X.shape[1])]

        task = self.config.task
        if task == "auto":
            task = _infer_task(y)
        if task == "classification" and not is_classifier(self.estimator):
            # align simple defaults when user picked a regressor for classification labels
            pass

        n = len(y)
        idx = np.arange(n)
        self._rng = np.random.default_rng(self.config.hpo.random_state)
        self._rng.shuffle(idx)
        split = int(n * (1 - test_size))
        tr, te = idx[:split], idx[split:]
        return X[tr], X[te], y[tr], y[te], feature_names, task

    def run(
        self,
        data: Union[str, Path, pd.DataFrame],
        test_size: float = 0.2,
        reference_data: Optional[Union[str, Path, pd.DataFrame]] = None,
        export_reports: bool = True,
        export_pdf: bool = False,
        report_html_path: Union[str, Path] = "results_report.html",
        report_pdf_path: Union[str, Path] = "results_report.pdf",
    ) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test, feature_names, task = self._load_xy(data, test_size)

        X_ref = None
        if reference_data is not None:
            ref_df = pd.read_csv(reference_data) if isinstance(reference_data, (str, Path)) else reference_data
            target = _detect_target_column(ref_df, self.config.target_column)
            X_ref = ref_df.drop(columns=[target]).select_dtypes(include=[np.number]).to_numpy(dtype=float)

        self.hpo = HyperparameterAutopilot(self.estimator, self.config.hpo, task, self.console)
        self.xai = SemanticDiagnostics(self.config.xai, task, self.console)

        def trial_callback(trial: int, score: float, _params: Dict) -> None:
            if self.hpo is not None:
                self.dashboard.on_trial(trial, score, self.hpo.trials)

        if self.config.tui.show_live:
            self.dashboard.set_phase("HPO running", self.config.hpo.n_trials)
            with self.dashboard.live:
                self.hpo.fit(X_train, y_train, on_trial=trial_callback)
                self.dashboard.set_phase("XAI diagnostics", self.config.hpo.n_trials)
                assert self.hpo.best_estimator_ is not None
                self.xai.analyze(self.hpo.best_estimator_, X_train, y_train, feature_names, X_ref)
                metrics = _holdout_metrics(self.hpo.best_estimator_, X_test, y_test, task)
                self.dashboard.set_insights(self.xai.insights_, metrics)
                self.dashboard.set_phase("Run complete", self.config.hpo.n_trials)
                time.sleep(0.5)
        else:
            self.hpo.fit(X_train, y_train, on_trial=trial_callback)
            assert self.hpo.best_estimator_ is not None
            self.xai.analyze(self.hpo.best_estimator_, X_train, y_train, feature_names, X_ref)
            metrics = _holdout_metrics(self.hpo.best_estimator_, X_test, y_test, task)

        self.result_ = {
            "task": task,
            "best_params": self.hpo.best_params_,
            "best_cv_score": self.hpo.best_score_,
            "holdout_metrics": metrics,
            "trials": self.hpo.trials,
            "feature_report": self.xai.feature_report_,
            "insights": self.xai.insights_,
            "drift": self.xai.drift_scores_,
            "model": self.hpo.best_estimator_,
        }

        if export_reports:
            report_paths = self.export_results_reports(
                html_path=report_html_path,
                pdf_path=report_pdf_path,
                export_pdf=export_pdf,
            )
            self.result_["report_paths"] = report_paths

        return self.result_

    def export_html_report(self, path: Union[str, Path] = "results_report.html") -> Path:
        """Write optimization results to a standalone HTML report."""
        if not self.result_:
            raise RuntimeError("No run results to export. Call run() first.")
        path = Path(path)
        path.write_text(_build_results_report_html(self.result_), encoding="utf-8")
        return path

    def export_pdf_report(self, path: Union[str, Path] = "results_report.pdf") -> Optional[Path]:
        """Write PDF report from the HTML template; returns path if successful."""
        if not self.result_:
            raise RuntimeError("No run results to export. Call run() first.")
        path = Path(path)
        html_content = _build_results_report_html(self.result_)
        if _write_pdf_from_html(html_content, path):
            return path
        return None

    def export_results_reports(
        self,
        html_path: Union[str, Path] = "results_report.html",
        pdf_path: Union[str, Path] = "results_report.pdf",
        export_pdf: bool = False,
    ) -> Dict[str, Optional[Path]]:
        """Export HTML report and optionally PDF after a run."""
        html_out = self.export_html_report(html_path)
        self.console.print(f"[dim]HTML report saved to {html_out}[/]")
        pdf_out: Optional[Path] = None
        if export_pdf:
            pdf_out = self.export_pdf_report(pdf_path)
            if pdf_out is not None:
                self.console.print(f"[dim]PDF report saved to {pdf_out}[/]")
            else:
                self.console.print(
                    "[yellow]PDF export skipped — install xhtml2pdf or weasyprint "
                    "(pip install corter-ml[reports])[/]"
                )
        return {"html": html_out, "pdf": pdf_out}

    def export_report(self, path: Union[str, Path]) -> None:
        """Serialize run artifacts to JSON (trials + insights; not the model)."""
        if not self.result_:
            raise RuntimeError("No run results to export. Call run() first.")
        path = Path(path)
        payload = {
            "task": self.result_["task"],
            "best_params": self.result_["best_params"],
            "best_cv_score": self.result_["best_cv_score"],
            "holdout_metrics": self.result_["holdout_metrics"],
            "insights": self.result_["insights"],
            "drift": self.result_["drift"],
            "trials": self.result_["trials"].to_dict(orient="records"),
            "feature_report": self.result_["feature_report"].to_dict(orient="records"),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo_config() -> str:
    return """\
# Corter example configuration
task: auto
target_column: target

model:
  name: random_forest
  params:
    n_estimators: 100

hpo:
  strategy: random
  n_trials: 16
  cv_folds: 3
  search_space:
    n_estimators:
      low: 80
      high: 200
      type: int
    max_depth: [4, 8, 12, null]
    min_samples_leaf:
      low: 1
      high: 6
      type: int

xai:
  top_k_features: 8
  permutation_repeats: 5
  drift_threshold: 0.12

tui:
  refresh_hz: 4
  show_live: true
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Corter — Autonomous ML Optimization Framework")
    parser.add_argument("data", nargs="?", help="CSV dataset path")
    parser.add_argument("-c", "--config", type=Path, help="YAML config path")
    parser.add_argument("--target", help="Target column (overrides config)")
    parser.add_argument("--no-tui", action="store_true", help="Disable live dashboard")
    parser.add_argument("--export", type=Path, help="Write JSON report to path")
    parser.add_argument("--write-example-config", type=Path, metavar="PATH", help="Emit example YAML and exit")
    parser.add_argument("--trials", type=int, help="Override HPO trial count")
    args = parser.parse_args(argv)

    if args.write_example_config:
        args.write_example_config.write_text(_demo_config(), encoding="utf-8")
        Console().print(f"[green]Wrote example config →[/] {args.write_example_config}")
        return 0

    if not args.data:
        parser.error("data CSV path is required unless --write-example-config is used")

    if args.config and args.config.exists():
        config = load_config(args.config)
    else:
        config = CorterConfig.from_mapping(yaml.safe_load(_demo_config()))

    if args.target:
        config.target_column = args.target
    if args.no_tui:
        config.tui.show_live = False
    if args.trials:
        config.hpo.n_trials = args.trials

    core = Corter(config)
    result = core.run(args.data)

    console = Console()
    console.print("\n[bold magenta]Corter run complete[/]\n")
    console.print(f"Best CV score: [green]{result['best_cv_score']:.4f}[/]")
    console.print(f"Hold-out: {result['holdout_metrics']}\n")
    for line in result["insights"]:
        console.print(f" • {line}")

    if args.export:
        core.export_report(args.export)
        console.print(f"\n[dim]Report saved to {args.export}[/]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Corter Web UI - Flask Dashboard
Provides browser-based interface for HPO progress, XAI insights, and metrics
"""

from __future__ import annotations

import json
import math
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import yaml
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

STATIC_DIR = Path(__file__).resolve().parent / "static"
WORKSPACE_ROOT = Path(os.environ.get("CORTER_WORKSPACE", Path.cwd())).resolve()
EXPERIMENTS_DIR = WORKSPACE_ROOT / os.environ.get("CORTER_EXPERIMENTS_DIR", "experiments")

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)

# Global state for dashboard data
dashboard_state = {
    "hpo_progress": {
        "current_trial": 0,
        "total_trials": 0,
        "best_score": 0.0,
        "current_score": 0.0,
        "trials_history": [],
        "score_history": [],
        "current_params": {},
        "best_params": {},
        "strategy": "random",
        "elapsed_time": 0,
        "status": "idle",
    },
    "xai_insights": {
        "top_features": [],
        "feature_importance": {},
        "drift_detected": [],
        "insights": [],
        "shap_values": {},
    },
    "system_metrics": {
        "cpu_percent": 0.0,
        "memory_mb": 0.0,
        "trials_per_second": 0.0,
    },
    "logs": [],
}

# Demo mode state
demo_mode = {
    "enabled": True,
    "start_time": time.time(),
    "trial_count": 0,
    "max_trials": 50,
    "base_score": 0.5,
    "feature_names": [
        "feature_A", "feature_B", "feature_C", "feature_D", "feature_E",
        "feature_F", "feature_G", "feature_H", "feature_I", "feature_J",
    ],
}

# Manually registered experiments (merged with discovered on list/compare)
_registered_experiments: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Experiment discovery & comparison
# ---------------------------------------------------------------------------


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _safe_read_yaml(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, yaml.YAMLError):
        return None


def _format_params(params: Mapping[str, Any]) -> str:
    if not params:
        return "—"
    parts = []
    for key, val in params.items():
        if val is None:
            parts.append(f"{key}=null")
        elif isinstance(val, float):
            parts.append(f"{key}={val:.4g}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts)


def _format_top_features(features: Sequence[Mapping[str, Any]], limit: int = 5) -> str:
    if not features:
        return "—"
    chunks = []
    for feat in features[:limit]:
        name = feat.get("name", "?")
        imp = feat.get("importance")
        if imp is not None:
            chunks.append(f"{name} ({float(imp):.4f})")
        else:
            chunks.append(str(name))
    return ", ".join(chunks)


def _config_summary(config: Optional[Mapping[str, Any]]) -> str:
    if not config:
        return "—"
    hpo = config.get("hpo") or {}
    model = config.get("model") or {}
    task = config.get("task", "auto")
    strategy = hpo.get("strategy", "?")
    model_name = model.get("name", "?")
    n_trials = hpo.get("n_trials", "?")
    return f"{task} · {strategy} · {model_name} · {n_trials} trials"


def _top_features_from_snapshots(path: Path) -> List[Dict[str, Any]]:
    data = _safe_read_json(path)
    if not data:
        return []
    snapshots = data.get("snapshots") or []
    if not snapshots:
        return []
    best = max(snapshots, key=lambda s: float(s.get("score", float("-inf"))))
    return list(best.get("top_features") or [])


def _top_features_from_insights(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Fallback: no structured features in results.json."""
    return []


def _iso_mtime(path: Path) -> str:
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except OSError:
        return datetime.now(timezone.utc).isoformat()


def _load_experiment_bundle(
    exp_id: str,
    directory: Path,
    results_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    snapshots_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    directory = directory.resolve()
    results_path = results_path or directory / "results.json"
    if not results_path.is_file():
        for candidate in sorted(directory.glob("results*.json")):
            results_path = candidate
            break
    if not results_path.is_file():
        return None

    results = _safe_read_json(results_path)
    if not results:
        return None

    config_path = config_path or directory / "config.yaml"
    if not config_path.is_file():
        alt = directory / "custom_config.yaml"
        config_path = alt if alt.is_file() else config_path
    config = _safe_read_yaml(config_path) if config_path.is_file() else None

    snapshots_path = snapshots_path or directory / "explanation_snapshots.json"
    top_features: List[Dict[str, Any]] = []
    if snapshots_path.is_file():
        top_features = _top_features_from_snapshots(snapshots_path)
    if not top_features:
        top_features = _top_features_from_insights(results)

    best_score = results.get("best_cv_score")
    if best_score is None:
        best_score = results.get("best_score")
    try:
        best_score_f = float(best_score) if best_score is not None else None
    except (TypeError, ValueError):
        best_score_f = None

    return {
        "id": exp_id,
        "name": exp_id,
        "source": str(directory),
        "timestamp": _iso_mtime(results_path),
        "config_summary": _config_summary(config),
        "config": config or {},
        "best_score": best_score_f,
        "best_params": dict(results.get("best_params") or {}),
        "top_features": top_features,
        "insights": list(results.get("insights") or []),
        "paths": {
            "results": str(results_path),
            "config": str(config_path) if config_path.is_file() else None,
            "snapshots": str(snapshots_path) if snapshots_path.is_file() else None,
        },
    }


def discover_experiments() -> List[Dict[str, Any]]:
    """Discover experiment runs from experiments/ subdirs and loose results*.json files."""
    found: Dict[str, Dict[str, Any]] = dict(_registered_experiments)
    seen_ids: set[str] = set(found.keys())

    if EXPERIMENTS_DIR.is_dir():
        for sub in sorted(EXPERIMENTS_DIR.iterdir()):
            if not sub.is_dir():
                continue
            exp_id = sub.name
            if exp_id in seen_ids:
                continue
            bundle = _load_experiment_bundle(exp_id, sub)
            if bundle:
                found[exp_id] = bundle
                seen_ids.add(exp_id)

    for results_file in sorted(WORKSPACE_ROOT.glob("results*.json")):
        exp_id = results_file.stem
        if exp_id in seen_ids:
            continue
        bundle = _load_experiment_bundle(exp_id, WORKSPACE_ROOT, results_path=results_file)
        if bundle:
            found[exp_id] = bundle
            seen_ids.add(exp_id)

    # Current workspace config + checkpoint (in-progress or last partial run)
    checkpoint = WORKSPACE_ROOT / "corter_checkpoint.json"
    if checkpoint.is_file():
        ckpt = _safe_read_json(checkpoint)
        if ckpt and not ckpt.get("completed") and ckpt.get("trials"):
            exp_id = "checkpoint_active"
            if exp_id not in seen_ids:
                trials = ckpt.get("trials") or []
                best = max(trials, key=lambda t: float(t.get("score", float("-inf"))))
                meta = {"trial", "score", "elapsed_s"}
                best_params_ckpt = dict(ckpt.get("best_params") or {})
                if not best_params_ckpt:
                    best_params_ckpt = {
                        k: v for k, v in best.items() if k not in meta
                    }
                config = _safe_read_yaml(WORKSPACE_ROOT / "config.yaml")
                snaps = WORKSPACE_ROOT / "explanation_snapshots.json"
                top_features = _top_features_from_snapshots(snaps) if snaps.is_file() else []
                found[exp_id] = {
                    "id": exp_id,
                    "name": "Active checkpoint",
                    "source": str(WORKSPACE_ROOT),
                    "timestamp": _iso_mtime(checkpoint),
                    "config_summary": _config_summary(config),
                    "config": config or {},
                    "best_score": float(ckpt.get("best_score", best.get("score", 0))),
                    "best_params": best_params_ckpt,
                    "top_features": top_features,
                    "insights": [],
                    "paths": {"checkpoint": str(checkpoint)},
                    "in_progress": True,
                }

    experiments = list(found.values())
    experiments.sort(
        key=lambda e: (e.get("timestamp") or "", e.get("id") or ""),
        reverse=True,
    )
    return experiments


def compare_experiments(ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Build side-by-side comparison payload for selected experiment ids."""
    all_experiments = discover_experiments()
    if ids:
        id_set = set(ids)
        selected = [e for e in all_experiments if e.get("id") in id_set]
    else:
        selected = all_experiments

    rows = []
    for exp in selected:
        rows.append({
            "id": exp.get("id"),
            "name": exp.get("name"),
            "config_summary": exp.get("config_summary", "—"),
            "best_score": exp.get("best_score"),
            "best_score_display": (
                f"{float(exp['best_score']):.4f}" if exp.get("best_score") is not None else "—"
            ),
            "best_params": exp.get("best_params", {}),
            "best_params_display": _format_params(exp.get("best_params") or {}),
            "top_features": exp.get("top_features", []),
            "top_features_display": _format_top_features(exp.get("top_features") or []),
            "timestamp": exp.get("timestamp"),
            "in_progress": bool(exp.get("in_progress")),
            "source": exp.get("source"),
        })

    best_row = None
    scored = [r for r in rows if r.get("best_score") is not None]
    if scored:
        best_row = max(scored, key=lambda r: float(r["best_score"]))

    return {
        "count": len(rows),
        "experiments": rows,
        "best_experiment_id": best_row.get("id") if best_row else None,
    }


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------


def generate_demo_data():
    """Generate realistic demo data for the dashboard"""
    global demo_mode, dashboard_state

    if not demo_mode["enabled"]:
        return

    elapsed = time.time() - demo_mode["start_time"]
    progress = min(elapsed / 100.0, 1.0)

    demo_mode["trial_count"] = int(progress * demo_mode["max_trials"])

    if demo_mode["trial_count"] >= demo_mode["max_trials"]:
        time.sleep(5)
        demo_mode["start_time"] = time.time()
        demo_mode["trial_count"] = 0
        dashboard_state["hpo_progress"]["score_history"] = []
        dashboard_state["logs"] = []
        return

    base_improvement = progress * 0.45
    noise = random.uniform(-0.02, 0.02)
    current_score = demo_mode["base_score"] + base_improvement + noise
    current_score = max(0.5, min(0.95, current_score))

    score_history = dashboard_state["hpo_progress"]["score_history"]
    score_history.append(current_score)
    best_score = max(score_history) if score_history else current_score

    current_params = {
        "n_estimators": random.choice([50, 100, 150, 200, 250]),
        "max_depth": random.choice([5, 10, 15, 20, None]),
        "min_samples_split": random.choice([2, 5, 10, 20]),
        "min_samples_leaf": random.choice([1, 2, 4, 8]),
        "learning_rate": round(random.uniform(0.01, 0.3), 4),
    }

    best_params = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "learning_rate": 0.1234,
    }

    dashboard_state["hpo_progress"].update({
        "current_trial": demo_mode["trial_count"],
        "total_trials": demo_mode["max_trials"],
        "best_score": best_score,
        "current_score": current_score,
        "score_history": score_history,
        "current_params": current_params,
        "best_params": best_params,
        "strategy": "random",
        "elapsed_time": elapsed,
        "status": "running" if demo_mode["trial_count"] < demo_mode["max_trials"] else "complete",
    })

    top_features = []
    for i, name in enumerate(demo_mode["feature_names"][:10]):
        base_importance = 0.25 * math.exp(-i * 0.3)
        importance = base_importance + random.uniform(-0.02, 0.02)
        importance = max(0.01, min(0.3, importance))
        top_features.append({"name": name, "importance": importance})

    top_features.sort(key=lambda x: x["importance"], reverse=True)

    insights = [
        f"Top 3 features explain {sum(f['importance'] for f in top_features[:3]) * 100:.1f}% of model decisions",
        f"Current trial #{demo_mode['trial_count']} achieved score of {current_score:.4f}",
        f"Best score improved to {best_score:.4f} (↑{(best_score - demo_mode['base_score']) * 100:.1f}%)",
        "No significant drift detected in feature distributions",
        f"Optimization is {progress * 100:.1f}% complete",
        f"Feature '{top_features[0]['name']}' shows highest importance ({top_features[0]['importance']:.4f})",
    ]

    dashboard_state["xai_insights"].update({
        "top_features": top_features,
        "insights": insights,
    })

    dashboard_state["system_metrics"].update({
        "cpu_percent": 45.0 + random.uniform(-15, 15),
        "memory_mb": 1024 + random.uniform(-200, 200),
        "trials_per_second": 0.5 + random.uniform(-0.1, 0.1),
    })

    log_messages = [
        f"Trial #{demo_mode['trial_count']}: Testing parameters...",
        f"Cross-validation score: {current_score:.4f}",
        f"Best score updated: {best_score:.4f}" if current_score == best_score else None,
        "Computing feature importance..." if demo_mode["trial_count"] % 5 == 0 else None,
        "Analyzing model performance..." if demo_mode["trial_count"] % 7 == 0 else None,
    ]

    for msg in log_messages:
        if msg:
            dashboard_state["logs"].append({
                "timestamp": time.time(),
                "message": msg,
            })

    dashboard_state["logs"] = dashboard_state["logs"][-100:]


def demo_mode_loop():
    """Background thread to update demo data"""
    while True:
        if demo_mode["enabled"]:
            generate_demo_data()
        time.sleep(2)


def update_dashboard_state(data: Dict[str, Any]):
    """Update the global dashboard state with new data"""
    global dashboard_state, demo_mode

    if demo_mode["enabled"]:
        demo_mode["enabled"] = False
        dashboard_state["logs"].append({
            "timestamp": time.time(),
            "message": "🔴 Demo mode disabled - Real optimization started",
        })

    if "hpo_progress" in data:
        dashboard_state["hpo_progress"].update(data["hpo_progress"])

    if "xai_insights" in data:
        dashboard_state["xai_insights"].update(data["xai_insights"])

    if "system_metrics" in data:
        dashboard_state["system_metrics"].update(data["system_metrics"])

    if "log" in data:
        dashboard_state["logs"].append({
            "timestamp": time.time(),
            "message": data["log"],
        })
        dashboard_state["logs"] = dashboard_state["logs"][-100:]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/static/logo.svg")
def logo_svg():
    """Serve brand logo."""
    return send_from_directory(STATIC_DIR, "logo.svg", mimetype="image/svg+xml")


@app.route("/")
def index():
    """Landing page"""
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """Main dashboard page"""
    return render_template("dashboard.html")


@app.route("/compare")
def compare_page():
    """Experiment comparison page"""
    return render_template("compare.html")


@app.route("/api/status")
def get_status():
    """Get current dashboard state"""
    return jsonify(dashboard_state)


@app.route("/api/hpo/progress")
def get_hpo_progress():
    """Get HPO progress data"""
    return jsonify(dashboard_state["hpo_progress"])


@app.route("/api/xai/insights")
def get_xai_insights():
    """Get XAI insights data"""
    return jsonify(dashboard_state["xai_insights"])


@app.route("/api/metrics")
def get_metrics():
    """Get system metrics"""
    return jsonify(dashboard_state["system_metrics"])


@app.route("/api/logs")
def get_logs():
    """Get recent logs"""
    return jsonify(dashboard_state["logs"])


@app.route("/api/experiments")
def list_experiments():
    """List discovered experiment runs."""
    return jsonify({
        "workspace": str(WORKSPACE_ROOT),
        "experiments_dir": str(EXPERIMENTS_DIR),
        "experiments": discover_experiments(),
    })


@app.route("/api/experiments/compare")
def api_compare_experiments():
    """Compare experiments side-by-side. Optional query: ?ids=run1,run2"""
    ids_param = request.args.get("ids", "").strip()
    ids = [x.strip() for x in ids_param.split(",") if x.strip()] if ids_param else None
    return jsonify(compare_experiments(ids))


@app.route("/api/experiments/register", methods=["POST"])
def register_experiment():
    """Register an experiment from explicit file paths."""
    data = request.get_json(silent=True) or {}
    exp_id = str(data.get("id") or data.get("name") or "").strip()
    if not exp_id:
        return jsonify({"error": "id is required"}), 400

    results_path = data.get("results_path")
    if not results_path:
        return jsonify({"error": "results_path is required"}), 400

    results_p = Path(results_path).resolve()
    config_p = Path(data["config_path"]).resolve() if data.get("config_path") else results_p.parent
    snapshots_p = (
        Path(data["snapshots_path"]).resolve()
        if data.get("snapshots_path")
        else results_p.parent / "explanation_snapshots.json"
    )

    bundle = _load_experiment_bundle(
        exp_id,
        results_p.parent,
        results_path=results_p,
        config_path=config_p if config_p.is_file() else results_p.parent / "config.yaml",
        snapshots_path=snapshots_p if snapshots_p.is_file() else None,
    )
    if not bundle:
        return jsonify({"error": f"Could not load experiment from {results_p}"}), 400

    if data.get("name"):
        bundle["name"] = str(data["name"])
    _registered_experiments[exp_id] = bundle
    return jsonify({"status": "ok", "experiment": bundle})


@app.route("/api/update", methods=["POST"])
def update_state():
    """Update dashboard state (called by Corter)"""
    data = request.get_json()
    update_dashboard_state(data)
    return jsonify({"status": "success"})


@app.route("/api/reset", methods=["POST"])
def reset_state():
    """Reset dashboard state"""
    global dashboard_state, demo_mode
    dashboard_state["hpo_progress"]["status"] = "idle"
    dashboard_state["hpo_progress"]["current_trial"] = 0
    dashboard_state["hpo_progress"]["score_history"] = []
    dashboard_state["logs"] = []

    demo_mode["enabled"] = True
    demo_mode["start_time"] = time.time()
    demo_mode["trial_count"] = 0

    return jsonify({"status": "success"})


@app.route("/api/demo/toggle", methods=["POST"])
def toggle_demo():
    """Toggle demo mode on/off"""
    global demo_mode
    demo_mode["enabled"] = not demo_mode["enabled"]

    if demo_mode["enabled"]:
        demo_mode["start_time"] = time.time()
        demo_mode["trial_count"] = 0
        dashboard_state["hpo_progress"]["score_history"] = []
        dashboard_state["logs"] = []
        dashboard_state["logs"].append({
            "timestamp": time.time(),
            "message": "🟢 Demo mode enabled - Showing simulated optimization",
        })

    return jsonify({"status": "success", "demo_enabled": demo_mode["enabled"]})


demo_thread = threading.Thread(target=demo_mode_loop, daemon=True)
demo_thread.start()


def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run the Flask server"""
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    print("🚀 Starting Corter Web UI...")
    print("📊 Dashboard available at: http://127.0.0.1:5000")
    print("📋 Compare experiments at: http://127.0.0.1:5000/compare")
    print("🎬 Demo mode: ENABLED (showing simulated optimization)")
    print("💡 Demo mode will auto-disable when real optimization starts")
    run_server(debug=True)

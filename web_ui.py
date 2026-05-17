"""
Corter Web UI - Flask Dashboard
Provides browser-based interface for HPO progress, XAI insights, and metrics
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import threading
import time
import random
import math
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

app = Flask(__name__)
CORS(app)

# Global state for dashboard data
dashboard_state = {
    'hpo_progress': {
        'current_trial': 0,
        'total_trials': 0,
        'best_score': 0.0,
        'current_score': 0.0,
        'trials_history': [],
        'score_history': [],
        'current_params': {},
        'best_params': {},
        'strategy': 'random',
        'elapsed_time': 0,
        'status': 'idle'
    },
    'xai_insights': {
        'top_features': [],
        'feature_importance': {},
        'drift_detected': [],
        'insights': [],
        'shap_values': {}
    },
    'system_metrics': {
        'cpu_percent': 0.0,
        'memory_mb': 0.0,
        'trials_per_second': 0.0
    },
    'logs': []
}

# Demo mode state
demo_mode = {
    'enabled': True,
    'start_time': time.time(),
    'trial_count': 0,
    'max_trials': 50,
    'base_score': 0.5,
    'feature_names': ['feature_A', 'feature_B', 'feature_C', 'feature_D', 'feature_E', 
                     'feature_F', 'feature_G', 'feature_H', 'feature_I', 'feature_J']
}

def generate_demo_data():
    """Generate realistic demo data for the dashboard"""
    global demo_mode, dashboard_state
    
    if not demo_mode['enabled']:
        return
    
    # Calculate progress
    elapsed = time.time() - demo_mode['start_time']
    progress = min(elapsed / 100.0, 1.0)  # 100 seconds to complete
    
    # Update trial count
    demo_mode['trial_count'] = int(progress * demo_mode['max_trials'])
    
    if demo_mode['trial_count'] >= demo_mode['max_trials']:
        # Reset demo after completion
        time.sleep(5)
        demo_mode['start_time'] = time.time()
        demo_mode['trial_count'] = 0
        dashboard_state['hpo_progress']['score_history'] = []
        dashboard_state['logs'] = []
        return
    
    # Generate improving score with some noise
    base_improvement = progress * 0.45  # Improve from 0.5 to 0.95
    noise = random.uniform(-0.02, 0.02)
    current_score = demo_mode['base_score'] + base_improvement + noise
    current_score = max(0.5, min(0.95, current_score))
    
    # Best score is always the max seen
    score_history = dashboard_state['hpo_progress']['score_history']
    score_history.append(current_score)
    best_score = max(score_history) if score_history else current_score
    
    # Generate current parameters
    current_params = {
        'n_estimators': random.choice([50, 100, 150, 200, 250]),
        'max_depth': random.choice([5, 10, 15, 20, None]),
        'min_samples_split': random.choice([2, 5, 10, 20]),
        'min_samples_leaf': random.choice([1, 2, 4, 8]),
        'learning_rate': round(random.uniform(0.01, 0.3), 4)
    }
    
    # Best parameters (slightly better values)
    best_params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'learning_rate': 0.1234
    }
    
    # Update HPO progress
    dashboard_state['hpo_progress'].update({
        'current_trial': demo_mode['trial_count'],
        'total_trials': demo_mode['max_trials'],
        'best_score': best_score,
        'current_score': current_score,
        'score_history': score_history,
        'current_params': current_params,
        'best_params': best_params,
        'strategy': 'random',
        'elapsed_time': elapsed,
        'status': 'running' if demo_mode['trial_count'] < demo_mode['max_trials'] else 'complete'
    })
    
    # Generate feature importance (with some variation)
    top_features = []
    for i, name in enumerate(demo_mode['feature_names'][:10]):
        # Create decreasing importance with some randomness
        base_importance = 0.25 * math.exp(-i * 0.3)
        importance = base_importance + random.uniform(-0.02, 0.02)
        importance = max(0.01, min(0.3, importance))
        top_features.append({
            'name': name,
            'importance': importance
        })
    
    # Sort by importance
    top_features.sort(key=lambda x: x['importance'], reverse=True)
    
    # Generate insights
    insights = [
        f"Top 3 features explain {sum(f['importance'] for f in top_features[:3]) * 100:.1f}% of model decisions",
        f"Current trial #{demo_mode['trial_count']} achieved score of {current_score:.4f}",
        f"Best score improved to {best_score:.4f} (↑{(best_score - demo_mode['base_score']) * 100:.1f}%)",
        "No significant drift detected in feature distributions",
        f"Optimization is {progress * 100:.1f}% complete",
        f"Feature '{top_features[0]['name']}' shows highest importance ({top_features[0]['importance']:.4f})"
    ]
    
    # Update XAI insights
    dashboard_state['xai_insights'].update({
        'top_features': top_features,
        'insights': insights
    })
    
    # Generate system metrics with realistic fluctuation
    dashboard_state['system_metrics'].update({
        'cpu_percent': 45.0 + random.uniform(-15, 15),
        'memory_mb': 1024 + random.uniform(-200, 200),
        'trials_per_second': 0.5 + random.uniform(-0.1, 0.1)
    })
    
    # Add log entries
    log_messages = [
        f"Trial #{demo_mode['trial_count']}: Testing parameters...",
        f"Cross-validation score: {current_score:.4f}",
        f"Best score updated: {best_score:.4f}" if current_score == best_score else None,
        "Computing feature importance..." if demo_mode['trial_count'] % 5 == 0 else None,
        "Analyzing model performance..." if demo_mode['trial_count'] % 7 == 0 else None,
    ]
    
    for msg in log_messages:
        if msg:
            dashboard_state['logs'].append({
                'timestamp': time.time(),
                'message': msg
            })
    
    # Keep only last 100 logs
    dashboard_state['logs'] = dashboard_state['logs'][-100:]

def demo_mode_loop():
    """Background thread to update demo data"""
    while True:
        if demo_mode['enabled']:
            generate_demo_data()
        time.sleep(2)  # Update every 2 seconds

def update_dashboard_state(data: Dict[str, Any]):
    """Update the global dashboard state with new data"""
    global dashboard_state, demo_mode
    
    # Disable demo mode when real data comes in
    if demo_mode['enabled']:
        demo_mode['enabled'] = False
        dashboard_state['logs'].append({
            'timestamp': time.time(),
            'message': '🔴 Demo mode disabled - Real optimization started'
        })
    
    if 'hpo_progress' in data:
        dashboard_state['hpo_progress'].update(data['hpo_progress'])
    
    if 'xai_insights' in data:
        dashboard_state['xai_insights'].update(data['xai_insights'])
    
    if 'system_metrics' in data:
        dashboard_state['system_metrics'].update(data['system_metrics'])
    
    if 'log' in data:
        dashboard_state['logs'].append({
            'timestamp': time.time(),
            'message': data['log']
        })
        # Keep only last 100 logs
        dashboard_state['logs'] = dashboard_state['logs'][-100:]

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current dashboard state"""
    return jsonify(dashboard_state)

@app.route('/api/hpo/progress')
def get_hpo_progress():
    """Get HPO progress data"""
    return jsonify(dashboard_state['hpo_progress'])

@app.route('/api/xai/insights')
def get_xai_insights():
    """Get XAI insights data"""
    return jsonify(dashboard_state['xai_insights'])

@app.route('/api/metrics')
def get_metrics():
    """Get system metrics"""
    return jsonify(dashboard_state['system_metrics'])

@app.route('/api/logs')
def get_logs():
    """Get recent logs"""
    return jsonify(dashboard_state['logs'])

@app.route('/api/update', methods=['POST'])
def update_state():
    """Update dashboard state (called by Corter)"""
    data = request.get_json()
    update_dashboard_state(data)
    return jsonify({'status': 'success'})

@app.route('/api/reset', methods=['POST'])
def reset_state():
    """Reset dashboard state"""
    global dashboard_state, demo_mode
    dashboard_state['hpo_progress']['status'] = 'idle'
    dashboard_state['hpo_progress']['current_trial'] = 0
    dashboard_state['hpo_progress']['score_history'] = []
    dashboard_state['logs'] = []
    
    # Re-enable demo mode
    demo_mode['enabled'] = True
    demo_mode['start_time'] = time.time()
    demo_mode['trial_count'] = 0
    
    return jsonify({'status': 'success'})

@app.route('/api/demo/toggle', methods=['POST'])
def toggle_demo():
    """Toggle demo mode on/off"""
    global demo_mode
    demo_mode['enabled'] = not demo_mode['enabled']
    
    if demo_mode['enabled']:
        demo_mode['start_time'] = time.time()
        demo_mode['trial_count'] = 0
        dashboard_state['hpo_progress']['score_history'] = []
        dashboard_state['logs'] = []
        dashboard_state['logs'].append({
            'timestamp': time.time(),
            'message': '🟢 Demo mode enabled - Showing simulated optimization'
        })
    
    return jsonify({'status': 'success', 'demo_enabled': demo_mode['enabled']})

# Start demo mode thread when module loads (for Railway/production)
demo_thread = threading.Thread(target=demo_mode_loop, daemon=True)
demo_thread.start()

def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run the Flask server"""
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    print("🚀 Starting Corter Web UI...")
    print("📊 Dashboard available at: http://127.0.0.1:5000")
    print("🎬 Demo mode: ENABLED (showing simulated optimization)")
    print("💡 Demo mode will auto-disable when real optimization starts")
    run_server(debug=True)

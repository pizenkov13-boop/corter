"""
AxiomCore Web UI - Flask Dashboard
Provides browser-based interface for HPO progress, XAI insights, and metrics
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
import threading
import time
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

def update_dashboard_state(data: Dict[str, Any]):
    """Update the global dashboard state with new data"""
    global dashboard_state
    
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
    """Update dashboard state (called by AxiomCore)"""
    data = request.get_json()
    update_dashboard_state(data)
    return jsonify({'status': 'success'})

@app.route('/api/reset', methods=['POST'])
def reset_state():
    """Reset dashboard state"""
    global dashboard_state
    dashboard_state['hpo_progress']['status'] = 'idle'
    dashboard_state['hpo_progress']['current_trial'] = 0
    dashboard_state['logs'] = []
    return jsonify({'status': 'success'})

def run_server(host='127.0.0.1', port=5000, debug=False):
    """Run the Flask server"""
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    print("🚀 Starting AxiomCore Web UI...")
    print("📊 Dashboard available at: http://127.0.0.1:5000")
    run_server(debug=True)

# Made with Bob

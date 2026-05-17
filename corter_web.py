"""
Corter Web Integration
Connects Corter with the Flask web dashboard
"""

import requests
import time
from typing import Dict, Any, Optional
import psutil
import os

class WebDashboardConnector:
    """Connector to send updates to the web dashboard"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000"):
        self.base_url = base_url
        self.enabled = True
        self.start_time = time.time()
        self.trial_count = 0
        self.last_trial_time = time.time()
        
        # Test connection
        try:
            requests.get(f"{self.base_url}/api/status", timeout=1)
        except:
            print("⚠️  Web dashboard not available. Run 'python web_ui.py' to start it.")
            self.enabled = False
    
    def update_hpo_progress(self, 
                           current_trial: int,
                           total_trials: int,
                           best_score: float,
                           current_score: float,
                           score_history: list,
                           current_params: dict,
                           best_params: dict,
                           strategy: str,
                           status: str = "running"):
        """Update HPO progress on dashboard"""
        if not self.enabled:
            return
        
        elapsed = time.time() - self.start_time
        
        data = {
            'hpo_progress': {
                'current_trial': current_trial,
                'total_trials': total_trials,
                'best_score': best_score,
                'current_score': current_score,
                'score_history': score_history,
                'current_params': current_params,
                'best_params': best_params,
                'strategy': strategy,
                'elapsed_time': elapsed,
                'status': status
            }
        }
        
        try:
            requests.post(f"{self.base_url}/api/update", json=data, timeout=1)
        except:
            pass
    
    def update_xai_insights(self,
                           top_features: list,
                           insights: list,
                           drift_detected: list = None):
        """Update XAI insights on dashboard"""
        if not self.enabled:
            return
        
        data = {
            'xai_insights': {
                'top_features': top_features,
                'insights': insights,
                'drift_detected': drift_detected or []
            }
        }
        
        try:
            requests.post(f"{self.base_url}/api/update", json=data, timeout=1)
        except:
            pass
    
    def update_system_metrics(self):
        """Update system metrics on dashboard"""
        if not self.enabled:
            return
        
        # Calculate trials per second
        current_time = time.time()
        time_diff = current_time - self.last_trial_time
        trials_per_sec = 1.0 / time_diff if time_diff > 0 else 0.0
        self.last_trial_time = current_time
        
        data = {
            'system_metrics': {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024,
                'trials_per_second': trials_per_sec
            }
        }
        
        try:
            requests.post(f"{self.base_url}/api/update", json=data, timeout=1)
        except:
            pass
    
    def log(self, message: str):
        """Send log message to dashboard"""
        if not self.enabled:
            return
        
        data = {'log': message}
        
        try:
            requests.post(f"{self.base_url}/api/update", json=data, timeout=1)
        except:
            pass
    
    def set_status(self, status: str):
        """Set overall status (idle, running, complete, error)"""
        if not self.enabled:
            return
        
        data = {
            'hpo_progress': {
                'status': status
            }
        }
        
        try:
            requests.post(f"{self.base_url}/api/update", json=data, timeout=1)
        except:
            pass


# Example usage function
def run_with_web_dashboard(config_path: str, data_path: str):
    """
    Run Corter with web dashboard integration
    
    Example:
        python corter_web.py
    """
    from corter import Corter
    
    # Initialize web connector
    web = WebDashboardConnector()
    web.log("🚀 Starting Corter optimization...")
    web.set_status("running")
    
    try:
        # Load Corter
        core = Corter.from_yaml(config_path)
        web.log(f"✅ Loaded configuration from {config_path}")
        
        # Monkey-patch to send updates during optimization
        original_fit = core.hpo.fit
        
        def fit_with_updates(X, y):
            # Store original callback
            score_history = []
            
            def trial_callback(trial_num, params, score):
                score_history.append(score)
                best_score = max(score_history) if score_history else 0.0
                
                web.update_hpo_progress(
                    current_trial=trial_num,
                    total_trials=core.config.hpo.n_trials,
                    best_score=best_score,
                    current_score=score,
                    score_history=score_history,
                    current_params=params,
                    best_params=core.hpo.best_params_ if hasattr(core.hpo, 'best_params_') else {},
                    strategy=core.config.hpo.strategy,
                    status="running"
                )
                
                web.update_system_metrics()
                web.log(f"Trial {trial_num}/{core.config.hpo.n_trials}: Score = {score:.4f}")
            
            # Call original fit
            result = original_fit(X, y)
            
            return result
        
        core.hpo.fit = fit_with_updates
        
        # Run optimization
        web.log(f"📊 Loading data from {data_path}...")
        result = core.run(data_path)
        
        # Update with final results
        web.log("✅ Optimization complete!")
        web.set_status("complete")
        
        # Send XAI insights
        if 'insights' in result:
            web.log("🔍 Analyzing feature importance...")
            
            # Format top features
            if 'feature_report' in result:
                top_features = []
                for _, row in result['feature_report'].head(10).iterrows():
                    top_features.append({
                        'name': row['feature'],
                        'importance': float(row['perm_mean'])
                    })
                
                web.update_xai_insights(
                    top_features=top_features,
                    insights=result['insights']
                )
        
        web.log(f"🏆 Best CV Score: {result['best_cv_score']:.4f}")
        web.log("Dashboard will remain active. Press Ctrl+C to exit.")
        
        # Keep running to show final results
        print("\n" + "="*60)
        print("✅ Optimization complete! View results at: http://127.0.0.1:5000")
        print("="*60)
        
        return result
        
    except Exception as e:
        web.log(f"❌ Error: {str(e)}")
        web.set_status("error")
        raise


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "data.csv"
    
    print("="*60)
    print("⚡ Corter with Web Dashboard")
    print("="*60)
    print(f"📊 Dashboard: http://127.0.0.1:5000")
    print(f"⚙️  Config: {config_path}")
    print(f"📁 Data: {data_path}")
    print("="*60)
    print()
    
    result = run_with_web_dashboard(config_path, data_path)
    
    # Keep script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

# Made with Bob

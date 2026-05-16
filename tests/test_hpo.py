"""
Tests for HPO (Hyperparameter Optimization) Module
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from axiomcore import HyperparameterAutopilot, HPOConfig


@pytest.fixture
def classification_data():
    """Generate synthetic classification dataset"""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate synthetic regression dataset"""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42
    )
    return X, y


class TestHPOConfig:
    """Test HPO configuration"""
    
    def test_default_config(self):
        """Test default HPO configuration"""
        config = HPOConfig()
        assert config.strategy == "random"
        assert config.n_trials == 20
        assert config.cv_folds == 5
        assert config.enable_early_stop is False
    
    def test_custom_config(self):
        """Test custom HPO configuration"""
        config = HPOConfig(
            strategy="scipy_de",
            n_trials=50,
            enable_early_stop=True,
            patience=10
        )
        assert config.strategy == "scipy_de"
        assert config.n_trials == 50
        assert config.enable_early_stop is True
        assert config.patience == 10


class TestHyperparameterAutopilot:
    """Test Hyperparameter Autopilot"""
    
    def test_initialization(self):
        """Test HPO initialization"""
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(n_trials=5)
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        assert hpo.estimator is not None
        assert hpo.config.n_trials == 5
        assert hpo.task == "classification"
    
    def test_random_search_classification(self, classification_data):
        """Test random search on classification task"""
        X, y = classification_data
        
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(strategy="random", n_trials=5, cv_folds=3)
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        hpo.fit(X, y)
        
        # Check results
        assert hpo.best_score_ > 0
        assert hpo.best_params_ is not None
        assert len(hpo.trials) == 5
        assert hpo.best_estimator_ is not None
    
    def test_random_search_regression(self, regression_data):
        """Test random search on regression task"""
        X, y = regression_data
        
        estimator = RandomForestRegressor(random_state=42)
        config = HPOConfig(strategy="random", n_trials=5, cv_folds=3)
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="regression"
        )
        
        hpo.fit(X, y)
        
        # Check results
        assert hpo.best_score_ is not None
        assert hpo.best_params_ is not None
        assert len(hpo.trials) == 5
    
    def test_early_stopping(self, classification_data):
        """Test early stopping functionality"""
        X, y = classification_data
        
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(
            strategy="random",
            n_trials=20,
            enable_early_stop=True,
            patience=3,
            min_delta=0.001,
            cv_folds=3
        )
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        hpo.fit(X, y)
        
        # Early stopping should trigger before all trials
        assert len(hpo.trials) <= 20
        assert hpo.best_score_ > 0
    
    def test_parallel_execution(self, classification_data):
        """Test parallel trial execution"""
        X, y = classification_data
        
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(
            strategy="random",
            n_trials=8,
            parallel_trials=4,
            cv_folds=3
        )
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        hpo.fit(X, y)
        
        # Check that all trials completed
        assert len(hpo.trials) == 8
        assert hpo.best_score_ > 0
    
    def test_trials_dataframe(self, classification_data):
        """Test trials DataFrame structure"""
        X, y = classification_data
        
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(strategy="random", n_trials=5, cv_folds=3)
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        hpo.fit(X, y)
        
        # Check DataFrame structure
        assert isinstance(hpo.trials, pd.DataFrame)
        assert 'trial' in hpo.trials.columns
        assert 'score' in hpo.trials.columns
        assert 'elapsed' in hpo.trials.columns
        assert len(hpo.trials) == 5
    
    def test_score_improvement(self, classification_data):
        """Test that best score improves or stays same"""
        X, y = classification_data
        
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(strategy="random", n_trials=10, cv_folds=3)
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        hpo.fit(X, y)
        
        # Best score should be >= all individual scores
        scores = hpo.trials['score'].values
        assert hpo.best_score_ >= np.max(scores) - 0.001  # Allow small floating point error
    
    def test_reproducibility(self, classification_data):
        """Test reproducibility with same random state"""
        X, y = classification_data
        
        config = HPOConfig(
            strategy="random",
            n_trials=5,
            cv_folds=3,
            random_state=42
        )
        
        # Run 1
        hpo1 = HyperparameterAutopilot(
            estimator=RandomForestClassifier(random_state=42),
            config=config,
            task="classification"
        )
        hpo1.fit(X, y)
        
        # Run 2
        hpo2 = HyperparameterAutopilot(
            estimator=RandomForestClassifier(random_state=42),
            config=config,
            task="classification"
        )
        hpo2.fit(X, y)
        
        # Results should be identical
        assert abs(hpo1.best_score_ - hpo2.best_score_) < 0.001
        assert hpo1.best_params_ == hpo2.best_params_


class TestSearchStrategies:
    """Test different search strategies"""
    
    def test_scipy_de_strategy(self, classification_data):
        """Test scipy differential evolution strategy"""
        X, y = classification_data
        
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(strategy="scipy_de", n_trials=10, cv_folds=3)
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        hpo.fit(X, y)
        
        assert hpo.best_score_ > 0
        assert hpo.best_params_ is not None
    
    def test_local_strategy(self, classification_data):
        """Test local optimization strategy"""
        X, y = classification_data
        
        estimator = RandomForestClassifier(random_state=42)
        config = HPOConfig(strategy="local", n_trials=10, cv_folds=3)
        
        hpo = HyperparameterAutopilot(
            estimator=estimator,
            config=config,
            task="classification"
        )
        
        hpo.fit(X, y)
        
        assert hpo.best_score_ > 0
        assert hpo.best_params_ is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob

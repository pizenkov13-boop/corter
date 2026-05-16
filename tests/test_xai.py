"""
Tests for XAI (Explainable AI) Module
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from axiomcore import SemanticDiagnostics, XAIConfig


@pytest.fixture
def classification_data():
    """Generate synthetic classification dataset with feature names"""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Create DataFrame with feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    return X_df, y, feature_names


@pytest.fixture
def trained_model(classification_data):
    """Train a simple model for testing"""
    X, y, _ = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


class TestXAIConfig:
    """Test XAI configuration"""
    
    def test_default_config(self):
        """Test default XAI configuration"""
        config = XAIConfig()
        assert config.top_k_features == 10
        assert config.permutation_repeats == 5
        assert config.use_shap is False
    
    def test_custom_config(self):
        """Test custom XAI configuration"""
        config = XAIConfig(
            top_k_features=15,
            permutation_repeats=10,
            use_shap=True,
            shap_sample_size=200
        )
        assert config.top_k_features == 15
        assert config.permutation_repeats == 10
        assert config.use_shap is True
        assert config.shap_sample_size == 200


class TestSemanticDiagnostics:
    """Test Semantic Diagnostics (XAI)"""
    
    def test_initialization(self):
        """Test XAI initialization"""
        config = XAIConfig()
        xai = SemanticDiagnostics(config=config, task="classification")
        
        assert xai.config is not None
        assert xai.task == "classification"
    
    def test_analyze_basic(self, classification_data, trained_model):
        """Test basic XAI analysis"""
        X, y, feature_names = classification_data
        
        config = XAIConfig(top_k_features=5, permutation_repeats=3)
        xai = SemanticDiagnostics(config=config, task="classification")
        
        result = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y,
            feature_names=feature_names
        )
        
        # Check result structure
        assert 'feature_report' in result
        assert 'insights' in result
        
        # Check feature report
        report = result['feature_report']
        assert isinstance(report, pd.DataFrame)
        assert len(report) <= 5  # top_k_features
        assert 'feature' in report.columns
        assert 'perm_mean' in report.columns
        assert 'perm_std' in report.columns
        
        # Check insights
        assert isinstance(result['insights'], list)
        assert len(result['insights']) > 0
    
    def test_permutation_importance(self, classification_data, trained_model):
        """Test permutation importance calculation"""
        X, y, feature_names = classification_data
        
        config = XAIConfig(permutation_repeats=5)
        xai = SemanticDiagnostics(config=config, task="classification")
        
        result = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y,
            feature_names=feature_names
        )
        
        report = result['feature_report']
        
        # Check that importance values are reasonable
        assert all(report['perm_mean'] >= 0)
        assert all(report['perm_std'] >= 0)
        
        # Check that features are sorted by importance
        importances = report['perm_mean'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))
    
    def test_mutual_information(self, classification_data, trained_model):
        """Test mutual information calculation"""
        X, y, feature_names = classification_data
        
        config = XAIConfig()
        xai = SemanticDiagnostics(config=config, task="classification")
        
        result = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y,
            feature_names=feature_names
        )
        
        report = result['feature_report']
        
        # Check MI column exists and has valid values
        assert 'mi_score' in report.columns
        assert all(report['mi_score'] >= 0)
    
    def test_shap_integration(self, classification_data, trained_model):
        """Test SHAP integration (if available)"""
        X, y, feature_names = classification_data
        
        config = XAIConfig(use_shap=True, shap_sample_size=50)
        xai = SemanticDiagnostics(config=config, task="classification")
        
        try:
            result = xai.analyze(
                estimator=trained_model,
                X=X.values,
                y=y,
                feature_names=feature_names
            )
            
            report = result['feature_report']
            
            # If SHAP is available, check for SHAP column
            if 'shap_importance' in report.columns:
                assert all(report['shap_importance'] >= 0)
        except ImportError:
            # SHAP not installed, skip test
            pytest.skip("SHAP not installed")
    
    def test_drift_detection(self, classification_data, trained_model):
        """Test drift detection functionality"""
        X, y, feature_names = classification_data
        
        # Create reference and current data
        X_ref = X.iloc[:100].values
        X_cur = X.iloc[100:].values
        
        config = XAIConfig(drift_threshold=0.05)
        xai = SemanticDiagnostics(config=config, task="classification")
        
        # Analyze with drift detection
        result = xai.analyze(
            estimator=trained_model,
            X=X_cur,
            y=y[100:],
            feature_names=feature_names,
            X_reference=X_ref
        )
        
        # Check that drift analysis was performed
        assert 'insights' in result
        insights = result['insights']
        
        # Should have some insight about drift
        drift_insights = [i for i in insights if 'drift' in i.lower()]
        assert len(drift_insights) > 0
    
    def test_insights_generation(self, classification_data, trained_model):
        """Test insights generation"""
        X, y, feature_names = classification_data
        
        config = XAIConfig(top_k_features=5)
        xai = SemanticDiagnostics(config=config, task="classification")
        
        result = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y,
            feature_names=feature_names
        )
        
        insights = result['insights']
        
        # Check insights structure
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        # Check that insights are strings
        assert all(isinstance(i, str) for i in insights)
        
        # Check for key insight types
        insight_text = ' '.join(insights).lower()
        assert 'feature' in insight_text or 'importance' in insight_text
    
    def test_top_k_features(self, classification_data, trained_model):
        """Test top_k_features parameter"""
        X, y, feature_names = classification_data
        
        for k in [3, 5, 10]:
            config = XAIConfig(top_k_features=k)
            xai = SemanticDiagnostics(config=config, task="classification")
            
            result = xai.analyze(
                estimator=trained_model,
                X=X.values,
                y=y,
                feature_names=feature_names
            )
            
            report = result['feature_report']
            assert len(report) == min(k, len(feature_names))
    
    def test_caching(self, classification_data, trained_model):
        """Test computation caching"""
        X, y, feature_names = classification_data
        
        config = XAIConfig(permutation_repeats=5)
        xai = SemanticDiagnostics(config=config, task="classification")
        
        # First analysis
        result1 = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y,
            feature_names=feature_names
        )
        
        # Second analysis with same data (should use cache)
        result2 = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y,
            feature_names=feature_names
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(
            result1['feature_report'],
            result2['feature_report']
        )
    
    def test_feature_names_handling(self, classification_data, trained_model):
        """Test handling of feature names"""
        X, y, feature_names = classification_data
        
        config = XAIConfig()
        xai = SemanticDiagnostics(config=config, task="classification")
        
        # Test with explicit feature names
        result = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y,
            feature_names=feature_names
        )
        
        report = result['feature_report']
        assert all(name in feature_names for name in report['feature'])
        
        # Test without feature names (should generate default names)
        result_no_names = xai.analyze(
            estimator=trained_model,
            X=X.values,
            y=y
        )
        
        report_no_names = result_no_names['feature_report']
        assert 'feature' in report_no_names.columns
        assert len(report_no_names) > 0


class TestXAIRegression:
    """Test XAI with regression tasks"""
    
    def test_regression_analysis(self):
        """Test XAI analysis on regression task"""
        from sklearn.datasets import make_regression
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate regression data
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Analyze
        config = XAIConfig(top_k_features=5)
        xai = SemanticDiagnostics(config=config, task="regression")
        
        result = xai.analyze(
            estimator=model,
            X=X,
            y=y,
            feature_names=feature_names
        )
        
        # Check results
        assert 'feature_report' in result
        assert 'insights' in result
        assert len(result['feature_report']) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob

# AxiomCore Optimization Improvements

## Executive Summary
This document provides comprehensive recommendations for optimizing the HPO module, XAI diagnostics layer, and TUI dashboard in axiomcore.py. Focus areas include computational efficiency, memory optimization, and enhanced ML capabilities.

---

## 1. HPO Module Optimizations

### 1.1 Early Stopping & Convergence Detection
**Current Issue**: All trials run regardless of convergence (lines 285-291, 315-347)

**Impact**: 20-40% reduction in HPO time for converged searches

**Recommendation**: Add early stopping parameters to [`HPOConfig`](axiomcore.py:58) and implement convergence detection in [`HyperparameterAutopilot`](axiomcore.py:185).

Add to config:
- `patience: int = 5` - Stop after N trials without improvement
- `min_delta: float = 0.001` - Minimum improvement threshold
- `enable_early_stop: bool = True`

Implementation requires tracking `_no_improvement_count` and `_best_score_history` in the autopilot class, then checking convergence after each trial in [`_fit_random()`](axiomcore.py:278).

---

### 1.2 Parallel Trial Evaluation
**Current Issue**: Sequential trial evaluation (line 224: `n_jobs=-1` only parallelizes CV folds)

**Impact**: 2-4x speedup on multi-core systems

**Recommendation**: Implement batch-parallel trial execution using `joblib.Parallel` with `delayed()`. Evaluate 4-8 trials simultaneously, then log results. This requires refactoring [`_fit_random()`](axiomcore.py:278) to use batch processing with `Parallel(n_jobs=-1, backend='loky')`.

---

### 1.3 Warm Start from Previous Runs
**Current Issue**: No ability to resume or leverage previous HPO runs

**Impact**: Enables incremental optimization and experiment continuity

**Recommendation**: Add `warm_start_trials: Optional[pd.DataFrame]` parameter to [`fit()`](axiomcore.py:252). Load previous trials, extract best parameters, and continue optimization from there. This allows users to incrementally refine hyperparameters across multiple sessions.

---

### 1.4 Adaptive Search Space Refinement
**Current Issue**: Static search space throughout optimization (lines 256-260)

**Impact**: Faster convergence to optimal regions

**Recommendation**: After every 10 trials, narrow the search space to ±1.5 standard deviations around the mean of top 5 performing trials. Add `_refine_search_space()` method that analyzes [`self.trials`](axiomcore.py:206) and dynamically adjusts bounds in the search space dictionary.

---

### 1.5 Smarter Scipy DE Configuration
**Current Issue**: Fixed DE parameters may not suit all problems (lines 338-347)

**Impact**: Better performance across different problem dimensions

**Recommendation**: Make [`differential_evolution()`](axiomcore.py:339) strategy and population size adaptive based on dimensionality:
- Low-dim (≤5): `strategy='best1bin'`, `popsize=10`
- Medium-dim (6-15): `strategy='best2bin'`, `popsize=15`
- High-dim (>15): `strategy='randtobest1bin'`, `popsize=20`

Also add convergence tolerances: `atol=0.001`, `tol=0.01`

---

## 2. XAI Diagnostics Layer Improvements

### 2.1 Caching for Expensive Computations
**Current Issue**: Permutation importance recomputed unnecessarily (lines 414-421)

**Impact**: 50-70% faster re-analysis on same dataset

**Recommendation**: Add `_cache: Dict[str, Any]` to [`SemanticDiagnostics`](axiomcore.py:382). Generate cache keys from data hash (MD5 of X and y bytes). Store permutation importance results and reuse when analyzing the same dataset. Use `hashlib.md5()` for key generation.

---

### 2.2 Incremental Feature Importance
**Current Issue**: All features analyzed at once (lines 414-433)

**Impact**: Handles datasets with 100+ features efficiently

**Recommendation**: For datasets with >50 features, compute [`permutation_importance()`](axiomcore.py:414) in batches of 50 features. This reduces memory pressure and allows progress tracking. Aggregate results into single importance array matching original feature order.

---

### 2.3 SHAP Integration for Better Explanations
**Current Issue**: Only permutation importance and MI used (lines 414-433)

**Impact**: More accurate feature importance, especially for tree models

**Recommendation**: Add optional SHAP analysis to [`analyze()`](axiomcore.py:398):
- Add `use_shap: bool = False` and `shap_sample_size: int = 100` to [`XAIConfig`](axiomcore.py:68)
- Use `shap.TreeExplainer` for tree models, `shap.KernelExplainer` for others
- Subsample data for efficiency
- Add `shap_importance` column to feature report

Requires optional `shap` package dependency.

---

### 2.4 Drift Detection Optimization
**Current Issue**: KS test computed for all features (lines 455-465)

**Impact**: 50% faster drift detection on high-dimensional data

**Recommendation**: Prioritize drift detection for high-importance features. Modify [`_population_drift()`](axiomcore.py:455) to accept optional `importance_scores` parameter. Compute drift only for top 50% of features by importance, or at minimum the top 10 features.

---

### 2.5 Enhanced Insight Generation
**Current Issue**: Basic narrative insights (lines 467-514)

**Impact**: More actionable insights for model improvement

**Recommendation**: Extend [`_build_insights()`](axiomcore.py:467) with actionable recommendations:
- Suggest interaction features when top 2 features are co-dominant
- Recommend feature pruning when 3+ features have zero importance
- Warn about high feature concentration (>50% mass in one feature)
- Suggest regularization or feature selection strategies

Add "💡 Recommendations:" section to insights list.

---

## 3. TUI Dashboard Enhancements

### 3.1 Performance Metrics History Plot
**Current Issue**: Only current metrics shown (lines 605-608)

**Impact**: Better visualization of optimization progress

**Recommendation**: Add ASCII sparkline to [`_hpo_table()`](axiomcore.py:573) showing score trend over trials. Track `_score_history: List[float]` in [`AxiomDashboard`](axiomcore.py:533). Use Unicode block characters (▁▂▃▄▅▆▇█) to create 40-character sparkline. Display as table caption.

---

### 3.2 Resource Usage Monitoring
**Current Issue**: No system resource tracking

**Impact**: Better understanding of computational costs

**Recommendation**: Add CPU and memory monitoring using `psutil`:
- Track `_cpu_percent` and `_memory_mb` in dashboard
- Update in [`_footer_panel()`](axiomcore.py:611) every refresh
- Display: "CPU: 45.2% • Memory: 1024 MB"

Requires `psutil` package.

---

### 3.3 Configurable Layout
**Current Issue**: Fixed 3:2 layout ratio (lines 561-564)

**Impact**: More flexible dashboard customization

**Recommendation**: Add layout configuration to [`TUIConfig`](axiomcore.py:76):
- `hpo_ratio: int = 3`
- `xai_ratio: int = 2`
- `show_header: bool = True`
- `show_footer: bool = True`

Modify [`_build_layout()`](axiomcore.py:554) to respect these settings.

---

### 3.4 Export Dashboard State
**Current Issue**: No way to save dashboard state

**Impact**: Enables sharing and documentation of runs

**Recommendation**: Add `export_snapshot(path)` method that uses `Console(record=True)` to render dashboard and export as HTML with `console.export_html(inline_styles=True)`. Allows saving dashboard state for reports.

---

## 4. General Performance Optimizations

### 4.1 Memory-Efficient Data Handling
**Current Issue**: Full dataset copies in multiple places (lines 713-714, 740-746)

**Impact**: 30-50% memory reduction

**Recommendation**: Use NumPy array views instead of copies in [`_load_xy()`](axiomcore.py:698). Return `X[tr_idx]` views rather than creating new arrays. Only copy when necessary for sklearn compatibility.

---

### 4.2 Lazy Loading for Large Datasets
**Current Issue**: Entire dataset loaded into memory (line 704)

**Impact**: Enables handling datasets larger than RAM

**Recommendation**: Add optional `chunk_size` parameter to [`run()`](axiomcore.py:734). Implement `_run_chunked()` method that processes data in chunks using `pd.read_csv(chunksize=...)`. Requires incremental fitting support.

---

## 5. Implementation Priority

### High Priority (Immediate Impact)
1. Early stopping in HPO (1.1) - 20-40% time savings
2. Parallel trial evaluation (1.2) - 2-4x speedup
3. XAI caching (2.1) - 50-70% faster re-analysis
4. Dashboard sparklines (3.1) - Better UX

### Medium Priority (Significant Improvement)
5. Adaptive search space (1.4) - 15-30% faster convergence
6. SHAP integration (2.3) - Better explanations
7. Resource monitoring (3.2) - Operational visibility
8. Warm start capability (1.3) - Workflow improvement

### Low Priority (Nice to Have)
9. Chunked processing (4.2) - Large dataset support
10. Dashboard export (3.4) - Documentation
11. Configurable layout (3.3) - Customization

---

## 6. Estimated Performance Gains

| Optimization | Time Savings | Memory Savings | Complexity |
|--------------|--------------|----------------|------------|
| Early Stopping | 20-40% | - | Low |
| Parallel Trials | 2-4x | - | Medium |
| XAI Caching | 50-70% | +10% | Low |
| Adaptive Search | 15-30% | - | Medium |
| Drift Optimization | 50% | - | Low |
| Memory Views | - | 30-50% | Low |

**Overall Expected**: 3-5x faster HPO, 2x faster XAI, 30% less memory

---

## 7. Code Quality Improvements

### 7.1 Type Hints Enhancement
Add more specific type hints throughout, especially for return types of internal methods.

### 7.2 Error Handling
Add try-except blocks around external library calls (sklearn, scipy) with informative error messages.

### 7.3 Logging
Replace console prints with proper logging framework for production use.

### 7.4 Configuration Validation
Add validation in config dataclasses to ensure valid parameter ranges.

---

## 8. Testing Recommendations

For each optimization:
1. **Unit tests** - Test individual functions in isolation
2. **Performance benchmarks** - Measure time and memory improvements
3. **Regression tests** - Ensure results match current behavior
4. **Integration tests** - Test with real datasets

Example benchmark structure:
```python
def benchmark_hpo():
    X, y = make_classification(n_samples=1000, n_features=20)
    
    # Baseline
    config_old = HPOConfig(strategy="random", n_trials=50)
    time_old = measure_time(lambda: hpo_old.fit(X, y))
    
    # Optimized
    config_new = HPOConfig(strategy="random", n_trials=50, enable_early_stop=True)
    time_new = measure_time(lambda: hpo_new.fit(X, y))
    
    assert time_new < time_old
    print(f"Speedup: {time_old / time_new:.2f}x")
```

---

## 9. Documentation Updates Needed

1. Update docstrings for modified methods
2. Add examples for new features (warm start, SHAP, etc.)
3. Create migration guide for API changes
4. Update README with performance benchmarks
5. Add troubleshooting section for common issues

---

## 10. Backward Compatibility

Most optimizations are backward compatible. Breaking changes:
- New config parameters (all have defaults)
- Optional dependencies (shap, psutil) - gracefully degrade if missing
- API additions only, no removals

Migration path: Existing code continues to work, new features opt-in.

---

## Conclusion

These optimizations will significantly improve AxiomCore's efficiency while maintaining code quality and extensibility. The modular nature allows incremental implementation and testing.

**Next Steps**:
1. Implement high-priority optimizations first
2. Add comprehensive test coverage
3. Benchmark against current version
4. Update documentation
5. Consider optional dependency management

**Estimated Development Time**:
- High priority items: 2-3 days
- Medium priority items: 3-4 days
- Low priority items: 2-3 days
- Testing and documentation: 2-3 days

**Total**: ~10-13 days for complete implementation
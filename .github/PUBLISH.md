# Publishing AxiomCore to PyPI

This guide explains how to publish AxiomCore to PyPI so users can install it with `pip install axiomcore`.

## Prerequisites

1. **PyPI Account**: Create account at https://pypi.org/account/register/
2. **API Token**: Generate at https://pypi.org/manage/account/token/
3. **GitHub Secret**: Add token as `PYPI_API_TOKEN` in repository secrets

## Publishing Steps

### 1. Manual Publishing (First Time)

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI (optional, for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### 2. Automated Publishing (GitHub Actions)

The CI/CD pipeline automatically publishes to PyPI when you create a version tag:

```bash
# Update version in pyproject.toml
# Commit changes
git add pyproject.toml
git commit -m "Bump version to 0.3.0"

# Create and push tag
git tag v0.3.0
git push origin v0.3.0
```

The GitHub Action will:
1. Run all tests
2. Build the package
3. Publish to PyPI automatically

### 3. Verify Installation

After publishing, verify users can install:

```bash
pip install axiomcore

# Test CLI
axiomcore version

# Test import
python -c "from axiomcore import AxiomCore; print('Success!')"
```

## Version Management

Update version in these files:
- `pyproject.toml` - line 6
- `setup.py` - line 12
- `axiomcore.py` - `__version__` variable

## Package Structure

```
axiomcore/
├── axiomcore.py          # Main module
├── web_ui.py             # Web dashboard
├── axiomcore_web.py      # Web integration
├── cli.py                # CLI interface
├── templates/            # HTML templates
│   ├── index.html
│   └── dashboard.html
├── tests/                # Test suite
│   ├── test_hpo.py
│   └── test_xai.py
├── pyproject.toml        # Package metadata
├── setup.py              # Setup configuration
└── MANIFEST.in           # Package data
```

## PyPI Page Optimization

The package page on PyPI will show:
- **Description**: From README.md
- **Classifiers**: Python versions, license, topics
- **Keywords**: For search optimization
- **Project URLs**: GitHub, docs, issues

## Troubleshooting

### Build Errors

```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info/

# Rebuild
python -m build
```

### Upload Errors

```bash
# Check credentials
twine check dist/*

# Use API token (not password)
twine upload dist/* --username __token__ --password pypi-...
```

### Import Errors After Install

```bash
# Verify package contents
pip show -f axiomcore

# Reinstall in development mode
pip install -e .
```

## Release Checklist

- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Build package: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Test installation locally
- [ ] Create git tag
- [ ] Push tag to trigger CI/CD
- [ ] Verify PyPI page
- [ ] Test `pip install axiomcore`
- [ ] Update documentation

## GitHub Secrets Setup

1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `PYPI_API_TOKEN`
4. Value: Your PyPI API token (starts with `pypi-`)
5. Click "Add secret"

## TestPyPI (Optional)

For testing before real release:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ axiomcore
```

## Post-Publication

After successful publication:

1. Announce on GitHub Discussions
2. Update README badges
3. Share on social media
4. Monitor PyPI download stats
5. Respond to issues promptly

## Support

For publishing issues:
- PyPI Help: https://pypi.org/help/
- Twine Docs: https://twine.readthedocs.io/
- GitHub Actions: https://docs.github.com/en/actions
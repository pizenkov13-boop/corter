"""
Corter Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="corter-ml",
    version="0.1.0",
    author="Corter Team",
    author_email="support@corter.dev",
    description="Autonomous ML Optimization Framework - No cloud required, no GPU farm needed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pizenkov13-boop/Corter",
    packages=find_packages(),
    py_modules=["corter", "web_ui", "corter_web"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "requests>=2.31.0",
        "psutil>=5.9.0",
        "gunicorn>=21.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
        ],
        "xai": [
            "shap>=0.42.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "corter=corter_pkg.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.html", "*.yaml"],
    },
    data_files=[
        ("templates", ["templates/index.html", "templates/dashboard.html"]),
    ],
    zip_safe=False,
    keywords="machine-learning ml optimization hyperparameter-tuning explainable-ai xai automl",
    project_urls={
        "Bug Reports": "https://github.com/pizenkov13-boop/Corter/issues",
        "Source": "https://github.com/pizenkov13-boop/Corter",
        "Documentation": "https://github.com/pizenkov13-boop/Corter/blob/main/README.md",
    },
)

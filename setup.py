from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="macro-sentinel",
    version="1.0.0",
    description="AI-driven geo-economic stress prediction and country risk scoring platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Data Science & Analytics Team",
    python_requires=">=3.10",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=[
        "pandas>=2.2",
        "numpy>=1.26",
        "scikit-learn>=1.5",
        "xgboost>=2.0",
        "lightgbm>=4.3",
        "mlflow>=2.14",
        "streamlit>=1.35",
        "plotly>=5.22",
        "matplotlib>=3.8",
        "seaborn>=0.13",
        "PyYAML>=6.0",
        "joblib>=1.4",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "pytest-cov>=5.0"],
    },
    entry_points={
        "console_scripts": [
            "macro-sentinel=main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

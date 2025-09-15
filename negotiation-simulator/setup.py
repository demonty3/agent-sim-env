"""
Setup script for the AI Agent Negotiation Simulator package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f
                       if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "typer>=0.9.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
    ]

setup(
    name="negotiation-simulator",
    version="1.0.0",
    author="AI Negotiation Team",
    author_email="contact@negotiation-sim.ai",
    description="A sophisticated multi-agent negotiation simulator with game theory analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/negotiation-simulator",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "web": [
            "streamlit>=1.25.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
        "ml": [
            "scikit-learn>=1.3.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "negotiate=cli:app",
            "negotiate-web=web_interface:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
)

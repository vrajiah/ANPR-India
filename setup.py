#!/usr/bin/env python3
"""
ANPR System - Automatic Number Plate Recognition
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="anpr-system",
    version="1.0.0",
    author="ANPR Development Team",
    author_email="anpr@example.com",
    description="Advanced Automatic Number Plate Recognition System for Indian License Plates",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ANPR-System",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5.0,<4.0.0",
        "norfair>=2.1.0,<2.4.0",
        "numpy>=1.23.0,<2.0.0",
        "paddlepaddle>=2.4.0,<4.0.0",
        "paddleocr>=2.0.1",
        "pandas>=1.4.0,<3.0.0",
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "opencv-python-headless>=4.5.0,<5.0.0",
        "pillow>=7.1.0,<12.0.0",
        "watchdog>=2.1.0",
    ],
    extras_require={
        "web": [
            "streamlit>=1.28.0",
            "streamlit-option-menu>=0.3.6",
            "plotly>=5.17.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "torch>=1.13.0",
            "torchvision>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "anpr=anpr_system.cli:main",
            "anpr-web=anpr_system.web_app:run_web_app",
        ],
    },
    include_package_data=True,
    package_data={
        "anpr_system": [
            "*.py",
        ],
    },
    keywords="anpr, license-plate, recognition, computer-vision, yolo, ocr, indian-plates",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ANPR-System/issues",
        "Source": "https://github.com/yourusername/ANPR-System",
        "Documentation": "https://github.com/yourusername/ANPR-System#readme",
    },
)

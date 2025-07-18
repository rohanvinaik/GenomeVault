"""
GenomeVault Package Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="GenomeVault",
    version="3.0.0",
    author="GenomeVault Team",
    author_email="team@genomevault.org",
    description="Privacy-preserving genomic data platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GenomeVault",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.9",
    install_requires=[
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pydantic>=2.5.0",
        "numpy>=1.26.2",
        "torch>=2.1.1",
        "cryptography>=41.0.7",
        "web3>=6.11.3",
    ],
    entry_points={
        "console_scripts": [
            "genomevault=api.app:main",
            "genomevault-node=scripts.node_runner:main",
        ],
    },
)

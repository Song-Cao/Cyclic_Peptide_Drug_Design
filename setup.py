from setuptools import setup, find_packages

# Load the README file as the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Cyclic_Peptide_Drug_Design",
    version="0.1.0",
    author="Song Cao",
    author_email="sc2424@cam.ac.uk",
    description="A Python package for designing orally available cyclic peptides",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Song-Cao/Cyclic_Peptide_Drug_Design",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torchtext>=0.10.0",
        "scikit-learn>=0.24.2",
        "biopython>=1.79",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
        "optuna>=2.10.0",
        "tqdm>=4.62.3",
    ],
    extras_require={
        "rdkit": ["rdkit>=2021.09.4"], 
        "dev": ["jupyterlab>=3.2.1", "notebook>=6.4.5"],  
    },
    scripts=[
        'Cyclic_Peptide_Drug_Design/mol_peptide_data_querying.py',
        'Cyclic_Peptide_Drug_Design/contrastive_model.py',
        'Cyclic_Peptide_Drug_Design/diffusion_model.py',
        'Cyclic_Peptide_Drug_Design/RL_framework.py',
        'Cyclic_Peptide_Drug_Design/integrated_training.py',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
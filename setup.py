from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='CRISGI',
    version='0.1.0',
    author='Lyu C., Jiang A., Ng K. H., Liu X., Chen L.',
    author_email='',
    description='Charting Critical Transient Gene Interactions in Disease Progression from Multi-modal Transcriptomics',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/compbioclub/CRISGI',
    project_urls={
        'Documentation': 'https://compbioclub.github.io/CRISGI/',
        'Bug Tracker': 'https://github.com/compbioclub/CRISGI/issues',
        'Source Code': 'https://github.com/compbioclub/CRISGI',
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'crisgi': ['*.pk'],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires='>=3.9',
    install_requires=[
        # Core scientific computing
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        
        # Machine learning and statistics
        'scikit-learn>=1.0.0',
        'torch>=2.5.0',
        'torchvision>=0.20.0',
        'joblib>=1.0.0',
        
        # Bioinformatics and genomics
        'scanpy>=1.8.0',
        'anndata>=0.8.0',
        
        # Plotting and visualization
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'pyvis>=0.2.0',
        
        # Survival analysis
        'scikit-survival>=0.17.0',
        
        # Gene set enrichment analysis
        'gseapy>=1.0.0',
        
        # Statistical analysis
        'pymannkendall>=1.4.0',
        'pyseat',
        
        # Image processing (for utility functions)
        'Pillow>=8.0.0',
        
        # ipython and Jupyter support
        'ipykernel>=6.0.0',
        'ipython>=8.0.0',
        'jupyter>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.10',
            'black>=21.0',
            'flake8>=3.8',
            'isort>=5.0',
        ],
        'docs': [
            'mkdocs>=1.4.0',
            'mkdocs-material>=8.0.0',
            'mkdocstrings[python]>=0.19.0',
            'mknotebooks>=0.7.0',
        ],
        'tutorial': [
            'notebook>=6.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            # Add console scripts if needed in the future
        ],
    },
    keywords=[
        'bioinformatics',
        'genomics', 
        'critical transitions',
        'gene interactions',
        'transcriptomics',
        'single-cell',
        'spatial transcriptomics',
        'disease progression',
        'machine learning',
        'neural networks'
    ],
    license="MIT",
)

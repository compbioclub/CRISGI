# Installation Guide

## Quick Installation

### From PyPI (Recommended - when available)
```bash
pip install crisgi
```

### From Source (Development)
```bash
# Clone the repository
git clone https://github.com/compbioclub/CRISGI.git
cd CRISGI

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,docs,tutorial]"
```

### Using requirements.txt
```bash
# Basic installation
pip install -r requirements.txt

# Development installation
pip install -r requirements-dev.txt

# Testing environment
pip install -r requirements-test.txt
```

## Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

## Verification

Test your installation:
```python
import crisgi
print(crisgi.__version__)
```

## Common Issues

1. **PyTorch Installation**: If you encounter issues with PyTorch, install it separately:
   ```bash
   pip install torch torchvision torchaudio
   ```

2. **Scanpy Dependencies**: For some systems, you may need to install scanpy dependencies separately:
   ```bash
   pip install scanpy[leiden]
   ```

3. **Memory Requirements**: Some operations require significant memory. Ensure adequate RAM is available.

## Development Setup

For contributors:
```bash
# Clone and install
git clone https://github.com/compbioclub/CRISGI.git
cd CRISGI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,docs,tutorial]"

# Install pre-commit hooks
pre-commit install
```

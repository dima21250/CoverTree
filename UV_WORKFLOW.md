# UV Workflow Guide for CoverTree Development

This guide explains how to use `uv` for fast, efficient Python package management in the CoverTree project.

## Why UV?

- **10-100x faster** than pip for package installation
- **Better dependency resolution** - handles complex dependency trees correctly
- **Python version management** - can install and manage Python versions automatically
- **Lock files** - reproducible environments with `uv.lock`
- **Native macOS support** - works great on both Intel and Apple Silicon

## Initial Setup

### Install UV

```bash
# On macOS
brew install uv

# Or using the official installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create Virtual Environment

```bash
# Navigate to project directory
cd /Users/dima/Code/CoverTree

# Create virtual environment (uv will install Python 3.11 if needed based on .python-version)
uv venv

# Activate it
source .venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x
```

## Development Workflow

### Installing Dependencies

```bash
# Install runtime dependencies only
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"

# Install with test dependencies (for running test_ensemble.py etc.)
uv pip install -e ".[test]"

# Install everything
uv pip install -e ".[dev,test,docs]"
```

### Building the C++ Extension

```bash
# IMPORTANT: First time, install build dependencies
uv pip install setuptools wheel numpy

# Development build (fast iteration)
uv pip install -e . --no-build-isolation

# Clean rebuild
rm -rf build dist *.egg-info
uv pip install -e . --force-reinstall --no-build-isolation

# Build with specific compiler flags (for debugging)
CFLAGS="-g -O0" uv pip install -e . --no-build-isolation

# Alternative: Let UV handle build isolation (slower but automatic)
# This installs build deps automatically but creates isolated build
uv pip install -e .
```

### Running Tests

```bash
# Run basic tests
python test.py

# Run ensemble tests
python test_ensemble.py config.json

# Run with pytest (if using new test structure)
uv run pytest tests/

# Run with coverage
uv run pytest --cov=covertree --cov-report=html
```

### Adding New Dependencies

```bash
# Add a runtime dependency
uv pip install package-name
# Then add it to pyproject.toml [project.dependencies]

# Add a development dependency
uv pip install --dev package-name
# Then add it to pyproject.toml [project.optional-dependencies.dev]

# Or directly edit pyproject.toml and sync
uv pip sync
```

### Lock File Management

```bash
# Generate lock file (creates uv.lock)
uv lock

# Install from lock file (reproducible)
uv sync

# Update dependencies
uv lock --upgrade
```

## Common Tasks

### Switch Python Versions

```bash
# Change .python-version file
echo "3.12" > .python-version

# Recreate virtual environment
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,test]"
```

### Clean Environment

```bash
# Remove virtual environment
rm -rf .venv

# Remove build artifacts
rm -rf build dist *.egg-info
rm -f covertreec.*.so

# Start fresh
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,test]"
```

### Debugging Build Issues

```bash
# Verbose build output
uv pip install -e . --no-build-isolation -v

# See exact compiler commands
uv pip install -e . --no-build-isolation -vv

# Check what's installed
uv pip list

# Check dependency tree
uv pip tree
```

### Working with Multiple Environments

```bash
# Create environment for Python 3.11
uv venv --python 3.11 .venv-py311

# Create environment for Python 3.12
uv venv --python 3.12 .venv-py312

# Switch between them
source .venv-py311/bin/activate  # Use Python 3.11
source .venv-py312/bin/activate  # Use Python 3.12
```

## Integration with Modern Development Tools

### With Ruff (Linting)

```bash
# Install ruff
uv pip install ruff

# Lint code
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### With MyPy (Type Checking)

```bash
# Install mypy
uv pip install mypy

# Type check
mypy covertree/
```

### With Pytest

```bash
# Install pytest
uv pip install pytest pytest-cov

# Run tests
uv run pytest

# With coverage
uv run pytest --cov=covertree --cov-report=term-missing
```

### With IPython/Jupyter

```bash
# Install jupyter
uv pip install ipython jupyter notebook

# Start IPython
uv run ipython

# Start Jupyter
uv run jupyter notebook
```

## Performance Tips

### Faster Installations

```bash
# Use UV's cache (automatic)
# Located at ~/.cache/uv on macOS

# Clear cache if needed
uv cache clean

# Install without cache (for testing)
uv pip install --no-cache package-name
```

### Parallel Builds

```bash
# Use multiple cores for C++ compilation
export MAX_JOBS=8
uv pip install -e . --no-build-isolation
```

### Pre-built Binaries

```bash
# UV automatically uses pre-built wheels when available
# For packages like numpy, scipy, this is much faster

# Check what you have
uv pip list --format=json | grep -i platform
```

## Troubleshooting

### "command not found: uv"

```bash
# Ensure UV is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Or use Homebrew version
which uv  # Should show /opt/homebrew/bin/uv or /usr/local/bin/uv
```

### Build Fails on macOS

```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install

# Check compiler
clang++ --version

# Try with explicit compiler
CC=clang CXX=clang++ uv pip install -e . --no-build-isolation
```

### Numpy/Eigen Issues

```bash
# Install system Eigen if needed
brew install eigen

# Point to system includes
export CPLUS_INCLUDE_PATH=/opt/homebrew/include:$CPLUS_INCLUDE_PATH
uv pip install -e . --no-build-isolation
```

### Lock File Conflicts

```bash
# If uv.lock causes issues
rm uv.lock
uv lock

# Or ignore lock file temporarily
uv pip install -e . --no-lock
```

## Migration from pip/venv

If you're migrating from traditional pip:

```bash
# 1. Deactivate old environment
deactivate

# 2. Backup requirements if you have them
pip freeze > old-requirements.txt

# 3. Create UV environment
uv venv
source .venv/bin/activate

# 4. Install from pyproject.toml
uv pip install -e ".[dev,test]"

# 5. Verify everything works
python test.py
```

## Best Practices

1. **Always use `uv pip`** instead of `pip` when UV environment is active
2. **Commit `pyproject.toml`** to version control
3. **Commit `uv.lock`** for reproducible environments (optional)
4. **Use `uv sync`** on new machines instead of `pip install -r requirements.txt`
5. **Keep `.venv` in `.gitignore`**
6. **Use `--no-build-isolation`** for faster C++ extension rebuilds during development

## Quick Reference

```bash
# Setup
uv venv                           # Create virtual environment
source .venv/bin/activate         # Activate it

# Install
uv pip install -e ".[dev,test]"  # Development install
uv pip install package            # Add package

# Maintain
uv lock                          # Create/update lock file
uv sync                          # Install from lock file
uv pip list                      # List installed packages

# Clean
rm -rf .venv build dist          # Clean everything
uv venv && source .venv/bin/activate  # Start fresh
```

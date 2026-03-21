# Quick Start - Getting Started Right Now

This is your immediate action plan to start modernizing CoverTree on macOS with UV.

## Step 1: Install Prerequisites (5 minutes)

```bash
# Install Xcode Command Line Tools (if not already)
xcode-select --install

# Install Homebrew (if not already)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential tools
brew install uv cmake eigen

# Verify installations
uv --version      # Should show uv version
cmake --version   # Should show cmake version
clang++ --version # Should show Apple clang
```

## Step 2: Set Up UV Environment (2 minutes)

```bash
# Navigate to your project
cd /Users/dima/Code/CoverTree

# Create virtual environment (uv will install Python 3.11 automatically)
uv venv

# Activate it
source .venv/bin/activate

# Verify you're using the right Python
which python      # Should be .venv/bin/python
python --version  # Should be 3.11.x
```

## Step 3: Install Dependencies (1 minute - UV is FAST!)

```bash
# Install runtime dependencies
uv pip install numpy scipy scikit-learn

# Install development tools
uv pip install pytest pytest-cov ipython

# Install test dependencies (for your ensemble tests)
uv pip install sentence-transformers faiss-cpu pandas
```

## Step 4: Build Current Code (2 minutes)

```bash
# First install build dependencies (needed for C++ extensions)
uv pip install setuptools wheel numpy

# Now build the Python extension
uv pip install -e . --no-build-isolation

# This will show build messages and detect your Mac architecture
# On Apple Silicon you should see: "Detected Apple Silicon - using -mcpu=apple-m1"
# On Intel Mac you should see: "Detected Intel Mac - using -march=native"

# Alternative: Let UV handle build isolation (slower but automatic)
# uv pip install -e .
```

## Step 5: Verify Everything Works (2 minutes)

```bash
# Test import
python -c "from covertree import CoverTree; print('✓ Import works')"

# Run basic test
python test.py

# If you have test data, try ensemble test
# python test_ensemble.py your_config.json
```

## Step 6: Baseline Your Current Setup (5 minutes)

```bash
# Check for memory leaks (optional but recommended)
# This will help you verify improvements later
instruments -t Leaks python test.py

# Benchmark current performance
time python test.py > baseline_performance.txt

# Save current test output for comparison
python test_ensemble.py config.json > baseline_ensemble.txt 2>&1
```

## You're Ready!

You now have:
- ✅ Clean UV-based Python environment
- ✅ macOS-optimized build
- ✅ Current code working
- ✅ Performance baseline

## What's Next?

Choose your path:

### Path A: Quick Win (Recommended to start)
Goal: Add basic clustering and remove JSON hack

```bash
# Follow Phase 1 in IMPLEMENTATION_PLAN.md
# Should take ~1 week
# Gets you immediate value with minimal risk
```

### Path B: Full Modernization
Goal: Complete migration to pybind11 and modern APIs

```bash
# Follow full plan in IMPLEMENTATION_PLAN.md
# Takes 6-8 weeks
# Professional, production-ready result
```

### Path C: Just Fix Bugs First
Goal: Fix critical issues before adding features

```bash
# Fix FIXMEs in sgtree.cpp
# Fix memory leaks in covertreecmodule.cxx
# Add tests to prevent regressions
```

## Common Issues & Solutions

### Issue: "command not found: uv"

```bash
# Make sure Homebrew is in PATH
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
source ~/.zprofile

# Verify
which uv
```

### Issue: Build fails with "clang: error: unsupported option"

```bash
# Your system might not support the optimization flags
# Edit setup.py and change the flags to simpler ones

# For testing, try:
export CFLAGS="-O2"
uv pip install -e . --no-build-isolation --force-reinstall
```

### Issue: "Cannot import name CoverTree"

```bash
# Make sure you built the extension
ls -la *.so  # Should see covertreec.*.so

# If not there, rebuild:
rm -rf build dist *.egg-info
uv pip install -e . --no-build-isolation --force-reinstall
```

### Issue: test_ensemble.py needs config.json

```bash
# Create a minimal config file
cat > test_config.json << EOF
{
  "in_file": "your_data.csv"
}
EOF

# Run with it
python test_ensemble.py test_config.json
```

### Issue: Memory leak warnings

```bash
# Install with address sanitizer for detailed info
CFLAGS="-fsanitize=address -g" uv pip install -e . --no-build-isolation --force-reinstall

# Run test
python test.py

# If you see leaks, they're probably in covertreecmodule.cxx
# This is expected and will be fixed in Phase 0
```

## Pro Tips

1. **Use UV for everything Python-related**
   ```bash
   uv pip install package  # Not pip install
   uv run pytest           # Not just pytest
   ```

2. **Keep rebuilding fast**
   ```bash
   # Use --no-build-isolation for faster rebuilds during development
   uv pip install -e . --no-build-isolation --force-reinstall
   ```

3. **Check what's installed**
   ```bash
   uv pip list
   uv pip tree  # Shows dependencies
   ```

4. **Activate environment automatically**
   ```bash
   # Add to ~/.zshrc or ~/.bashrc
   alias ct='cd /Users/dima/Code/CoverTree && source .venv/bin/activate'

   # Then just type 'ct' to jump in
   ```

5. **Use separate environments for testing**
   ```bash
   # Create test environment
   uv venv --python 3.12 .venv-py312

   # Test compatibility
   source .venv-py312/bin/activate
   uv pip install -e .
   python test.py
   ```

## Development Workflow

Daily workflow looks like:

```bash
# 1. Activate environment
cd /Users/dima/Code/CoverTree
source .venv/bin/activate

# 2. Make changes to C++ or Python code

# 3. Rebuild if you changed C++
uv pip install -e . --no-build-isolation --force-reinstall

# 4. Test
python test.py

# 5. Commit
git add .
git commit -m "Your changes"
```

## Getting Help

If you get stuck:

1. Check `IMPLEMENTATION_PLAN.md` for detailed steps
2. Check `UV_WORKFLOW.md` for UV-specific help
3. Check `CLAUDE.md` for project structure
4. Look at existing code for examples
5. Run with verbose output:
   ```bash
   uv pip install -e . --no-build-isolation -vv
   ```

## Next Steps

After you've completed this quick start:

1. Read through `IMPLEMENTATION_PLAN.md`
2. Decide which phase to start with
3. Create a git branch for your work:
   ```bash
   git checkout -b feature/modernization
   ```
4. Start coding!

Good luck! 🚀

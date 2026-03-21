# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements parallel C++14 implementations of two tree-based nearest neighbor search data structures:
- **Cover Tree**: Based on academic papers by Beygelzimer et al. (2006) and Izbicki & Shelton (2015)
- **SG-Tree**: A new data structure inspired by Cover Tree, optimized for hierarchical clustering with dimensional pruning

Both implementations are thread-safe, highly parallelizable, and provide Python bindings.

## macOS-Specific Setup

### Prerequisites
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install cmake eigen

# Install uv (fast Python package manager)
brew install uv

# uv will manage Python versions - no need to install python separately
# Create virtual environment with uv (automatically installs Python if needed)
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install Python dependencies with uv (much faster than pip)
uv pip install numpy scipy scikit-learn pytest
```

### macOS Build Notes

- Default compiler is `clang++` (Apple clang), not `g++`
- On Apple Silicon (M1/M2/M3), use `-mcpu=apple-m1` instead of `-march=core-avx2`
- On Intel Mac, use `-march=native` for optimal performance
- Homebrew paths differ by architecture:
  - Apple Silicon: `/opt/homebrew`
  - Intel: `/usr/local`
- The makefiles detect macOS automatically but prefer using `make llvm` for best results
- For OpenMP support: `brew install libomp` and add appropriate flags to setup.py

## Build System

CoverTree supports **two build systems**:

### Option 1: CMake (Recommended - Modern & Fast)

```bash
# Quick build (C++ executables)
./build.sh

# Build everything (C++ + Python)
./build.sh --python

# Clean build
./build.sh --clean --python

# Debug build
./build.sh --debug

# Or manually with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

**Benefits:**
- 3-4x faster parallel compilation
- Automatic dependency detection
- IDE integration (VS Code, CLion, Xcode)
- Professional build infrastructure

See [CMAKE_BUILD.md](CMAKE_BUILD.md) for detailed guide.

### Option 2: Makefile (Traditional)

```bash
# Build all modules (uses g++ by default)
make

# Build with Intel C++ Compiler
make intel

# Build with LLVM/clang
make llvm

# Build specific module
make cover_tree
make sg_tree

# Clean everything
make clean
```

### Build Output
- Object files: `build/<module-name>/`
- Executables: `dist/<module-name>`
- Binary data: `data/` (not tracked in git)

### Compiler Requirements
- gcc >= 5.0 or Intel C++ Compiler 2017 (for C++14 support)
- C++14 standard is required
- Default: AVX2 instructions (`-march=core-avx2`)
- **Troubleshooting**: If build fails with "instruction not found", change `march=core-avx2` to `march=corei7` in `setup.py` and `src/*/makefile`

## Python Integration

### Installation

```bash
# Activate virtual environment first
source .venv/bin/activate

# Install in development mode (recommended)
uv pip install -e .

# Or install normally
uv pip install .
```

The package `covertree` will be installed, providing Python bindings to the C++ Cover Tree implementation.

**Note**: Use `uv pip` instead of regular `pip` for faster installations and better dependency resolution.

### Python API Usage

```python
from covertree import CoverTree
import numpy as np

# Create cover tree from numpy matrix (row-major)
points = np.random.rand(1000, 128)
ct = CoverTree.from_matrix(points)

# Nearest neighbor search
query = np.random.rand(10, 128)
results = ct.NearestNeighbour(query)

# k-nearest neighbors
k_results = ct.kNearestNeighbours(query, k=10)

# Insert/remove points
ct.insert(new_point)
ct.remove(existing_point)

# Verify tree properties
ct.test_covering()
```

## Architecture

### Module Structure

```
src/
├── commons/          # Shared utilities (no executable)
├── cover_tree/       # Cover Tree implementation
│   ├── cover_tree.h/cpp     # Core CoverTree class
│   ├── covertreecmodule.cxx # Python bindings
│   ├── main.cpp             # Standalone C++ executable
│   └── utils.h              # Helper functions
└── sg_tree/          # SG-Tree implementation
    ├── sgtree.h/cpp         # Core SGTree class
    ├── sgtreemodule.cxx     # Python bindings
    ├── main.cpp             # Standalone C++ executable
    ├── cover_tree.h         # Symlink to ../cover_tree/cover_tree.h
    └── utils.h              # Helper functions
```

**Important**: `src/sg_tree/cover_tree.h` is a symbolic link to `src/cover_tree/cover_tree.h`. Don't edit it directly.

### CoverTree Class Design

The `CoverTree` class (in both modules) uses Eigen for linear algebra:
- **Point type**: `Eigen::VectorXd` (typedef'd as `pointType`)
- **Thread-safe**: Uses `shared_mutex` (clang) or `shared_timed_mutex` (gcc) for concurrent operations
- **Hierarchical structure**: Nodes organized in levels with parent-child relationships
- **Base expansion**: Uses base=1.3 for level calculations with precomputed power table

### Key Public Methods

```cpp
// Construction
static CoverTree* from_matrix(Eigen::MatrixXd& pMatrix, int truncate = -1, bool use_multi_core = true);
static CoverTree* from_points(std::vector<pointType>& pList, int truncate = -1, bool use_multi_core = true);

// Queries
std::pair<Node*, double> NearestNeighbour(const pointType& p);
std::vector<std::pair<Node*, double>> kNearestNeighbours(const pointType& p, unsigned k = 10);
std::vector<std::pair<Node*, double>> rangeNeighbours(const pointType& queryPt, double range = 1.0);

// Modifications
bool insert(const pointType& p);
bool remove(const pointType& p);

// Serialization (for distributed computing)
char* serialize();
void deserialize(char* buff);

// Validation
bool check_covering();
```

## Data Format

The C++ executables use a binary format for point files:

```
[numPoints: int32] [numDims: int32] [point_data: double array]
```

Generate test data:
```bash
python data/generateData.py
```

Run Cover Tree standalone:
```bash
dist/cover_tree data/train_100d_1000k_1000.dat data/test_100d_1000k_10.dat
```

## Dependencies

- **Eigen**: Header-only linear algebra library in `lib/Eigen/`
- **Python packages** (for Python bindings):
  - numpy >= 1.13.1
  - scipy >= 0.17
  - scikit-learn >= 0.18.1

## Testing

Run Python tests:
```bash
python test.py                    # Main test suite
python test_ensemble.py           # Ensemble tests
python test_text_input.py         # Text input tests
python test_slice_n_dice_2.py     # Slicing tests
```

## Performance Notes

- Can insert 1M vectors (1000 dimensions, L2 norm) in <250 seconds on c4.8xlarge
- Query performance: >300 queries/second/core
- Multi-core construction enabled by default via `use_multi_core=true` parameter
- Thread parallelism uses `std::thread` and `std::future`

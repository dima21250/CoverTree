# CoverTree Modernization Plan (macOS + UV Edition)

Complete strategy and implementation plan for modernizing CoverTree with clean clustering APIs, macOS support, and UV package management.

## Executive Summary

**Current Issues:**
- Python bindings use unsafe raw pointer casting
- Memory leaks in C++ extension
- No native clustering API (requires JSON parsing hack)
- WSL-focused, needs macOS optimization
- Traditional pip/venv workflow

**Goals:**
- Clean, modern Python bindings with pybind11
- Native clustering at multiple granularities
- macOS-optimized (Apple Silicon + Intel)
- Fast development with UV
- Production-ready memory management

**Timeline:** 6-8 weeks for complete migration

---

## Phase 0: macOS + UV Foundation (Week 1)

### Day 1-2: Environment Setup with UV

```bash
# Install system dependencies
xcode-select --install
brew install cmake eigen uv

# Create UV-managed environment
cd /Users/dima/Code/CoverTree
uv venv
source .venv/bin/activate

# Install dependencies with UV (10-100x faster than pip!)
uv pip install -e ".[dev,test]"

# Verify current build works
python test.py
```

**Deliverables:**
- [ ] UV environment working
- [ ] All dependencies installed via UV
- [ ] Current code builds and runs on macOS
- [ ] Document any macOS-specific issues

### Day 3-4: Fix Immediate C++ Issues

**Tasks:**
1. Fix FIXMEs in `src/sg_tree/sgtree.cpp`:
   ```bash
   grep -n "FIXME" src/sg_tree/sgtree.cpp
   # Lines 926, 946, 967 - duplicate code in if/else blocks
   ```

2. Fix memory leaks in `covertreecmodule.cxx`:
   ```cpp
   // Current (bad):
   double *results = new double[numDims*numPoints];
   return PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, results);
   // Never freed!

   // Fixed:
   npy_intp dims[2] = {numPoints, numDims};
   PyObject* out = PyArray_SimpleNew(2, dims, NPY_FLOAT64);
   double* results = static_cast<double*>(PyArray_DATA(out));
   // numpy owns memory
   ```

3. Add basic RAII wrappers

**Deliverables:**
- [ ] All FIXMEs resolved
- [ ] Memory leaks patched
- [ ] Address Sanitizer tests pass:
   ```bash
   CFLAGS="-fsanitize=address -g" uv pip install -e . --no-build-isolation --force-reinstall
   python test.py
   ```

### Day 5: Verify & Document

```bash
# Run all tests
python test.py
python test_ensemble.py sample_config.json

# Memory test
instruments -t Leaks python test.py

# Performance baseline
time python test.py > baseline_performance.txt
```

**Deliverables:**
- [ ] All existing tests pass on macOS
- [ ] Performance baseline documented
- [ ] Known issues documented in `KNOWN_ISSUES.md`

---

## Phase 1: Quick Win - Basic Clustering (Week 2)

Goal: Replace JSON parsing hack with native C++ clustering API

### C++ Changes (`src/cover_tree/cover_tree.h`)

Add these methods:

```cpp
class CoverTree {
public:
    // Tree metadata
    int getMinLevel() const { return min_scale.load(); }
    int getMaxLevel() const { return max_scale.load(); }
    std::map<int, unsigned> getLevelCounts() const;

    // Get nodes at specific level
    std::vector<Node*> getNodesAtLevel(int level) const;

    // Get point IDs in a node's subtree
    std::vector<unsigned> getSubtreePointIDs(Node* node) const;
    std::vector<pointType> getSubtreePoints(Node* node) const;

    // Simple clustering interface
    struct ClusterInfo {
        unsigned node_id;
        int level;
        pointType center;
        std::vector<unsigned> point_ids;
        double covering_distance;
    };
    std::vector<ClusterInfo> getClustersAtLevel(int level) const;
};
```

### Python Bindings (`covertreecmodule.cxx`)

Add these functions:

```cpp
static PyObject *covertreec_get_level_stats(PyObject *self, PyObject *args);
static PyObject *covertreec_get_clusters_at_level(PyObject *self, PyObject *args);
```

### Update Python Wrapper (`covertree/covertree.py`)

```python
class CoverTree(object):
    # Existing methods...

    def get_level_stats(self):
        """Get tree level statistics."""
        return covertreec.get_level_stats(self.this)

    def get_clusters_at_level(self, level):
        """Get clusters at specified level.

        Returns:
            list of dicts with keys: 'node_id', 'center', 'point_ids', 'level'
        """
        return covertreec.get_clusters_at_level(self.this, level)
```

### Refactor test_ensemble.py

**Before (hack):**
```python
ct_json = ct.dumps()
ct_data = json.loads(ct_json)
node_data = [{"id":n["id"], "level":n["level"], ...} for n in ct_data["nodes"]]
```

**After (clean):**
```python
stats = ct.get_level_stats()
print(f"Tree has {stats['num_levels']} levels")

# Get coarse clusters
clusters = ct.get_clusters_at_level(stats['max_level'] - 3)
for cluster in clusters:
    print(f"Cluster {cluster['node_id']}: {len(cluster['point_ids'])} points")
```

### Build & Test with UV

```bash
# Rebuild
rm -rf build dist *.egg-info
uv pip install -e . --no-build-isolation --force-reinstall

# Test
python test_ensemble.py config.json

# Verify no memory leaks
instruments -t Leaks python test_ensemble.py config.json
```

**Deliverables:**
- [ ] Native clustering methods in C++
- [ ] Python bindings for clustering
- [ ] `test_ensemble.py` uses native API (no JSON hack)
- [ ] 50%+ speed improvement in test_ensemble.py
- [ ] Documentation in docstrings

---

## Phase 2: Modern Build System (Week 3)

### Create CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(CoverTree CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # For IDE support

# Detect macOS architecture
if(APPLE)
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Building for macOS on ${ARCH}")

    if(ARCH STREQUAL "arm64")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
        message(STATUS "Using Apple Silicon optimizations")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        message(STATUS "Using Intel Mac optimizations")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

# Find Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR} lib/)

# Cover Tree library
add_library(covertree_core SHARED
    src/cover_tree/cover_tree.cpp
)
target_include_directories(covertree_core PUBLIC
    src/cover_tree
    src/commons
)
target_link_libraries(covertree_core pthread)

# Executables
add_executable(cover_tree src/cover_tree/main.cpp)
target_link_libraries(cover_tree covertree_core)

add_executable(sg_tree src/sg_tree/main.cpp src/sg_tree/sgtree.cpp)
target_link_libraries(sg_tree covertree_core pthread)

# Install targets
install(TARGETS covertree_core cover_tree sg_tree
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
```

### Update pyproject.toml for CMake builds

```toml
[build-system]
requires = ["setuptools>=61.0", "numpy>=1.13.1", "cmake>=3.15"]
build-backend = "setuptools.build_meta"

[tool.cmake]
minimum_version = "3.15"
build_type = "Release"
```

### Build with CMake

```bash
# Create build directory
mkdir -p build && cd build

# Configure (UV will find Python automatically)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . -j$(sysctl -n hw.ncpu)

# Install
cmake --install . --prefix=../dist

# Test
cd ..
./dist/bin/cover_tree data/train.dat data/test.dat
```

**Deliverables:**
- [ ] CMake build working
- [ ] Faster compilation (parallel builds)
- [ ] Better IDE support (compile_commands.json)
- [ ] Makefile still works (backward compatibility)

---

## Phase 3: Pybind11 Migration (Weeks 4-5)

### Add pybind11

```bash
# Add as git submodule
git submodule add https://github.com/pybind/pybind11.git lib/pybind11
git submodule update --init --recursive

# Or let UV install it
uv pip install pybind11
```

### Create New Binding File

Create `src/cover_tree/covertree_pybind.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "cover_tree.h"

namespace py = pybind11;

// Custom holder type for automatic memory management
using CoverTreePtr = std::shared_ptr<CoverTree>;

PYBIND11_MODULE(covertree2, m) {
    m.doc() = "Modern CoverTree bindings with clustering support";

    // Cluster info struct
    py::class_<CoverTree::ClusterInfo>(m, "ClusterInfo")
        .def_readonly("node_id", &CoverTree::ClusterInfo::node_id)
        .def_readonly("level", &CoverTree::ClusterInfo::level)
        .def_readonly("center", &CoverTree::ClusterInfo::center)
        .def_readonly("point_ids", &CoverTree::ClusterInfo::point_ids)
        .def_readonly("covering_distance", &CoverTree::ClusterInfo::covering_distance)
        .def("__repr__", [](const CoverTree::ClusterInfo& c) {
            return "<Cluster node=" + std::to_string(c.node_id) +
                   " level=" + std::to_string(c.level) +
                   " size=" + std::to_string(c.point_ids.size()) + ">";
        });

    // Main CoverTree class
    py::class_<CoverTree, CoverTreePtr>(m, "CoverTree")
        .def(py::init<>())

        // Construction
        .def_static("from_matrix",
            [](Eigen::Ref<Eigen::MatrixXd> matrix, int truncate, bool multicore) {
                return CoverTreePtr(CoverTree::from_matrix(matrix, truncate, multicore));
            },
            py::arg("matrix"),
            py::arg("truncate") = -1,
            py::arg("multicore") = true,
            "Build cover tree from matrix (rows are points)")

        // Queries - automatic numpy conversion!
        .def("nearest_neighbor",
            [](CoverTreePtr self, Eigen::Ref<Eigen::MatrixXd> query) {
                long numPoints = query.rows();
                long numDims = query.cols();
                Eigen::MatrixXd results(numPoints, numDims);

                for(long i = 0; i < numPoints; ++i) {
                    auto nn = self->NearestNeighbour(query.row(i));
                    results.row(i) = nn.first->_p;
                }
                return results;
            },
            py::arg("query"),
            "Find nearest neighbors")

        .def("k_nearest_neighbors",
            [](CoverTreePtr self, Eigen::Ref<Eigen::MatrixXd> query, int k) {
                long numPoints = query.rows();
                long numDims = query.cols();

                // Return as list of lists for clarity
                std::vector<std::vector<Eigen::VectorXd>> results;
                results.reserve(numPoints);

                for(long i = 0; i < numPoints; ++i) {
                    auto knn = self->kNearestNeighbours(query.row(i), k);
                    std::vector<Eigen::VectorXd> row_results;
                    row_results.reserve(k);
                    for(const auto& neighbor : knn) {
                        row_results.push_back(neighbor.first->_p);
                    }
                    results.push_back(row_results);
                }
                return results;
            },
            py::arg("query"),
            py::arg("k") = 10,
            "Find k nearest neighbors")

        // Clustering interface
        .def("get_clusters_at_level",
            &CoverTree::getClustersAtLevel,
            py::arg("level"),
            "Get clusters at specified tree level")

        .def("get_level_counts",
            &CoverTree::getLevelCounts,
            "Get number of nodes at each level")

        // Modifications
        .def("insert",
            py::overload_cast<const pointType&>(&CoverTree::insert),
            py::arg("point"),
            "Insert a point into the tree")

        .def("remove",
            py::overload_cast<const pointType&>(&CoverTree::remove),
            py::arg("point"),
            "Remove a point from the tree")

        // Properties (no function call needed!)
        .def_property_readonly("min_level", &CoverTree::getMinLevel)
        .def_property_readonly("max_level", &CoverTree::getMaxLevel)
        .def_property_readonly("num_points",
            [](const CoverTree& self) { return self.count_points(); })

        // Validation
        .def("check_covering",
            &CoverTree::check_covering,
            "Verify tree properties")

        // String representation
        .def("__repr__",
            [](const CoverTree& tree) {
                return "<CoverTree points=" + std::to_string(tree.count_points()) +
                       " levels=" + std::to_string(tree.getMaxLevel() - tree.getMinLevel() + 1) +
                       ">";
            });
}
```

### Update setup.py for dual modules

```python
# Keep old module for backward compatibility
covertreec_module = Extension(...)

# Add new pybind11 module
covertree2_module = Extension(
    'covertree2',
    sources=[
        'src/cover_tree/covertree_pybind.cpp',
        'src/cover_tree/cover_tree.cpp'
    ],
    include_dirs=['lib/', 'lib/pybind11/include'],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language='c++'
)

setup(
    ext_modules=[covertreec_module, covertree2_module],
    ...
)
```

### Build with UV

```bash
# Rebuild with both modules
rm -rf build dist *.egg-info
uv pip install -e . --no-build-isolation --force-reinstall

# Test old module still works
python -c "from covertree import CoverTree; print('Old API works')"

# Test new module
python -c "import covertree2; print('New API works')"
```

### Create High-Level Python API

Create `covertree/modern.py`:

```python
"""Modern, Pythonic interface to CoverTree."""
from typing import List, Optional, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import covertree2

@dataclass
class Cluster:
    """Represents a cluster from the cover tree."""
    node_id: int
    level: int
    center: np.ndarray
    point_ids: np.ndarray
    covering_distance: float

    @property
    def size(self) -> int:
        return len(self.point_ids)

    def get_points(self, data: np.ndarray) -> np.ndarray:
        """Get actual points in this cluster from original data."""
        return data[self.point_ids]

    def __repr__(self) -> str:
        return f"<Cluster(id={self.node_id}, level={self.level}, size={self.size})>"


class HierarchicalCoverTree:
    """High-level interface for hierarchical clustering with CoverTree.

    Example:
        >>> hct = HierarchicalCoverTree()
        >>> hct.fit(data)
        >>> clusters = hct.get_clusters(n_clusters=10)
        >>> for cluster in clusters:
        ...     print(f"Cluster {cluster.node_id}: {cluster.size} points")
    """

    def __init__(self, truncate: int = -1, multicore: bool = True):
        self._tree: Optional[covertree2.CoverTree] = None
        self._data: Optional[np.ndarray] = None
        self.truncate = truncate
        self.multicore = multicore

    def fit(self, X: np.ndarray) -> 'HierarchicalCoverTree':
        """Fit the cover tree to data.

        Args:
            X: Data matrix (n_samples, n_features)

        Returns:
            self
        """
        self._data = np.asarray(X, dtype=np.float64)
        self._tree = covertree2.CoverTree.from_matrix(
            self._data,
            self.truncate,
            self.multicore
        )
        return self

    def get_clusters(self,
                    n_clusters: Optional[int] = None,
                    level: Optional[int] = None) -> List[Cluster]:
        """Get clusters at different granularities.

        Args:
            n_clusters: Desired number of clusters (approximate)
            level: Specific tree level to extract clusters from

        Returns:
            List of Cluster objects
        """
        if self._tree is None:
            raise ValueError("Must call fit() before get_clusters()")

        if level is not None:
            target_level = level
        elif n_clusters is not None:
            target_level = self._find_level_for_count(n_clusters)
        else:
            raise ValueError("Must specify either n_clusters or level")

        cluster_infos = self._tree.get_clusters_at_level(target_level)

        return [
            Cluster(
                node_id=c.node_id,
                level=c.level,
                center=c.center,
                point_ids=np.array(c.point_ids, dtype=np.int64),
                covering_distance=c.covering_distance
            )
            for c in cluster_infos
        ]

    def _find_level_for_count(self, target_count: int) -> int:
        """Find tree level that gives approximately target_count clusters."""
        level_counts = self._tree.get_level_counts()

        # Find level with count closest to target
        best_level = self._tree.max_level
        best_diff = float('inf')

        for level, count in level_counts.items():
            diff = abs(count - target_count)
            if diff < best_diff:
                best_diff = diff
                best_level = level

        return best_level

    @property
    def min_level(self) -> int:
        """Minimum level in tree."""
        return self._tree.min_level if self._tree else 0

    @property
    def max_level(self) -> int:
        """Maximum level in tree."""
        return self._tree.max_level if self._tree else 0

    @property
    def num_levels(self) -> int:
        """Number of levels in tree."""
        return self.max_level - self.min_level + 1 if self._tree else 0

    def get_level_info(self) -> Dict[int, int]:
        """Get number of nodes at each level."""
        if self._tree is None:
            return {}
        return self._tree.get_level_counts()
```

### Test New API

Create `test_modern_api.py`:

```python
import numpy as np
from covertree.modern import HierarchicalCoverTree

# Generate test data
np.random.seed(42)
data = np.random.randn(1000, 128)

# Fit tree
hct = HierarchicalCoverTree()
hct.fit(data)

print(f"Tree has {hct.num_levels} levels")
print(f"Level range: {hct.min_level} to {hct.max_level}")

# Get coarse clusters
print("\nCoarse clustering (10 clusters):")
coarse = hct.get_clusters(n_clusters=10)
for cluster in coarse:
    print(f"  {cluster}")
    points = cluster.get_points(data)
    print(f"    Mean: {points.mean(axis=0)[:5]}...")

# Get fine clusters
print("\nFine clustering (100 clusters):")
fine = hct.get_clusters(n_clusters=100)
print(f"  Got {len(fine)} clusters")
print(f"  Sizes: min={min(c.size for c in fine)}, "
      f"max={max(c.size for c in fine)}, "
      f"mean={np.mean([c.size for c in fine]):.1f}")
```

**Deliverables:**
- [ ] pybind11 module (`covertree2`) working
- [ ] All old functionality available in new API
- [ ] New clustering features working
- [ ] High-level Python API (`HierarchicalCoverTree`)
- [ ] Tests pass for both old and new APIs
- [ ] Documentation and examples

---

## Phase 4: Scikit-learn Compatible API (Week 6)

Create `covertree/sklearn.py`:

```python
"""Scikit-learn compatible clustering interface."""
from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from .modern import HierarchicalCoverTree

class CoverTreeClusterer(BaseEstimator, ClusterMixin):
    """Scikit-learn compatible CoverTree clusterer.

    Example:
        >>> from covertree.sklearn import CoverTreeClusterer
        >>> clusterer = CoverTreeClusterer(n_clusters=5)
        >>> labels = clusterer.fit_predict(X)
    """

    def __init__(self,
                 n_clusters: int = 8,
                 truncate: int = -1,
                 multicore: bool = True):
        self.n_clusters = n_clusters
        self.truncate = truncate
        self.multicore = multicore

    def fit(self, X, y=None):
        """Fit the clusterer."""
        X = check_array(X, dtype=np.float64)

        self.tree_ = HierarchicalCoverTree(
            truncate=self.truncate,
            multicore=self.multicore
        )
        self.tree_.fit(X)

        # Get clusters
        clusters = self.tree_.get_clusters(n_clusters=self.n_clusters)

        # Assign labels
        self.labels_ = np.full(X.shape[0], -1, dtype=np.int64)
        self.cluster_centers_ = np.zeros((len(clusters), X.shape[1]))

        for i, cluster in enumerate(clusters):
            self.labels_[cluster.point_ids] = i
            self.cluster_centers_[i] = cluster.center

        return self

    def fit_predict(self, X, y=None):
        """Fit and return cluster labels."""
        self.fit(X, y)
        return self.labels_

    def predict(self, X):
        """Predict cluster labels for new data."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        # Use nearest neighbor to assign to clusters
        # (implementation detail)
        raise NotImplementedError("Prediction not yet implemented")
```

**Deliverables:**
- [ ] Scikit-learn compatible API
- [ ] Works with sklearn pipelines
- [ ] Passes sklearn estimator checks

---

## Phase 5: Testing & Documentation (Week 7)

### Create Test Suite

```bash
# Create test directory
mkdir -p tests
```

Create `tests/test_core.py`:

```python
import pytest
import numpy as np
import covertree2

def test_basic_construction():
    """Test basic tree construction."""
    data = np.random.randn(100, 10)
    tree = covertree2.CoverTree.from_matrix(data)
    assert tree.num_points == 100

def test_nearest_neighbor():
    """Test NN search."""
    data = np.random.randn(100, 10)
    tree = covertree2.CoverTree.from_matrix(data)

    query = np.random.randn(5, 10)
    results = tree.nearest_neighbor(query)

    assert results.shape == (5, 10)

def test_clustering():
    """Test clustering extraction."""
    data = np.random.randn(100, 10)
    tree = covertree2.CoverTree.from_matrix(data)

    clusters = tree.get_clusters_at_level(tree.max_level - 2)
    assert len(clusters) > 0

    total_points = sum(len(c.point_ids) for c in clusters)
    assert total_points == 100
```

### Run Tests with UV

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Run tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=covertree --cov-report=html

# View coverage
open htmlcov/index.html
```

### Create Documentation

Create `docs/` directory with:
- API reference
- Tutorial notebooks
- Migration guide from old API
- Performance benchmarks

**Deliverables:**
- [ ] Comprehensive test suite (>80% coverage)
- [ ] All tests pass on macOS (Intel + Apple Silicon)
- [ ] Memory leak tests pass
- [ ] Performance benchmarks documented
- [ ] User documentation complete

---

## Phase 6: Cleanup & Release (Week 8)

### Deprecation Plan

1. Keep old API (`covertree`) for one release
2. Add deprecation warnings
3. Update all examples to use new API
4. Release v2.0 with both APIs
5. Release v3.0 removing old API

### Update CLAUDE.md

Add migration guide and modern workflows.

### Create Migration Script

`scripts/migrate_to_modern.py`:

```python
"""Helper script to migrate from old to new API."""
import sys
import re

def migrate_file(filepath):
    with open(filepath) as f:
        content = f.read()

    # Replace common patterns
    content = re.sub(
        r'from covertree import CoverTree',
        'from covertree.modern import HierarchicalCoverTree as CoverTree',
        content
    )

    # More replacements...

    print(f"Migrated {filepath}")
    # Optionally write back
    # with open(filepath, 'w') as f:
    #     f.write(content)

if __name__ == '__main__':
    for filepath in sys.argv[1:]:
        migrate_file(filepath)
```

**Deliverables:**
- [ ] Migration guide published
- [ ] All tests green
- [ ] Performance equal or better
- [ ] Documentation complete
- [ ] Tagged release v2.0

---

## Quick Start Guide

For immediate use:

```bash
# Setup (5 minutes)
brew install uv cmake eigen
cd /Users/dima/Code/CoverTree
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,test]"

# Test current code
python test.py

# Start development
# ... follow Phase 0 tasks above
```

## Success Metrics

- [ ] No memory leaks (verified with instruments)
- [ ] 10x faster than JSON parsing for clustering
- [ ] Type hints throughout Python code
- [ ] >80% test coverage
- [ ] Works on both Intel and Apple Silicon Macs
- [ ] UV-based workflow <30 seconds from clone to test

## Resources

- [UV Documentation](https://github.com/astral-sh/uv)
- [pybind11 Documentation](https://pybind11.readthedocs.io/)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)
- [Modern Python Packaging](https://packaging.python.org/)

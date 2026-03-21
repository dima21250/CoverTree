# pybind11 Migration Guide

This guide explains the differences between the old C API bindings (`covertree`) and the new pybind11 bindings (`covertree2`).

## Quick Start

Both modules work side-by-side:

```python
# Old API (still works)
from covertree import CoverTree

# New API (modern, recommended)
from covertree2 import CoverTree
```

## Key Improvements

### 1. Type Safety ✅

**Before (old):**
```python
# Raw pointer casting, potential memory issues
ct = CoverTree.from_matrix(data)  # Returns wrapped pointer
```

**After (new):**
```python
# Proper C++ objects with automatic memory management
tree = CoverTree.from_matrix(data)  # Returns smart pointer
```

### 2. Automatic Numpy ↔ Eigen Conversion ✅

**Before (old):**
```python
# Manual memory allocation and copying
results = ct.NearestNeighbour(queries)
```

**After (new):**
```python
# Automatic conversion, no copies unless necessary
results = tree.nearest_neighbor(queries)
```

### 3. Pythonic Properties ✅

**Before (old):**
```python
# No direct access to tree properties
stats = ct.get_level_stats()
min_level = stats['min_level']
```

**After (new):**
```python
# Direct property access
min_level = tree.min_level
max_level = tree.max_level
num_points = tree.num_points
```

### 4. Better Error Messages ✅

**Before (old):**
```
TypeError: covertreec.NearestNeighbour() argument 1 must be numpy.ndarray, not list
```

**After (new):**
```
TypeError: nearest_neighbor(): incompatible function arguments. The following argument types are supported:
    1. (self: covertree2.CoverTree, query: numpy.ndarray[numpy.float64]) -> numpy.ndarray[numpy.float64]

Invoked with: <CoverTree(points=200, levels=8)>, [1.0, 2.0, 3.0]
```

### 5. Modern C++ ✅

**Before (old):**
- C-style API
- Manual memory management
- Raw pointer casting
- No RAII

**After (new):**
- C++17 features
- Automatic memory management
- Smart pointers
- Full RAII

## API Comparison

### Construction

```python
# Both APIs - Same interface
tree = CoverTree.from_matrix(X, truncate=-1, multicore=True)
```

### Nearest Neighbor

```python
# Old API
results = ct.NearestNeighbour(queries)  # British spelling

# New API
results = tree.nearest_neighbor(queries)  # American spelling, snake_case
```

### k-Nearest Neighbors

```python
# Old API
results = ct.kNearestNeighbours(queries, k=10)  # camelCase

# New API
results = tree.k_nearest_neighbors(queries, k=10)  # snake_case
```

### Range Queries

```python
# Old API - NOT EXPOSED

# New API
results = tree.range_neighbors(queries, radius=2.0)
# Returns: List of numpy arrays (one per query)
```

### Clustering

```python
# Both APIs - Same interface
stats = tree.get_level_stats()
clusters = tree.get_clusters_at_level(level)

# New API also has properties
min_level = tree.min_level  # Instead of stats['min_level']
max_level = tree.max_level
```

### Tree Properties

```python
# Old API - Access via methods
stats = ct.get_level_stats()
num_points = stats['num_points']

# New API - Direct properties
num_points = tree.num_points
min_level = tree.min_level
max_level = tree.max_level
```

### Insert/Remove

```python
# Both APIs - Same interface
tree.insert(point)
tree.remove(point)
```

### Validation

```python
# Both APIs - Same interface
is_valid = tree.check_covering()
```

### Display

```python
# Old API
ct.display()  # Prints to stdout

# New API
tree.display()  # Prints to stdout
print(tree)    # Uses __repr__ for better output
# Output: <CoverTree(points=200, levels=8, min_level=2, max_level=9)>
```

## Complete Migration Example

### Before (Old API)

```python
import numpy as np
from covertree import CoverTree

# Build tree
X = np.random.randn(1000, 128)
ct = CoverTree.from_matrix(X)

# Query
queries = np.random.randn(10, 128)
nn_results = ct.NearestNeighbour(queries)
knn_results = ct.kNearestNeighbours(queries, 5)

# Clustering (via JSON hack)
import json
ct_json = ct.dumps()
ct_data = json.loads(ct_json)
# ... manual reconstruction ...

# Stats
stats = ct.get_level_stats()
print(f"Points: {stats['num_points']}")
print(f"Levels: {stats['max_level'] - stats['min_level'] + 1}")
```

### After (New API)

```python
import numpy as np
from covertree2 import CoverTree

# Build tree
X = np.random.randn(1000, 128)
tree = CoverTree.from_matrix(X)

# Query (cleaner method names)
queries = np.random.randn(10, 128)
nn_results = tree.nearest_neighbor(queries)
knn_results = tree.k_nearest_neighbors(queries, k=5)

# Clustering (native API, no JSON!)
clusters = tree.get_clusters_at_level(tree.max_level - 3)
for cluster in clusters:
    print(f"Cluster {cluster.node_id}: {len(cluster.point_ids)} points")

# Stats (properties!)
print(f"Points: {tree.num_points}")
print(f"Levels: {tree.max_level - tree.min_level + 1}")
print(tree)  # Nice repr!
```

## Performance

Both APIs have **identical performance** for core operations:
- Tree construction: Same speed
- Queries: Same speed
- Memory usage: Slightly better in new API (smart pointers)

The new API is faster for:
- ✅ Clustering (no JSON parsing)
- ✅ Property access (direct, no dictionary lookup)

## Memory Management

### Old API

```python
ct = CoverTree.from_matrix(X)  # Allocates tree
# Manual cleanup needed (relies on Python __del__)
del ct  # Calls delete on C++ pointer
```

### New API

```python
tree = CoverTree.from_matrix(X)  # Returns shared_ptr
# Automatic cleanup when Python object is garbage collected
# Multiple Python objects can reference same C++ tree safely
```

## Thread Safety

### Old API
- ⚠️ Not thread-safe for modifications
- ✅ Read-only queries can be parallel (manual)

### New API
- ⚠️ Not thread-safe for modifications (same as old)
- ✅ Read-only queries can be parallel (same as old)
- ✅ Better GIL handling with pybind11

## Backward Compatibility

The old API (`covertree`) will remain available during the transition:

### Phase 1: Both APIs Available (Current)
```python
from covertree import CoverTree   # Old API
from covertree2 import CoverTree  # New API
```

### Phase 2: New API as Default (Future)
```python
from covertree import CoverTree   # New API (pybind11)
from covertree.legacy import CoverTree  # Old API (deprecated)
```

### Phase 3: Old API Removed (Later)
```python
from covertree import CoverTree   # New API only
```

## Migration Checklist

- [ ] Install pybind11: `uv pip install pybind11`
- [ ] Test new module: `python test_pybind11.py`
- [ ] Update imports: `from covertree2 import CoverTree`
- [ ] Update method names:
  - [ ] `NearestNeighbour` → `nearest_neighbor`
  - [ ] `kNearestNeighbours` → `k_nearest_neighbors`
- [ ] Use properties where available:
  - [ ] `get_level_stats()['num_points']` → `tree.num_points`
  - [ ] `get_level_stats()['min_level']` → `tree.min_level`
  - [ ] `get_level_stats()['max_level']` → `tree.max_level`
- [ ] Remove JSON clustering hacks:
  - [ ] Replace `dumps()` + `json.loads()` with `get_clusters_at_level()`
- [ ] Test thoroughly
- [ ] Update documentation

## Common Issues

### Import Error

```
ImportError: No module named 'covertree2'
```

**Solution:**
```bash
uv pip install pybind11
uv pip install -e . --no-build-isolation --force-reinstall
```

### Type Errors

```
TypeError: incompatible function arguments
```

**Solution:** Check that you're passing numpy arrays, not lists:
```python
# Wrong
tree.nearest_neighbor([[1, 2, 3]])

# Right
tree.nearest_neighbor(np.array([[1, 2, 3]]))
```

### Both APIs Installed

If you want to test both:
```python
from covertree import CoverTree as OldTree
from covertree2 import CoverTree as NewTree

old = OldTree.from_matrix(X)
new = NewTree.from_matrix(X)
```

## Benchmarks

Run `test_pybind11.py` to verify both APIs give identical results:

```bash
python test_pybind11.py
```

You should see:
```
✓ PASS: Old and new APIs give same results
```

## Getting Help

- Check `test_pybind11.py` for usage examples
- See `IMPLEMENTATION_PLAN.md` for architecture details
- Report issues with specific error messages

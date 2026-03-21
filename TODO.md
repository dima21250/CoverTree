# TODO List

## High Priority

### Update Eigen Library
**Status:** Not started
**Priority:** Medium
**Effort:** 1-2 hours

**Current situation:**
- Using bundled Eigen in `lib/Eigen/` (older version)
- Uses deprecated `std::result_of` (removed in C++20)
- Causes ~20 compiler warnings (suppressed but still present)

**Action items:**
1. Download latest Eigen (3.4+) from https://eigen.tuxfamily.org/
2. Replace `lib/Eigen/` directory
3. Test compilation and all functionality
4. Remove warning suppression from CMakeLists.txt
5. Verify Python extension still builds

**Benefits:**
- No compiler warnings
- Better C++17/20 compatibility
- Performance improvements in newer Eigen
- Future-proof

**Commands:**
```bash
# Download latest Eigen
cd /tmp
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzf eigen-3.4.0.tar.gz

# Backup current Eigen
cd /Users/dima/Code/CoverTree
mv lib/Eigen lib/Eigen.old

# Copy new Eigen
cp -r /tmp/eigen-3.4.0/Eigen lib/

# Test build
./build.sh --clean

# If successful, remove backup
rm -rf lib/Eigen.old
```

---

## Medium Priority

### Add Comprehensive Test Suite
**Status:** Partially done (test_sanity.py, test_clustering.py exist)
**Priority:** Medium
**Effort:** 1 week

**Missing tests:**
- Memory leak tests with larger datasets
- Thread safety tests
- Performance regression tests
- Edge cases (empty trees, single point, etc.)
- Range queries
- Serialization/deserialization

**Framework:**
- Use pytest for Python tests
- Consider Google Test for C++ unit tests

---

### Update test_ensemble.py to Use New API
**Status:** Not started
**Priority:** Low (current version works with JSON)
**Effort:** 2-3 hours

**Current:**
```python
ct_json = ct.dumps()
ct_data = json.loads(ct_json)
# ... manual reconstruction
```

**Target:**
```python
stats = ct.get_level_stats()
clusters = ct.get_clusters_at_level(level)
# Done!
```

**Benefits:**
- 10-100x faster
- Cleaner code
- Type-safe
- Demonstrates new API

---

### Scikit-learn Compatible Interface
**Status:** Designed in IMPLEMENTATION_PLAN.md
**Priority:** Low
**Effort:** 1 week

From Phase 4 of the implementation plan:
- `CoverTreeClusterer` class
- `.fit()` and `.predict()` methods
- Compatible with sklearn pipelines
- Passes sklearn estimator checks

---

### pybind11 Migration
**Status:** Designed in IMPLEMENTATION_PLAN.md
**Priority:** Low
**Effort:** 2-3 weeks

Replace current Python bindings with modern pybind11:
- Type-safe bindings
- Automatic numpy ↔ Eigen conversion
- Better error messages
- Cleaner code (~50% less binding code)

See Phase 3 in IMPLEMENTATION_PLAN.md

---

## Low Priority / Future Enhancements

### Documentation Improvements
- [ ] API reference with Sphinx
- [ ] Tutorial Jupyter notebooks
- [ ] Performance benchmarks documentation
- [ ] Architecture diagrams

### CI/CD Setup
- [ ] GitHub Actions for testing
- [ ] Automatic builds for macOS/Linux
- [ ] Automated release process
- [ ] Pre-built wheels for PyPI

### Performance Optimizations
- [ ] Profile hot paths
- [ ] Consider SIMD optimizations
- [ ] GPU acceleration exploration
- [ ] Benchmark against other NN libraries

### Code Quality
- [ ] Add clang-format configuration
- [ ] Add pre-commit hooks
- [ ] Static analysis (clang-tidy)
- [ ] Documentation strings for all public APIs

---

## Completed ✅

### Phase 0: macOS + UV Foundation
- [x] UV package manager setup
- [x] macOS architecture detection
- [x] Virtual environment with UV
- [x] Dependencies installed

### Phase 1: Bug Fixes & Clustering
- [x] Fixed critical float32/64 bug
- [x] Fixed 3 FIXMEs in sgtree.cpp
- [x] Fixed 4 memory leaks in covertreecmodule.cxx
- [x] Added native clustering API in C++
- [x] Exposed clustering in Python bindings
- [x] Created test_clustering.py

### Phase 2: Modern Build System
- [x] CMakeLists.txt created
- [x] build.sh helper script
- [x] Parallel compilation working
- [x] Shared library builds
- [x] Installation structure (dist/)
- [x] Warning suppression for Eigen

---

## Notes

- Keep TODO.md updated as work progresses
- Mark items with ✅ when completed
- Add new items as they're identified
- Link to relevant documentation/plans

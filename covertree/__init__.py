# Legacy C API (for backward compatibility)
from covertree.covertree import CoverTree

# Modern pybind11 interface is available via:
#   from covertree.modern import CoverTree
# or directly:
#   import covertree2
#
# The modern interface provides:
# - Type hints
# - Better error messages
# - Pythonic properties (tree.num_points vs tree.get_level_stats()['num_points'])
# - Automatic numpy type conversion
# - ClusterInfo wrapper class

__all__ = ['CoverTree']

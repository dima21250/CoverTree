"""
Modern Python interface to CoverTree using pybind11 bindings.

This module provides a clean, Pythonic interface to the CoverTree data structure
with type hints, better documentation, and convenience methods.

Example:
    >>> import numpy as np
    >>> from covertree.modern import CoverTree
    >>>
    >>> # Build tree from data
    >>> X = np.random.randn(1000, 128)
    >>> tree = CoverTree.from_matrix(X)
    >>>
    >>> # Query nearest neighbors
    >>> queries = np.random.randn(10, 128)
    >>> neighbors = tree.nearest_neighbor(queries)
    >>>
    >>> # Get hierarchical clusters
    >>> clusters = tree.get_clusters_at_level(tree.max_level - 2)
"""

from typing import List, Dict, Any, Optional
import numpy as np
import covertree2 as _ct2


class ClusterInfo:
    """Information about a cluster in the tree.

    Attributes:
        node_id: Unique identifier for the cluster node
        level: Tree level of this cluster
        center: Cluster center point
        point_ids: List of point IDs belonging to this cluster
        covering_distance: Radius of the cluster
        distance_to_parent: Distance from this cluster to its parent
    """

    def __init__(self, cluster: _ct2.ClusterInfo):
        self._cluster = cluster

    @property
    def node_id(self) -> int:
        """Unique identifier for this cluster node."""
        return self._cluster.node_id

    @property
    def level(self) -> int:
        """Tree level of this cluster."""
        return self._cluster.level

    @property
    def center(self) -> np.ndarray:
        """Cluster center point."""
        return self._cluster.center

    @property
    def point_ids(self) -> List[int]:
        """List of point IDs belonging to this cluster."""
        return self._cluster.point_ids

    @property
    def covering_distance(self) -> float:
        """Radius of the cluster."""
        return self._cluster.covering_distance

    @property
    def distance_to_parent(self) -> float:
        """Distance from this cluster to its parent."""
        return self._cluster.distance_to_parent

    def __repr__(self) -> str:
        return (f"ClusterInfo(node_id={self.node_id}, level={self.level}, "
                f"size={len(self.point_ids)}, radius={self.covering_distance:.4f})")


class CoverTree:
    """Cover Tree for fast nearest neighbor search and hierarchical clustering.

    A cover tree is a tree-based data structure that provides efficient nearest
    neighbor search with O(log n) query time and supports hierarchical clustering
    at multiple granularities.

    Examples:
        Build and query a tree:

        >>> import numpy as np
        >>> from covertree.modern import CoverTree
        >>>
        >>> # Create tree
        >>> X = np.random.randn(1000, 128)
        >>> tree = CoverTree.from_matrix(X)
        >>>
        >>> # Nearest neighbor
        >>> queries = np.random.randn(10, 128)
        >>> nn = tree.nearest_neighbor(queries)
        >>>
        >>> # k-NN
        >>> knn = tree.k_nearest_neighbors(queries, k=5)
        >>>
        >>> # Range search
        >>> neighbors = tree.range_neighbors(queries, radius=2.0)

        Hierarchical clustering:

        >>> # Get tree statistics
        >>> print(f"Tree has {tree.num_points} points")
        >>> print(f"Levels: {tree.min_level} to {tree.max_level}")
        >>>
        >>> # Extract clusters at different granularities
        >>> coarse = tree.get_clusters_at_level(tree.max_level - 5)  # Few large clusters
        >>> fine = tree.get_clusters_at_level(tree.max_level - 1)    # Many small clusters
        >>>
        >>> # Examine a cluster
        >>> cluster = coarse[0]
        >>> print(f"Cluster has {len(cluster.point_ids)} points")
        >>> print(f"Center: {cluster.center}")
    """

    def __init__(self, tree: Optional[_ct2.CoverTree] = None):
        """Create a CoverTree instance.

        Args:
            tree: Optional pybind11 CoverTree instance. If None, creates empty tree.
        """
        self._tree = tree if tree is not None else _ct2.CoverTree()

    @classmethod
    def from_matrix(
        cls,
        X: np.ndarray,
        truncate: int = -1,
        multicore: bool = True
    ) -> 'CoverTree':
        """Build a cover tree from a data matrix.

        Args:
            X: Data matrix of shape (n_samples, n_features). Each row is a point.
            truncate: Truncation level (default: -1, no truncation).
                Higher values create shallower trees (faster but less accurate).
            multicore: Use parallel construction (default: True).

        Returns:
            A new CoverTree instance.

        Raises:
            ValueError: If X is not a 2D array or not float64.

        Examples:
            >>> X = np.random.randn(1000, 128)
            >>> tree = CoverTree.from_matrix(X)
            >>> print(tree)
            <CoverTree(points=1000, levels=...)>
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")

        # Ensure float64
        X = np.asarray(X, dtype=np.float64, order='C')

        tree = _ct2.CoverTree.from_matrix(X, truncate=truncate, multicore=multicore)
        return cls(tree)

    def nearest_neighbor(self, queries: np.ndarray) -> np.ndarray:
        """Find the nearest neighbor for each query point.

        Args:
            queries: Query points of shape (n_queries, n_features).

        Returns:
            Array of shape (n_queries, n_features) containing the nearest
            neighbor for each query point.

        Examples:
            >>> queries = np.random.randn(10, 128)
            >>> neighbors = tree.nearest_neighbor(queries)
            >>> assert neighbors.shape == (10, 128)
        """
        queries = np.asarray(queries, dtype=np.float64, order='C')
        return self._tree.nearest_neighbor(queries)

    def k_nearest_neighbors(self, queries: np.ndarray, k: int = 10) -> np.ndarray:
        """Find k nearest neighbors for each query point.

        Args:
            queries: Query points of shape (n_queries, n_features).
            k: Number of neighbors to find (default: 10).

        Returns:
            Array of shape (n_queries, k, n_features) containing k nearest
            neighbors for each query.

        Examples:
            >>> queries = np.random.randn(10, 128)
            >>> knn = tree.k_nearest_neighbors(queries, k=5)
            >>> assert knn.shape == (10, 5, 128)
        """
        queries = np.asarray(queries, dtype=np.float64, order='C')
        return self._tree.k_nearest_neighbors(queries, k=k)

    def range_neighbors(
        self,
        queries: np.ndarray,
        radius: float = 1.0
    ) -> List[np.ndarray]:
        """Find all neighbors within a given radius for each query.

        Args:
            queries: Query points of shape (n_queries, n_features).
            radius: Search radius (default: 1.0).

        Returns:
            List of numpy arrays, one per query point. Each array has shape
            (n_neighbors, n_features) where n_neighbors varies per query.

        Examples:
            >>> queries = np.random.randn(10, 128)
            >>> neighbors = tree.range_neighbors(queries, radius=2.0)
            >>> for i, nbrs in enumerate(neighbors):
            ...     print(f"Query {i}: {len(nbrs)} neighbors")
        """
        queries = np.asarray(queries, dtype=np.float64, order='C')
        return self._tree.range_neighbors(queries, radius=radius)

    def insert(self, point: np.ndarray) -> bool:
        """Insert a point into the tree.

        Args:
            point: Point to insert, shape (n_features,).

        Returns:
            True if insertion succeeded, False otherwise.
        """
        point = np.asarray(point, dtype=np.float64)
        return self._tree.insert(point)

    def remove(self, point: np.ndarray) -> bool:
        """Remove a point from the tree.

        Args:
            point: Point to remove, shape (n_features,).

        Returns:
            True if removal succeeded, False otherwise.
        """
        point = np.asarray(point, dtype=np.float64)
        return self._tree.remove(point)

    def get_level_stats(self) -> Dict[str, Any]:
        """Get tree statistics.

        Returns:
            Dictionary with keys:
                - min_level: Minimum tree level
                - max_level: Maximum tree level
                - num_points: Total number of points
                - level_counts: Dict mapping level -> node count

        Examples:
            >>> stats = tree.get_level_stats()
            >>> print(f"Points: {stats['num_points']}")
            >>> print(f"Levels: {stats['min_level']} to {stats['max_level']}")
        """
        return self._tree.get_level_stats()

    def get_clusters_at_level(self, level: int) -> List[ClusterInfo]:
        """Extract clusters at a specific tree level.

        Lower levels have fewer, larger clusters (coarse granularity).
        Higher levels have more, smaller clusters (fine granularity).

        Args:
            level: Tree level to extract clusters from. Use min_level for
                coarsest clustering, max_level for finest.

        Returns:
            List of ClusterInfo objects, one per cluster at that level.

        Examples:
            >>> # Coarse clustering (few large clusters)
            >>> coarse = tree.get_clusters_at_level(tree.min_level + 2)
            >>>
            >>> # Fine clustering (many small clusters)
            >>> fine = tree.get_clusters_at_level(tree.max_level - 1)
            >>>
            >>> # Examine clusters
            >>> for cluster in coarse:
            ...     print(f"Cluster {cluster.node_id}: {len(cluster.point_ids)} points")
        """
        clusters = self._tree.get_clusters_at_level(level)
        return [ClusterInfo(c) for c in clusters]

    def check_covering(self) -> bool:
        """Verify the tree satisfies the covering property.

        Returns:
            True if tree is valid, False otherwise.
        """
        return self._tree.check_covering()

    def display(self) -> None:
        """Display the tree structure to stdout."""
        self._tree.display()

    @property
    def min_level(self) -> int:
        """Minimum level in the tree (coarsest granularity)."""
        return self._tree.min_level

    @property
    def max_level(self) -> int:
        """Maximum level in the tree (finest granularity)."""
        return self._tree.max_level

    @property
    def num_points(self) -> int:
        """Total number of points in the tree."""
        return self._tree.num_points

    @property
    def num_levels(self) -> int:
        """Total number of levels in the tree."""
        return self.max_level - self.min_level + 1

    def __repr__(self) -> str:
        return (f"<CoverTree(points={self.num_points}, levels={self.num_levels}, "
                f"min_level={self.min_level}, max_level={self.max_level})>")

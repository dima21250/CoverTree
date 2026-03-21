import covertreec

class CoverTree(object):
    """CoverTree Class"""
    def __init__(self, this):
        self.this = this

    def __del__(self):
        covertreec.delete(self.this)

    @classmethod
    def from_matrix(cls, points):
        ptr = covertreec.new(points)
        return cls(ptr)

    def insert(self, point):
        return covertreec.insert(self.this, point)

    def remove(self, point):
        return covertreec.remove(self.this, point)

    def NearestNeighbour(self, points):
        return covertreec.NearestNeighbour(self.this, points)

    def kNearestNeighbours(self, points, k=10):
        return covertreec.kNearestNeighbours(self.this, points, k)

    def display(self):
        return covertreec.display(self.this)

    def dumps(self):
        return covertreec.dumps(self.this)

    def test_covering(self):
        return covertreec.test_covering(self.this)

    def get_level_stats(self):
        """Get tree level statistics.

        Returns:
            dict with keys:
                - min_level: minimum level in tree
                - max_level: maximum level in tree
                - num_points: total number of points
                - level_counts: dict mapping level -> node count
        """
        return covertreec.get_level_stats(self.this)

    def get_clusters_at_level(self, level):
        """Get clusters at a specific tree level.

        Args:
            level: Tree level to extract clusters from

        Returns:
            list of dicts, each containing:
                - node_id: unique node identifier
                - level: tree level
                - center: numpy array of cluster center point
                - point_ids: list of point IDs in this cluster
                - covering_distance: radius of cluster
                - distance_to_parent: distance from this node to parent
        """
        return covertreec.get_clusters_at_level(self.this, level)


/*
 * Modern pybind11 bindings for CoverTree
 * Clean, type-safe, automatic memory management
 */

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "cover_tree.h"

#include <future>
#include <thread>
#include <memory>
#include <sstream>

namespace py = pybind11;

//-----------------------------------------------------------------------------
// Helper: Convert numpy array to Eigen matrix (handles any memory layout)
//-----------------------------------------------------------------------------
Eigen::MatrixXd numpy_to_eigen(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2)
        throw std::runtime_error("Array must be 2D");

    long rows = buf.shape[0];
    long cols = buf.shape[1];

    Eigen::MatrixXd mat(rows, cols);
    double *ptr = static_cast<double*>(buf.ptr);
    long row_stride = buf.strides[0] / sizeof(double);
    long col_stride = buf.strides[1] / sizeof(double);

    for (long i = 0; i < rows; ++i) {
        for (long j = 0; j < cols; ++j) {
            mat(i, j) = ptr[i * row_stride + j * col_stride];
        }
    }

    return mat;
}

//-----------------------------------------------------------------------------
// Helper: Parallel execution for queries (same as old bindings but cleaner)
//-----------------------------------------------------------------------------
template<class UnaryFunction>
void parallel_for_each(size_t first, size_t last, UnaryFunction f)
{
    if (first >= last) return;

    unsigned cores = std::thread::hardware_concurrency();
    const size_t total_length = last - first;
    const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));

    auto task = [&f](size_t start, size_t end) {
        for (; start < end; ++start)
            f(start);
    };

    size_t chunk_start = first;
    std::vector<std::future<void>> threads;

    for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i) {
        const auto chunk_stop = chunk_start + chunk_length;
        threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
        chunk_start = chunk_stop;
    }
    threads.push_back(std::async(std::launch::async, task, chunk_start, last));

    for (auto& thread : threads)
        thread.get();
}

//-----------------------------------------------------------------------------
// Module Definition
//-----------------------------------------------------------------------------
PYBIND11_MODULE(covertree2, m) {
    m.doc() = R"pbdoc(
        Modern CoverTree Python Bindings
        ---------------------------------

        Fast nearest neighbor search with hierarchical clustering support.

        Features:
            - Thread-safe tree construction
            - Parallel nearest neighbor queries
            - Native clustering at multiple granularities
            - Automatic memory management
            - Type-safe numpy integration
    )pbdoc";

    //-------------------------------------------------------------------------
    // ClusterInfo struct
    //-------------------------------------------------------------------------
    py::class_<CoverTree::ClusterInfo>(m, "ClusterInfo", R"pbdoc(
        Information about a cluster in the tree.

        Attributes:
            node_id: Unique identifier for the cluster node
            level: Tree level of this cluster
            center: Numpy array representing cluster center point
            point_ids: List of point IDs belonging to this cluster
            covering_distance: Radius of the cluster
            distance_to_parent: Distance from this cluster to its parent
    )pbdoc")
        .def_readonly("node_id", &CoverTree::ClusterInfo::node_id)
        .def_readonly("level", &CoverTree::ClusterInfo::level)
        .def_readonly("center", &CoverTree::ClusterInfo::center)
        .def_readonly("point_ids", &CoverTree::ClusterInfo::point_ids)
        .def_readonly("covering_distance", &CoverTree::ClusterInfo::covering_distance)
        .def_readonly("distance_to_parent", &CoverTree::ClusterInfo::distance_to_parent)
        .def("__repr__", [](const CoverTree::ClusterInfo& c) {
            std::ostringstream oss;
            oss << "<ClusterInfo(node=" << c.node_id
                << ", level=" << c.level
                << ", size=" << c.point_ids.size() << ")>";
            return oss.str();
        });

    //-------------------------------------------------------------------------
    // Main CoverTree class
    //-------------------------------------------------------------------------
    py::class_<CoverTree, std::shared_ptr<CoverTree>>(m, "CoverTree", R"pbdoc(
        Cover Tree data structure for fast nearest neighbor search.

        A cover tree is a tree-based data structure that provides efficient
        nearest neighbor search with theoretical guarantees.

        Example:
            >>> import numpy as np
            >>> from covertree2 import CoverTree
            >>>
            >>> # Create tree from data
            >>> X = np.random.randn(1000, 128)
            >>> tree = CoverTree.from_matrix(X)
            >>>
            >>> # Query nearest neighbors
            >>> queries = np.random.randn(10, 128)
            >>> results = tree.nearest_neighbor(queries)
            >>>
            >>> # Get hierarchical clusters
            >>> stats = tree.get_level_stats()
            >>> clusters = tree.get_clusters_at_level(stats['max_level'] - 2)
    )pbdoc")

        //---------------------------------------------------------------------
        // Construction
        //---------------------------------------------------------------------
        .def(py::init<>(), "Create an empty cover tree")

        .def_static("from_matrix",
            [](py::array_t<double> matrix, int truncate, bool multicore) {
                // Convert numpy array (rows=points, cols=dims) to Eigen matrix
                Eigen::MatrixXd mat = numpy_to_eigen(matrix);

                // Transpose: (rows=points, cols=dims) -> (cols=points, rows=dims)
                Eigen::MatrixXd transposed = mat.transpose();
                return std::shared_ptr<CoverTree>(
                    CoverTree::from_matrix(transposed, truncate, multicore)
                );
            },
            py::arg("matrix"),
            py::arg("truncate") = -1,
            py::arg("multicore") = true,
            R"pbdoc(
                Build a cover tree from a numpy matrix.

                Args:
                    matrix: Numpy array of shape (n_samples, n_features)
                    truncate: Truncation level (default: -1, no truncation)
                    multicore: Use parallel construction (default: True)

                Returns:
                    A new CoverTree instance
            )pbdoc")

        //---------------------------------------------------------------------
        // Nearest Neighbor Queries
        //---------------------------------------------------------------------
        .def("nearest_neighbor",
            [](std::shared_ptr<CoverTree> self, py::array_t<double> query) {
                // Convert and transpose query
                Eigen::MatrixXd queryMat = numpy_to_eigen(query);
                long numPoints = queryMat.rows();
                long numDims = queryMat.cols();

                Eigen::MatrixXd queryT = queryMat.transpose();
                Eigen::MatrixXd results(numPoints, numDims);

                parallel_for_each(0L, numPoints, [&](long i) {
                    auto nn = self->NearestNeighbour(queryT.col(i));
                    results.row(i) = nn.first->_p.transpose();
                });

                return results;
            },
            py::arg("query"),
            R"pbdoc(
                Find the nearest neighbor for each query point.

                Args:
                    query: Numpy array of shape (n_queries, n_features)

                Returns:
                    Numpy array of shape (n_queries, n_features) containing
                    the nearest neighbor for each query point
            )pbdoc")

        .def("k_nearest_neighbors",
            [](std::shared_ptr<CoverTree> self, py::array_t<double> query, int k) {
                // Convert and transpose query
                Eigen::MatrixXd queryMat = numpy_to_eigen(query);
                long numPoints = queryMat.rows();
                long numDims = queryMat.cols();

                Eigen::MatrixXd queryT = queryMat.transpose();

                // Result tensor: (n_queries, k, n_features)
                std::vector<std::vector<Eigen::VectorXd>> results(numPoints);

                parallel_for_each(0L, numPoints, [&](long i) {
                    auto knn = self->kNearestNeighbours(queryT.col(i), k);
                    results[i].reserve(k);
                    for (const auto& neighbor : knn) {
                        results[i].push_back(neighbor.first->_p);
                    }
                });

                // Convert to 3D numpy array
                std::vector<ssize_t> shape = {numPoints, k, numDims};
                py::array_t<double> output(shape);
                auto r = output.mutable_unchecked<3>();

                for (long i = 0; i < numPoints; ++i) {
                    for (int j = 0; j < k; ++j) {
                        for (long d = 0; d < numDims; ++d) {
                            r(i, j, d) = results[i][j][d];
                        }
                    }
                }

                return output;
            },
            py::arg("query"),
            py::arg("k") = 10,
            R"pbdoc(
                Find k nearest neighbors for each query point.

                Args:
                    query: Numpy array of shape (n_queries, n_features)
                    k: Number of neighbors to find (default: 10)

                Returns:
                    Numpy array of shape (n_queries, k, n_features)
            )pbdoc")

        .def("range_neighbors",
            [](std::shared_ptr<CoverTree> self, py::array_t<double> query, double radius) {
                // Convert and transpose query
                Eigen::MatrixXd queryMat = numpy_to_eigen(query);
                long numPoints = queryMat.rows();
                long numDims = queryMat.cols();

                Eigen::MatrixXd queryT = queryMat.transpose();

                std::vector<std::vector<Eigen::VectorXd>> results(numPoints);

                parallel_for_each(0L, numPoints, [&](long i) {
                    auto neighbors = self->rangeNeighbours(queryT.col(i), radius);
                    results[i].reserve(neighbors.size());
                    for (const auto& neighbor : neighbors) {
                        results[i].push_back(neighbor.first->_p);
                    }
                });

                // Convert to list of numpy arrays
                py::list output;
                for (const auto& neighbors : results) {
                    if (neighbors.empty()) {
                        std::vector<ssize_t> empty_shape = {0, numDims};
                        output.append(py::array_t<double>(empty_shape));
                    } else {
                        long n = neighbors.size();
                        long d = neighbors[0].size();
                        std::vector<ssize_t> arr_shape = {n, d};
                        py::array_t<double> arr(arr_shape);
                        auto r = arr.mutable_unchecked<2>();
                        for (long i = 0; i < n; ++i) {
                            for (long j = 0; j < d; ++j) {
                                r(i, j) = neighbors[i][j];
                            }
                        }
                        output.append(arr);
                    }
                }

                return output;
            },
            py::arg("query"),
            py::arg("radius") = 1.0,
            R"pbdoc(
                Find all neighbors within a given radius.

                Args:
                    query: Numpy array of shape (n_queries, n_features)
                    radius: Search radius (default: 1.0)

                Returns:
                    List of numpy arrays, one per query point.
                    Each array has shape (n_neighbors, n_features)
            )pbdoc")

        //---------------------------------------------------------------------
        // Modifications
        //---------------------------------------------------------------------
        .def("insert",
            [](std::shared_ptr<CoverTree> self, Eigen::Ref<Eigen::VectorXd> point) {
                return self->insert(point);
            },
            py::arg("point"),
            "Insert a point into the tree")

        .def("remove",
            [](std::shared_ptr<CoverTree> self, Eigen::Ref<Eigen::VectorXd> point) {
                return self->remove(point);
            },
            py::arg("point"),
            "Remove a point from the tree")

        //---------------------------------------------------------------------
        // Clustering API
        //---------------------------------------------------------------------
        .def("get_level_stats",
            [](std::shared_ptr<CoverTree> self) {
                py::dict stats;
                stats["min_level"] = self->getMinLevel();
                stats["max_level"] = self->getMaxLevel();
                stats["num_points"] = self->count_points();

                // Convert level counts to dict
                auto level_counts = self->getLevelCounts();
                py::dict counts;
                for (const auto& pair : level_counts) {
                    counts[py::int_(pair.first)] = pair.second;
                }
                stats["level_counts"] = counts;

                return stats;
            },
            R"pbdoc(
                Get tree statistics.

                Returns:
                    Dictionary with keys:
                        - min_level: Minimum tree level
                        - max_level: Maximum tree level
                        - num_points: Total number of points
                        - level_counts: Dict mapping level -> node count
            )pbdoc")

        .def("get_clusters_at_level",
            &CoverTree::getClustersAtLevel,
            py::arg("level"),
            R"pbdoc(
                Extract clusters at a specific tree level.

                Args:
                    level: Tree level to extract clusters from

                Returns:
                    List of ClusterInfo objects
            )pbdoc")

        //---------------------------------------------------------------------
        // Properties
        //---------------------------------------------------------------------
        .def_property_readonly("min_level", &CoverTree::getMinLevel,
            "Minimum level in the tree")

        .def_property_readonly("max_level", &CoverTree::getMaxLevel,
            "Maximum level in the tree")

        .def_property_readonly("num_points",
            [](const CoverTree& self) { return self.count_points(); },
            "Total number of points in the tree")

        //---------------------------------------------------------------------
        // Validation
        //---------------------------------------------------------------------
        .def("check_covering", &CoverTree::check_covering,
            "Verify the tree satisfies the covering property")

        .def("display",
            [](const CoverTree& self) {
                std::ostringstream oss;
                oss << self;
                py::print(oss.str());
            },
            "Display the tree structure")

        //---------------------------------------------------------------------
        // String representation
        //---------------------------------------------------------------------
        .def("__repr__",
            [](const CoverTree& tree) {
                std::ostringstream oss;
                oss << "<CoverTree(points=" << tree.count_points()
                    << ", levels=" << (tree.getMaxLevel() - tree.getMinLevel() + 1)
                    << ", min_level=" << tree.getMinLevel()
                    << ", max_level=" << tree.getMaxLevel() << ")>";
                return oss.str();
            });

    //-------------------------------------------------------------------------
    // Module-level information
    //-------------------------------------------------------------------------
    m.attr("__version__") = "2.0.0";
    m.attr("__author__") = "Manzil Zaheer";
}

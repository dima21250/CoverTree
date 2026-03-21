/*
 * Copyright (c) 2017 Manzil Zaheer All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "numpy/arrayobject.h"
#include "cover_tree.h"

#include <future>
#include <thread>

#include <iostream>
#include <iomanip>
#include <sstream>

// Always use FLOAT64 since code uses 'double' everywhere
#define MY_NPY_FLOAT NPY_FLOAT64

template<class UnaryFunction>
UnaryFunction parallel_for_each(size_t first, size_t last, UnaryFunction f)
{
    if (first >= last) {
        return f;
    }

  unsigned cores = std::thread::hardware_concurrency();
  //std::cout << "Number of cores: " << cores << std::endl;

  auto task = [&f](size_t start, size_t end)->void{
    for (; start < end; ++start)
      f(start);
  };

  const size_t total_length = last - first;
  const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));
  size_t chunk_start = first;
  std::vector<std::future<void>>  for_threads;
  for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
  {
    const auto chunk_stop = chunk_start + chunk_length;
    for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
    chunk_start = chunk_stop;
  }
  for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

  for (auto& thread : for_threads)
    thread.get();
  return f;
}

static inline void progressbar(unsigned int x, unsigned int n, unsigned int w = 50){
    if ( (x != n) && (x % (n/10+1) != 0) ) return;

    float ratio =  x/(float)n;
    unsigned c = ratio * w;

    std::cout << std::setw(3) << (int)(ratio*100) << "% [";
    for (unsigned x=0; x<c; x++) std::cout << "=";
    for (unsigned x=c; x<w; x++) std::cout << " ";
    std::cout << "]\r" << std::flush;
}

template<class UnaryFunction>
UnaryFunction parallel_for_progressbar(size_t first, size_t last, UnaryFunction f)
{
    if (first >= last) {
        return f;
    }

    unsigned cores = std::thread::hardware_concurrency();
    const size_t total_length = last - first;
    const size_t chunk_length = std::max(size_t(total_length / cores), size_t(1));

    auto task = [&f,&chunk_length](size_t start, size_t end)->void{
        for (; start < end; ++start){
            progressbar(start%chunk_length, chunk_length);
            f(start);
        }
    };

    size_t chunk_start = first;
    std::vector<std::future<void>>  for_threads;
    for (unsigned i = 0; i < (cores - 1) && i < total_length; ++i)
    {
        const auto chunk_stop = chunk_start + chunk_length;
        for_threads.push_back(std::async(std::launch::async, task, chunk_start, chunk_stop));
        chunk_start = chunk_stop;
    }
    for_threads.push_back(std::async(std::launch::async, task, chunk_start, last));

    for (auto& thread : for_threads)
        thread.get();
    progressbar(chunk_length, chunk_length);
    std::cout << std::endl;
    return f;
}


static PyObject *CovertreecError;

static PyObject *new_covertreec(PyObject *self, PyObject *args)
{
  PyArrayObject *in_array;

  if (!PyArg_ParseTuple(args,"O!:new_covertreec", &PyArray_Type, &in_array))
    return NULL;

  long numPoints = PyArray_DIM(in_array, 0);
  long numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  long idx[2] = {0, 0};
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> pointMatrix(fnp, numDims, numPoints);

  CoverTree* cTree = CoverTree::from_matrix(pointMatrix, -1, false);
  size_t int_ptr = reinterpret_cast< size_t >(cTree);

  return Py_BuildValue("k", int_ptr);
}

static PyObject *delete_covertreec(PyObject *self, PyObject *args)
{
  CoverTree *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args,"k:delete_covertreec", &int_ptr))
    return NULL;

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  delete obj;

  return Py_BuildValue("k", int_ptr);
}


static PyObject *covertreec_insert(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "kO!:covertreec_insert", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  int d = PyArray_NDIM(in_array);
  std::vector<npy_intp> idx(d, 0);  // RAII - no memory leak
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx.data()) );
  Eigen::Map<pointType> value(fnp, PyArray_SIZE(in_array));

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  obj->insert(value);

  Py_RETURN_NONE;
}

static PyObject *covertreec_remove(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "kO!:covertreec_insert", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  int d = PyArray_NDIM(in_array);
  std::vector<npy_intp> idx(d, 0);  // RAII - no memory leak
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx.data()) );
  Eigen::Map<pointType> value(fnp, PyArray_SIZE(in_array));

  obj = reinterpret_cast< CoverTree * >(int_ptr);
  bool val = obj->remove(value);

  if (val)
  	Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}

static PyObject *covertreec_nn(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "kO!:covertreec_nn", &int_ptr, &PyArray_Type, &in_array))
    return NULL;

  long idx[2] = {0,0};
  long numPoints = PyArray_DIM(in_array, 0);
  long numDims = PyArray_DIM(in_array, 1);
  //std::cout<<numPoints<<", "<<numDims<<std::endl;
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  //obj->dist_count.clear();

  // Let numpy allocate and own the memory - no memory leak
  npy_intp dims[2] = {numPoints, numDims};
  PyObject *out_array = PyArray_SimpleNew(2, dims, MY_NPY_FLOAT);
  if (!out_array) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
    return NULL;
  }

  double *results = static_cast<double*>(PyArray_DATA((PyArrayObject*)out_array));

  parallel_for_progressbar(0L, numPoints, [&](long i)->void{
    std::pair<CoverTree::Node*, double> ct_nn = obj->NearestNeighbour(queryPts.col(i));
    double *data = ct_nn.first->_p.data();
    long offset = i*numDims;
    for(long j=0; j<numDims; ++j)
      results[offset++] = data[j];
  });

  return out_array;
}

static PyObject *covertreec_knn(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  PyArrayObject *in_array;
  long k;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "kO!l:covertreec_knn", &int_ptr, &PyArray_Type, &in_array, &k))
    return NULL;

  long idx[2] = {0,0};
  long numPoints = PyArray_DIM(in_array, 0);
  long numDims = PyArray_DIM(in_array, 1);
  double * fnp = reinterpret_cast< double * >( PyArray_GetPtr(in_array, idx) );
  Eigen::Map<Eigen::MatrixXd> queryPts(fnp, numDims, numPoints);

  obj = reinterpret_cast< CoverTree * >(int_ptr);

  // Let numpy allocate and own the memory - no memory leak
  npy_intp dims[3] = {numPoints, k, numDims};
  PyObject *out_array = PyArray_SimpleNew(3, dims, MY_NPY_FLOAT);
  if (!out_array) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate output array");
    return NULL;
  }

  double *results = static_cast<double*>(PyArray_DATA((PyArrayObject*)out_array));

  parallel_for_progressbar(0L, numPoints, [&](long i)->void{
    std::vector<std::pair<CoverTree::Node*, double>> ct_nn = obj->kNearestNeighbours(queryPts.col(i), k);
    long offset = k*numDims*i;
    for(long t=0; t<k; ++t)
    {
      double *data = ct_nn[t].first->_p.data();
      for(long j=0; j<numDims; ++j)
        results[offset++] = data[j];
    }
  });

  return out_array;
}


static PyObject *covertreec_display(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "k:covertreec_display", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< CoverTree * >(int_ptr);
  std::cout << *obj;

  Py_RETURN_NONE;
}


static PyObject *covertreec_dumps(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;
  std::stringstream os;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "k:covertreec_dumps", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< CoverTree * >(int_ptr);
  os << *obj;

  static const std::string tmp = os.str();
  const char *s = tmp.c_str();

  return Py_BuildValue("s", s);
}


static PyObject *covertreec_test_covering(PyObject *self, PyObject *args) {

  CoverTree *obj;
  size_t int_ptr;

  /*  parse the input, from python int to c++ int */
  if (!PyArg_ParseTuple(args, "k:covertreec_test_covering", &int_ptr))
    return NULL;
  
  obj = reinterpret_cast< CoverTree * >(int_ptr);
  if(obj->check_covering())
    Py_RETURN_TRUE;

  Py_RETURN_FALSE;
}


/******************************************* Clustering API ***************************************************/

static PyObject *covertreec_get_level_stats(PyObject *self, PyObject *args) {
  CoverTree *obj;
  size_t int_ptr;

  if (!PyArg_ParseTuple(args, "k:covertreec_get_level_stats", &int_ptr))
    return NULL;

  obj = reinterpret_cast<CoverTree*>(int_ptr);

  PyObject* dict = PyDict_New();
  PyDict_SetItemString(dict, "min_level", PyLong_FromLong(obj->getMinLevel()));
  PyDict_SetItemString(dict, "max_level", PyLong_FromLong(obj->getMaxLevel()));
  PyDict_SetItemString(dict, "num_points", PyLong_FromLong(obj->count_points()));

  // Add level counts
  std::map<int, unsigned> level_counts = obj->getLevelCounts();
  PyObject* counts_dict = PyDict_New();
  for (const auto& pair : level_counts) {
    PyDict_SetItem(counts_dict, PyLong_FromLong(pair.first), PyLong_FromLong(pair.second));
  }
  PyDict_SetItemString(dict, "level_counts", counts_dict);

  return dict;
}

static PyObject *covertreec_get_clusters_at_level(PyObject *self, PyObject *args) {
  CoverTree *obj;
  size_t int_ptr;
  int level;

  if (!PyArg_ParseTuple(args, "ki:covertreec_get_clusters_at_level", &int_ptr, &level))
    return NULL;

  obj = reinterpret_cast<CoverTree*>(int_ptr);

  std::vector<CoverTree::ClusterInfo> clusters = obj->getClustersAtLevel(level);

  // Build Python list of cluster dictionaries
  PyObject* result_list = PyList_New(clusters.size());

  for (size_t i = 0; i < clusters.size(); ++i) {
    const auto& cluster = clusters[i];
    PyObject* cluster_dict = PyDict_New();

    // Add cluster properties
    PyDict_SetItemString(cluster_dict, "node_id", PyLong_FromLong(cluster.node_id));
    PyDict_SetItemString(cluster_dict, "level", PyLong_FromLong(cluster.level));
    PyDict_SetItemString(cluster_dict, "covering_distance", PyFloat_FromDouble(cluster.covering_distance));
    PyDict_SetItemString(cluster_dict, "distance_to_parent", PyFloat_FromDouble(cluster.distance_to_parent));

    // Convert center point to numpy array
    npy_intp center_dims[1] = {static_cast<npy_intp>(cluster.center.size())};
    PyObject* center_array = PyArray_SimpleNew(1, center_dims, NPY_FLOAT64);
    double* center_data = static_cast<double*>(PyArray_DATA((PyArrayObject*)center_array));
    for (int j = 0; j < cluster.center.size(); ++j) {
      center_data[j] = cluster.center[j];
    }
    PyDict_SetItemString(cluster_dict, "center", center_array);

    // Convert point IDs to list
    PyObject* ids_list = PyList_New(cluster.point_ids.size());
    for (size_t j = 0; j < cluster.point_ids.size(); ++j) {
      PyList_SetItem(ids_list, j, PyLong_FromLong(cluster.point_ids[j]));
    }
    PyDict_SetItemString(cluster_dict, "point_ids", ids_list);

    PyList_SetItem(result_list, i, cluster_dict);
  }

  return result_list;
}

PyMODINIT_FUNC PyInit_covertreec(void)
{
  PyObject *m;
  static PyMethodDef CovertreecMethods[] = {
    {"new", new_covertreec, METH_VARARGS, "Initialize a new Cover Tree."},
    {"delete", delete_covertreec, METH_VARARGS, "Delete the Cover Tree."},
    {"insert", covertreec_insert, METH_VARARGS, "Insert a point to the Cover Tree."},
    {"remove", covertreec_remove, METH_VARARGS, "Remove a point from the Cover Tree."},
    {"NearestNeighbour", covertreec_nn, METH_VARARGS, "Find the nearest neighbour."},
    {"kNearestNeighbours", covertreec_knn, METH_VARARGS, "Find the k nearest neighbours."},
    {"display", covertreec_display, METH_VARARGS, "Display the Cover Tree."},
    {"dumps", covertreec_dumps, METH_VARARGS, "Dump the Cover Tree into a Python string."},
    {"test_covering", covertreec_test_covering, METH_VARARGS, "Check if covering property is satisfied."},
    {"get_level_stats", covertreec_get_level_stats, METH_VARARGS, "Get tree level statistics."},
    {"get_clusters_at_level", covertreec_get_clusters_at_level, METH_VARARGS, "Get clusters at a specific level."},
    {NULL, NULL, 0, NULL}
  };
  static struct PyModuleDef mdef = {PyModuleDef_HEAD_INIT,
                                 "covertreec",
                                 "Example module that creates an extension type.",
                                 -1,
                                 CovertreecMethods};
  m = PyModule_Create(&mdef);
  if ( m == NULL )
    return NULL;

  /* IMPORTANT: this must be called */
  import_array();

  CovertreecError = PyErr_NewException("covertreec.error", NULL, NULL);
  Py_INCREF(CovertreecError);
  PyModule_AddObject(m, "error", CovertreecError);

  return m;
}

int main(int argc, char *argv[])
{
  /* Convert to wchar */
  wchar_t *program = Py_DecodeLocale(argv[0], NULL);
  if (program == NULL) {
     std::cerr << "Fatal error: cannot decode argv[0]" << std::endl;
     return 1;
  }
  
  /* Add a built-in module, before Py_Initialize */
  //PyImport_AppendInittab("covertreec", PyInit_covertreec);
    
  /* Pass argv[0] to the Python interpreter */
  Py_SetProgramName(program);

  /* Initialize the Python interpreter.  Required. */
  Py_Initialize();

  /* Add a static module */
  //PyInit_covertreec();
}


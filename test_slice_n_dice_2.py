
import pandas as pd
import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import torch
from covertree import CoverTree
import numpy as np
from collections import defaultdict
from statistics import mean, median, quantiles
import hashlib

def hash_vec(v, places=1):
    fmt = "{{:+.{}E}}".format(places)
    vec = np.array(v).astype('float64')
    vec = np.round(vec, decimals=places)
    vec_string = "".join([fmt.format(float(x)) for x in vec]).encode("utf-8")
    return hashlib.sha256(vec_string).hexdigest()

def enum_hash_vecs(arr, places=1):
    hashed_vecs = [hash_vec(v, places=places) for v in arr]
    enumd_hashes = list(enumerate(hashed_vecs))
    return dict(enumd_hashes)

def map_hashed_vecs_to_enums(enumd_hashed_vecs):
    hashed_vecs_to_enum = defaultdict(set)
    for i,hv in enumd_hashed_vecs.items():
        hashed_vecs_to_enum[hv].add(i)
    return hashed_vecs_to_enum

def gen_test_data(N=100):
    '''Generate an N by N square array for test data to sanity check cover tree implementation.'''
    row = [float(i) for i in range(N)]
    arr = np.array(row).reshape(1,N)
    for i in range(1,N):
        row.append(row.pop(0))
        r = np.array(row).reshape(1,N)
        arr = np.append(arr, r, axis=0)
    return arr

def gen_test_data_dup(N=100, duplication=3):
    '''Generate an duplication*N by N array to sanity check cover tree implemenation.'''
    arr = gen_test_data(N)
    for i in range(1,duplication):
        arr = np.append(arr, gen_test_data(N), axis=0)
    return arr

args = sys.argv
largs = len(args)
if largs != 2:
    print("Error: Wrong number of arguments: {}".format(largs), file=sys.stderr)
    sys.exit(-1)

try:
    with open(args[1]) as fargs:
        config = json.load(fargs)
except Exception as e:
    print("Error while reading arguments: {}".format(e), file=sys.stderr)
    sys.exit(-1)

try:
    # Input CSV file
    in_file = Path(config["in_file"])

except Exception as e:
    print("Configuration file error: {}".format(e), file=sys.stderr)
    sys.exit(-1)

df = pd.read_csv(in_file)
df_t = df.iloc[:,3:].transpose() # d in m space
#df_t = df.iloc[:,3:] # m in d space
#df_t = df.iloc[:,2:] # d in X
embeddings = df_t.to_numpy().copy() # copy seems necessary to prevent issues with pandas dataframe
#embeddings = gen_test_data()
#embeddings = gen_test_data_dup()

# Set up the mappings between row numbers and hashed vectors and do a sanity check
e2hv = enum_hash_vecs(embeddings)
hv2e = map_hashed_vecs_to_enums(e2hv)
for i in range(len(embeddings)):
    assert(i in hv2e[e2hv[i]])

# CoverTree will core dump if array contains float32
emb = np.float64(embeddings)
ct = CoverTree.from_matrix(emb)

# Making sure that the rows in emb and embeddings hash to the same values
for left, right in zip(enum_hash_vecs(emb).values(), enum_hash_vecs(embeddings).values()):
    assert(left == right)

# Map back to indexes in original emb array
e2i = {tuple(a):i for i,a in enumerate([e for e in emb])}

# Faiss wants float32 and c-contiguous arrays
embeddings = np.float32(embeddings.copy(order="C"))
d = embeddings.shape[1]

#gpu_id = torch.cuda.current_device()
gpu_id = False # fix environment later

if gpu_id:
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id
    flat_config = [cfg]
    resources = [faiss.StandardGpuResources()]
    index = faiss.GpuIndexFlatIP(resources[0], d, flat_config[0])
else:
    index = faiss.IndexFlatL2(d)

index.add(embeddings)

# Dump cover tree into JSON and create Python data structures
ct_json = ct.dumps()
ct_data = json.loads(ct_json)

node_data = [{"id":n["id"],
               "level":n["level"],
               "covdist":n["covdist"],
               "sepdist":n["sepdist"],
               "rootdist":n["rootdist"]} for n in ct_data["nodes"]]

node_points = {n["id"]:n["point"] for n in ct_data["nodes"]}

n2hv = {id:hash_vec(v) for (id,v) in node_points.items()} 
assert(set(n2hv.values()) == set(e2hv.values()))

def node2enums(id, np=node_points, hv2e=hv2e):
    return hv2e[hash_vec(np[id])]

# Tree level given node
node_level = {n["id"]:n["level"] for n in node_data}

# Nodes associated with each tree level
level_nodes = defaultdict(set)
for k,v in node_level.items():
    level_nodes[v].add(k)

# Number of nodes associated with each tree level
level_count = {k:len(v) for (k,v) in level_nodes.items()}

# Number of children associated with each node
children = [(ln["parent"], ln["child"]) for ln in ct_data["links"]] 
node_children = defaultdict(set)
for p,c in children:
    node_children[p].add(c)

root_dist = [(n['id'], n['level'], n['rootdist']) for n in node_data]
root_dist_mean = mean([n[2] for n in root_dist])
root_dist_median = median([n[2] for n in root_dist])
root_dist_sorted = sorted(root_dist, key=lambda n:n[2], reverse=True)

def k_outliers(k):
    return {n for (n,_,_) in root_dist_sorted[0:k]}

# map node id to index point query
id2q = {id: np.reshape(np.float32(p), (1,d)) for (id,p) in node_points.items()}

# map map node ID to original dataframe rows
def id2rows(id, k=5, t=1):
    if not id in id2q:
        return set()
    else:
        D,I = index.search(id2q[id], k)
        n = I.tolist()[0]
        d = D.tolist()[0]
        rows = []
        for i in range(len(d)):
            if d[i] < t:
                rows.append(n[i])
        return set(rows)

# map original dataframe row to item name
row2name = {item[0]:item[1] for item in enumerate(list(df_t.transpose().columns))}

def id2names(id,k):
    return {row2name[r] for r in id2rows(id,k)} 

def dump_id_names(k=5, t=1):
    for id in id2q.keys():
        for r in id2rows(id,k,t):
            print(id,row2name[r])

def parent_child_graph(fname):
    head = """
digraph {
        layout=twopi
        overlap=prism
        overlap_scaling=35
    """

    A = 3

    with open(fname, "w") as fout:
        top_B = k_outliers(2*A)
        top_A = k_outliers(A)
        next_A = top_B - top_A

        print(head, file=fout)
        print("\t{} [style=filled, fillcolor=\"blue\"]".format(0), file=fout)

        for outlier in top_A:
            print("\t{} [style=filled, fillcolor=\"red\"]".format(outlier), file=fout)

        for outlier in next_A:
            print("\t{} [style=filled, fillcolor=\"purple\"]".format(outlier), file=fout)

        for (p,c) in children:
            print("\t{} -> {}".format(p,c), file=fout)

        print("}", file=fout)

def parent_child_graph_2():
    head = """
digraph {
        layout=twopi
        overlap=prism
        overlap_scaling=35
    """
    print(head)
    for (p,c) in children:
        print("\t{} -> {}".format(p,c))
    print("}")



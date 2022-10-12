
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


def hash_vec(v, places=3):
    fmt = "{{:+.{}E}}".format(places)
    vec_string = "".join([fmt.format(x) for x in v]).encode("utf-8")
    return hashlib.sha256(vec_string).hexdigest()

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
#df_t = df.iloc[:,3:].transpose() # d in m space
#df_t = df.iloc[:,3:] # m in d space
df_t = df.iloc[:,2:] # d in X
embeddings = df_t.to_numpy()


# CoverTree will core dump if array contains float32
emb = np.float64(embeddings)
ct = CoverTree.from_matrix(emb)

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

    with open(fname, "w") as fout:
        top_50 = k_outliers(50)
        top_25 = k_outliers(25)
        next_25 = top_50 - top_25

        print(head, file=fout)
        print("\t{} [style=filled, fillcolor=\"blue\"]".format(0), file=fout)

        for outlier in top_25:
            print("\t{} [style=filled, fillcolor=\"red\"]".format(outlier), file=fout)

        for outlier in next_25:
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


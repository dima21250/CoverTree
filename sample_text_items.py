
import pandas as pd
import json
import sys
import faiss
import torch
import numpy as np
import random
import hashlib

from pathlib import Path
from sentence_transformers import SentenceTransformer
from covertree import CoverTree
from collections import defaultdict
from statistics import mean, median, quantiles

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

    # Field in input CSV file containing text
    text_field = config["text_field"]

    # Sample size
    sample_size = config["sample_size"]

    # Random seed
    random_seed = config["random_seed"]

    # Output CSV file
    out_file = Path(config["out_file"])

    # JSON dump of cover tree
    fdumps = None
    if "ct_file" in config:
        fdumps = config["ct_file"]

except Exception as e:
    print("Configuration file error: {}".format(e), file=sys.stderr)
    sys.exit(-1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Want to be able to specify a single random seed and have get_prototypes have a seed parameter.
# That is I want to be able to resuse the random seed without quite reusing it.
hashed_seed = hashlib.sha256(str(random_seed).encode()).hexdigest()
random.seed(hashed_seed)

df = pd.read_csv(in_file).sample(frac=1).reset_index(drop=True)
#df = df[df[text_field].notna()]
text = df[text_field]
text = text.fillna("")

embeddings = model.encode(text)
faiss.normalize_L2(embeddings)
d = embeddings.shape[1]

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
    index = faiss.GpuIndexFlatL2(resources[0], d, flat_config[0])
else:
    index = faiss.IndexFlatL2(d)

index.add(embeddings)

# Dump cover tree into JSON and create Python data structures
ct_json = ct.dumps()
ct_data = json.loads(ct_json)

if fdumps:
    fres = Path(fdumps).expanduser().resolve()
    with open(fres, "w") as fd:
        print(ct_json, file=fd)

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
            if d[i] <= t:
                rows.append(n[i])
        return set(rows)

def id2row(id, t=1):
    """
    Returns None if point does not exist in id2q or if the closest point
    in dataset is farther than t away from point in index.
    """
    if not id in id2q:
        return None
    else:
        D,I = index.search(id2q[id], 1)
        n = I.tolist()[0]
        d = D.tolist()[0]
        row = None
        for i in range(len(d)):
            if d[0] <= t:
                row = n[0]
        return row

def get_prototypes(seed):
    row_ids = set()
    remainder = sample_size
    for lv in sorted(level_count, reverse=True):
        if remainder >= level_count[lv]:
            for n in level_nodes[lv]:
                row_id = id2row(n)
                if row_id != None:
                    row_ids.add(row_id)
                    remainder -= 1
                else:
                    print("Warning: ID {} doesn't have a corresponding row.".format(n), file=sys.stderr)
        else:
            assert(remainder < level_count[lv])
            random.seed(seed)
            node_sample = random.sample(list(level_nodes[lv]), remainder)
            for n in node_sample:
                row_id = id2row(n)
                if row_id != None:
                    row_ids.add(row_id)
                    remainder -= 1
                else:
                    print("Warning: ID {} doesn't have a corresponding row.".format(n), file=sys.stderr)
                    print("Warning: final sample size = {}".format(len(row_ids)))

    sample_ids = sorted(row_ids)
    df_sample = df.loc[sample_ids]
    return df_sample

df_prototypes = get_prototypes(random_seed)
df_prototypes.to_csv(out_file, encoding='utf-8', index=False)


import pandas as pd
import json
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
from covertree import CoverTree
import numpy as np
from collections import defaultdict

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


except Exception as e:
    print("Configuration file error: {}".format(e), file=sys.stderr)
    sys.exit(-1)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
df = pd.read_csv(in_file)
text = df[text_field]
embeddings = model.encode(text)
faiss.normalize_L2(embeddings)
d = embeddings.shape[1]

# CoverTree will core dump if array contains float32
embeddings = np.float64(embeddings)
ct = CoverTree.from_matrix(embeddings)

# Dump cover tree into JSON and create Python data structures
ct_json = ct.dumps()
ct_data = json.loads(ct_json)

node_data = [{"id":n["id"], "level":n["level"]} for n in ct_data["nodes"]]

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


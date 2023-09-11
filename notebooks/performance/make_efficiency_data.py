# pylint: disable=missing-module-docstring,missing-function-docstring
# pylint: disable=import-error,redefined-outer-name
from pathlib import Path
from time import time
import pickle
import numpy as np
import pandas as pd
import igraph as ig
from tqdm.auto import tqdm
from msb import Balance

# Paths
HERE = Path(__file__ ).parent
ROOT = HERE.parent.parent
DATA = ROOT/"data"

# Main parameters
M = np.array([1, 2, 4, 8, 16])
METHODS = ("balance", "k_balance", "node_balance")

Epi   = ig.Graph.Read_GraphMLz(DATA/"epinions-trust.graphml.gz")
Slash = ig.Graph.Read_GraphMLz(DATA/"slashdot-zoo.graphml.gz")
Wiki  = ig.Graph.Read_GraphMLz(DATA/"wikipedia-elections.graphml.gz")
NETWORKS = [Epi, Slash, Wiki]

# Measure time function -------------------------------------------------------

def measure_time(B, method, *args, **kwds):
    start = time()
    getattr(B, method)(*args, **kwds)
    return time()-start


# Data frames -----------------------------------------------------------------

data = pd.DataFrame({
    "name": np.repeat([ G["name"] for G in NETWORKS ], len(M)),
    "n_nodes": np.repeat([ G.vcount() for G in NETWORKS ], len(M)),
    "n_edges": np.repeat([ G.ecount() for G in NETWORKS ], len(M)),
    "m": list(M)*len(NETWORKS)
})


# Main loop -------------------------------------------------------------------

for method in tqdm(METHODS):
    for weak in tqdm((False, True)):
        results = []
        for G in tqdm(NETWORKS):
            for m in tqdm(M):
                vals = np.empty(10, dtype=float)
                for i in tqdm(range(vals.size)):
                    B = Balance(G, m=m)
                    vals[i] = measure_time(B, method, weak=weak)
                suffix = "w" if weak else "s"
                results.append(vals)
        data[f"{method}_{suffix}"] = results

# -----------------------------------------------------------------------------

with open(HERE/"efficiency.pkl", "wb") as fh:
    pickle.dump(data, fh)

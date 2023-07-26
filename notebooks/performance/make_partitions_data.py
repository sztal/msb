# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=import-error,redefined-outer-name
from pathlib import Path
import pickle
from multiprocessing import cpu_count
from more_itertools import set_partitions
import numpy as np
import pandas as pd
import igraph as ig
from sklearn.metrics import adjusted_mutual_info_score as ami
from tqdm.auto import tqdm
from pqdm.processes import pqdm
from msb import Balance
from msb.utils import frustration_index


# Functions -------------------------------------------------------------------

def iter_partitions(n, kmax=None):
    groups = list(range(n))
    def _iter():
        if kmax is None:
            yield from set_partitions(groups)
        else:
            if kmax > n:
                raise ValueError("'kmax' cannot be greater than 'n'")
            for k in range(1, kmax+1):
                yield from set_partitions(groups, k)

    for part in _iter():
        gvec = np.zeros(n, dtype=int)
        for idx, group in enumerate(part):
            gvec[group] = idx
        yield gvec

def find_best_partition(graph, weighted=True, attr="weight", progress=True, **kwds):
    A = graph.get_adjacency_sparse(attribute=attr)
    if not weighted:
        A.data = np.sign(A.data)
    best_part = None
    best_fidx = None
    for part in tqdm(iter_partitions(graph.vcount(), **kwds), disable=not progress):
        fidx = frustration_index(A, part)
        if best_fidx is None or fidx < best_fidx:
            best_part = part
            best_fidx = fidx
    return best_fidx, best_part

def find_partitions(graph, weighted=True, attr="weight", **kwds):
    try:
        name = graph["name"]
    except KeyError:
        name = f"Monks ({int(graph['t'])})"
    fidx, hc = Balance(graph, weighted=weighted, attr=attr) \
        .find_clusters(full_results=False)
    fidx_best, part_best = \
        find_best_partition(graph, weighted=weighted, attr=attr, **kwds)
    fidx_eb, hc_eb = Balance(graph, weighted=weighted, semi=False) \
        .find_clusters(beta=1, full_results=False)
    part = pd.factorize(hc.labels_)[0]
    part_eb = pd.factorize(hc_eb.labels_)[0]
    part_best = pd.factorize(part_best)[0]
    record = {
        "name":      name,
        "weighted":  weighted,
        "n_nodes":   graph.vcount(),
        "n_edges":   graph.ecount(),
        "fidx":      fidx,
        "fidx_eb":   fidx_eb,
        "fidx_best": fidx_best,
        "k":         len(set(part)),
        "k_eb":      len(set(part_eb)),
        "k_best":    len(set(part_best)),
        "ami":       ami(part, part_best),
        "ami_eb":    ami(part_eb, part_best),
        "ndiff":     sum(part != part_best),
        "ndiff_eb":  sum(part_eb != part_best),
        "part":      part,
        "part_eb":   part_eb,
        "part_best": part_best,
    }
    return record

# -----------------------------------------------------------------------------


# Paths
HERE = Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT/"data"

Tribes = ig.Graph.Read_GraphMLz(DATA/"new-guinea-tribes.graphml.gz")
Monks1 = ig.Graph.Read_GraphMLz(DATA/"sampson"/"t1.graphml.gz")
Monks2 = ig.Graph.Read_GraphMLz(DATA/"sampson"/"t2.graphml.gz")
Monks3 = ig.Graph.Read_GraphMLz(DATA/"sampson"/"t3.graphml.gz")
Monks4 = ig.Graph.Read_GraphMLz(DATA/"sampson"/"t4.graphml.gz")
Monks5 = ig.Graph.Read_GraphMLz(DATA/"sampson"/"t5.graphml.gz")

MONKS  = [Monks1, Monks2, Monks3, Monks4, Monks5]
GRAPHS = [Tribes, *MONKS]
KMAX   = 4

def compute(args, kmax=KMAX):
    weighted, graph = args
    rec = find_partitions(graph, weighted=weighted, kmax=kmax, progress=False)
    return weighted, rec

ARGS = [
    *[(False, graph) for graph in GRAPHS],
    *[(True, graph) for graph in MONKS]
]
N_JOBS = min(len(ARGS), max(cpu_count() - 1, 1))

results = pqdm(ARGS, compute, n_jobs=N_JOBS)

unweighted = pd.DataFrame([ rec for weighted, rec in results if not weighted ])
weighted   = pd.DataFrame([ rec for weighted, rec in results if weighted ])

data = pd.concat([unweighted, weighted], ignore_index=True)

with open(HERE/f"partitions-K{KMAX}.pkl", "wb") as fh:
    pickle.dump(data, fh)

"""Make notebook data."""
# pylint: disable=import-error
import pickle
import gzip
from time import time
from pathlib import Path
import numpy as np
from numba import get_num_threads, set_num_threads
import igraph as ig
from tqdm.auto import tqdm
from msb import Balance
from msb.cycleindex import balance_ratio, set_sampler_seed


def main():
    """Main function."""
    # Paths -------------------------------------------------------------------

    here = Path(__file__).absolute().parent
    root = here.parent.parent
    data = root/"data"

    # Parameters --------------------------------------------------------------

    bkws = {"m": 20}    # Balance params

    # Loading networks --------------------------------------------------------

    labels = {
        "new-guinea-tribes":   "Tribes",
        "epinions-trust":      "Epinions",
        "wikipedia-elections": "Wikipedia",
        "slashdot-zoo":        "Slashdot"
    }
    networks = {
        v: ig.Graph.Read_GraphMLz(data/f"{k}.graphml.gz")
        for k, v in labels.items()
    }

    # Make balance objects ----------------------------------------------------

    balance = {}
    for name, graph in tqdm(networks.items()):
        n_nodes = graph.vcount()
        start = time()
        B = Balance(graph, semi=True, **bkws)
        W = Balance(graph, beta=1, semi=False, **bkws)
        K = B.K(kmax=min(n_nodes, 1000))
        contrib = B.contrib(K=K)
        local = B.local_balance(K=K)
        elapsed = time() - start
        bmin = np.finfo(float).eps
        balance[name] = {
            "Balance":     B.balance(B.B(bmin, 4, 100)),
            "Beta-max":    local,
            "Beta-max-C":  contrib,
            "Beta-1":      W.local_balance(K=K),
            "Beta-1-C":    W.contrib(K=K),
            "Beta-time":   elapsed,
            "Beta-params": {
                "n_nodes": n_nodes,
                "n_edges": graph.ecount(),
                "m": B.m,
                "kmin": B.K().min(),
                "kmax": B.K().max()
            }
        }

    # Local balance based on simple cycles ------------------------------------

    L0 = 20                       # Maximum cycle length considered
    sample_size = 100             # Number of subgraphs to sample
    n_threads = get_num_threads() # Get available number of threads

    set_sampler_seed(101)                # Set random seed for sampling subgraphs
    set_num_threads(max(n_threads-4, 1)) # Set number of threads for parallel processing

    for name, graph in tqdm(networks.items()):
        A = graph.get_adjacency_sparse(attribute="weight") \
            .astype(np.int8) \
            .toarray()
        sample_n = sample_size if len(A) > 40 else None
        if len(A) > 10000:
            sample_n = min(sample_n, 1000)
        parallel = len(A) > 10000 and n_threads > 1
        start = time()
        ratio = balance_ratio(A, L0, sample_size=sample_n, parallel=parallel)
        elapsed = time() - start
        balance[name]["simple"] = ratio
        balance[name]["simple-time"] = elapsed
        balance[name]["simple-params"] = {
            "n_nodes": graph.vcount(),
            "n_edges": graph.ecount(),
            "L0": 20,
            "sample_size": sample_n,
            "parallel": parallel
        }

    # Save data ---------------------------------------------------------------

    with gzip.open(here/"data.pkl.gz", "wb") as fh:
        pickle.dump(balance, fh)


if __name__ == "__main__":
    main()

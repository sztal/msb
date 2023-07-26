from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import igraph as ig
from tqdm.auto import tqdm
from msb import Balance
from msb.utils import gini


# Paths
HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
DATA = ROOT/"data"

# Main parameters
M = [1, 2, 4, 8, 16, 32, 64]
K = [10, 20, 30]
MMAX = None
KMAX = 50
LABELS = ["Real", "ER", "Configuration"]


# Get network dataset (Wikipedia Elections as an undirected network) ----------

G = ig.Graph.Read_GraphMLz(DATA/"congress"/"S114.graphml.gz") \
    .as_undirected(combine_edges="mean")
# Erdos-Renyi random graph with the same number of edges
E = ig.Graph.Erdos_Renyi(n=G.vcount(), m=G.ecount(), directed=False)
E.es["weight"] = 2*np.random.randint(0, 2, E.ecount()) - 1
# Configuration model with the same degree sequence
C = ig.Graph.Degree_Sequence(G.degree(), method="simple")
C.es["weight"] = 2*np.random.randint(0, 2, C.ecount()) - 1


## Reference balance object with K=50 and m=None (full spectrum) --------------
## Original network
Bg = Balance(G, m=MMAX, kmax=KMAX)
## ER model
Be = Balance(E, m=MMAX, kmax=KMAX)
## Configuration model
Bc = Balance(C, m=MMAX, kmax=KMAX)


# Dictionaries with balance objects -------------------------------------------

Bgv = { m: Balance(G, m=m, kmax=KMAX) for m in tqdm(M) }
Bgv[None] = Bg
Bev = { m: Balance(E, m=m, kmax=KMAX) for m in tqdm(M) }
Bev[None] = Be
Bcv = { m: Balance(C, m=m, kmax=KMAX) for m in tqdm(M) }
Bcv[None] = Bc

Bvec = [Bgv, Bev, Bcv]


# Make contribution data -----------------------------------------------------=

cdata = []

for B, label in zip(Bvec, LABELS):
    C0 = B[None].contrib()
    C1 = B[8].contrib()
    df = pd.DataFrame({
        "label": label,
        "n_nodes": G.vcount(),
        "n_edges": G.ecount(),
        "K": B[None].K(),
        "contrib": C0.to_numpy(),
        "contrib8": C1.to_numpy()
    })
    cdata.append(df)

cdata = pd.concat(cdata, axis=0, ignore_index=True)


# Make accuracy data ----------------------------------------------------------

def get_relative_error(Bmap, m, method, **kwds):
    benchmark = np.array(getattr(Bmap[None], method)(**kwds))
    values = np.array(getattr(Bmap[m], method)(**kwds))
    return np.mean(np.abs(benchmark - values) / benchmark)

methods = {
    r"$B(G, \beta_{\max})$": "balance",
    r"$B(G, k)$": "k_balance",
    r"$B(G, \beta_{\max}, i)$": "node_balance",
    r"$B(G, \beta_{\max}, i, j)$": "pairwise_cohesion"
}

adata = []

for key, method in tqdm(methods.items()):
    for B, label in tqdm(list(zip(Bvec, LABELS))):
        err_s = np.array([ get_relative_error(B, m, method) for m in M ])
        err_w = np.array([ get_relative_error(B, m, method, weak=True) for m in M ])
        df = pd.DataFrame({
            "label": label,
            "key": key,
            "method": method,
            "m": M,
            "err_s": err_s,
            "err_w": err_w,
            "gini_m": [ gini(np.abs(B[m].eigen(B[m].U)[0])) for m in M ],
            "gini": gini(np.abs(B[None].eigen(B[None].U)[0]))
        })
        adata.append(df)

adata = pd.concat(adata, axis=0, ignore_index=True)

data = dict(cdata=cdata, adata=adata)
with open(HERE/"accuracy.pkl", "wb") as fh:
    pickle.dump(data, fh)

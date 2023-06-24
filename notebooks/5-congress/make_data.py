"""Make data for the notebook analysis."""
# pylint: disable=import-error,redefined-outer-name
import pickle
import gzip
from pathlib import Path
from datetime import date
import json
import numpy as np
import pandas as pd
import igraph as ig
from tqdm.auto import tqdm
from sklearn.metrics import adjusted_mutual_info_score as ami
from msb import Balance
from msb.utils import frustration_index


# Paths -----------------------------------------------------------------------

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
DATA = ROOT/"data"/"congress"
FIGS = ROOT/"figs"
FIGS.mkdir(exist_ok=True)


# Balance params --------------------------------------------------------------

BKWS = { "m": 10 }


# Get network datasets --------------------------------------------------------

NETWORKS = { "H": [], "S": [] }
for i in range(93, 115):
    for chamber in ("H", "S"):
        G = ig.Graph.Read_GraphMLz(str(DATA/f"{chamber}{i}.graphml.gz"))
        NETWORKS[chamber].append(G)


# Additional metadata ---------------------------------------------------------

DATES = {
    (n+93): (date(1973+n*2, 1, 3), date(1974+n*2, 1, 3), date(1975+n*2, 1, 3))
    for n in range(len(NETWORKS["H"]))
}
## US PRESIDENTS (from: https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States)
with open(DATA/"presidents.json", "r", encoding="utf-8") as fh:
    PRESIDENTS = json.loads(fh.read())


# Compute balance objects -----------------------------------------------------

BALANCE = {
    k: [ Balance(G, **BKWS) for G in tqdm(v) ]
    for k, v in NETWORKS.items()
}
BALANCE_WB = {
    k: [ Balance(G, beta=1, semi=False, **BKWS) for G in tqdm(v) ]
    for k, v in NETWORKS.items()
}


# Descriptive stats for the networks ------------------------------------------

stats = pd.concat([ pd.DataFrame({
    "congress": [ int(G["congress"]) for G in v ],
    "n_rep":    [ sum(1 for x in G.vs["party"] if x == "R") for G in v],
    "n_dem":    [ sum(1 for x in G.vs["party"] if x == "D") for G in v],
    "n_oth":    [ sum(1 for x in G.vs["party"] if x not in ("R", "D")) for G in v],
    "n_nodes":  [ G.vcount() for G in v ],
    "n_edges":  [ G.ecount() for G in v ],
    "n_pos":    [ len(G.es.select(weight_lt=0)) for G in v ],
    "n_neg":    [ len(G.es.select(weight_gt=0)) for G in v ],
    "dobs":     [ B.balance(beta=B.beta.max(), weak=False) for B in BALANCE[k] ],
    "dobw":     [ B.balance(beta=B.beta.max(), weak=True) for B in BALANCE[k] ],
    "dbar":     [ np.mean(G.degree()) for G in v ],
    "dstd":     [ np.std(G.degree()) for G in v ]
}) for k, v in NETWORKS.items() ], keys=list(NETWORKS), names=["chamber"]) \
    .assign(
        dcv=lambda df: df["dstd"]/df["dbar"],
        f_pos=lambda df: df["n_pos"] / df["n_edges"],
        f_neg=lambda df: df["n_neg"] / df["n_edges"],
        f_rep=lambda df: df["n_rep"] / df["n_nodes"],
        f_dem=lambda df: df["n_dem"] / df["n_nodes"]
    ) \
    .reset_index(level=-1, drop=True) \
    .set_index("congress", append=True)


# Compute clusterings ---------------------------------------------------------

CLUSTERS = {
    k: [ B.find_clusters(max_clusters=200, full_results=True) for B in tqdm(v) ]
    for k, v in BALANCE.items()
}


# Get clustering data ---------------------------------------------------------

def get_best_clust(df, cols=("fidx_s", "fidx_w")):
    """Get best clustering."""
    cols = list(cols)
    fcol = df[cols].min().idxmin()
    idx  = df[fcol].idxmin()
    mode = fcol.split("_")[-1]
    return df.loc[idx, "hc_"+mode].labels_

## Data for analysis
cols = ["fidx_s", "fidx_w"]
cong = [ int(G["congress"]) for G in NETWORKS["H"] ]
pres = pd.DataFrame(PRESIDENTS)[["name", "midname", "surname", "affil", "start", "end"]] \
    .assign(
        start=lambda df: pd.to_datetime(df["start"]),
        end=lambda df: pd.to_datetime(df["end"])
    )

data = pd.concat([ pd.DataFrame({
    "congress": cong,
    "start":    [ DATES[c][0] for c in cong ],
    "mid":      [ DATES[c][1] for c in cong ],
    "end":      [ DATES[c][2] for c in cong ],
    "n":        [ df.set_index("n")[cols].idxmin().min() for df in v ],
    "fbip":     [
        (c := np.sort(np.unique(get_best_clust(cdf), return_counts=True)[1]))[-2:].sum() / c.sum()
        for cdf in CLUSTERS[k]
    ],
    "fidx":     [ df.set_index("n")[cols].min().min() for df in v ],
    "fidx2":    [ df.set_index("n").loc[2, cols].min().min() for df in v ],
    "fidxp":    [ frustration_index(B.S, G.vs["party"]) for G, B in zip(NETWORKS[k], BALANCE[k]) ],
    "amip":     [
        ami(
            G.vs["party"],
            get_best_clust(cdf, cols=cols)
        ) for G, cdf in zip(NETWORKS[k], CLUSTERS[k])
    ],
    "amip2":    [
        ami(G.vs["party"], cdf.loc[cdf["n"] == 2, "hc_s"].to_numpy()[0].labels_,)
        for G, cdf in zip(NETWORKS[k], CLUSTERS[k])
    ],
    "dobs":     [ B.balance(beta=B.beta.max(), weak=False) for B in BALANCE[k] ],
    "dobw":     [ B.balance(beta=B.beta.max(), weak=True) for B in BALANCE[k] ],
    "dobs1":    [ B.balance(beta=1, weak=False) for B in BALANCE_WB[k] ],
    "dobw1":    [ B.balance(beta=1, weak=True) for B in BALANCE_WB[k] ]
}) for k, v in CLUSTERS.items() ], axis=0, keys=list(CLUSTERS), names=["chamber"]) \
    .reset_index(level=0) \
    .reset_index(drop=True)


# Save data -------------------------------------------------------------------

with gzip.open(HERE/"data.pkl.gz", "wb") as fh:
    pickle.dump((data, stats), fh)

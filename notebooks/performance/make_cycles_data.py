"""Make cycles accuracy data."""
# pylint: disable=import-error,redefined-outer-name
import pickle
import gzip
from pathlib import Path
import numpy as np
import igraph as ig
from tqdm.auto import tqdm
from msb import Balance
from msb.cycleindex import balance_ratio


# Paths -----------------------------------------------------------------------

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
DATA = ROOT/"data"/"congress"
FIGS = ROOT/"figs"
FIGS.mkdir(exist_ok=True)


# Balance params --------------------------------------------------------------

KMAX = 15
NSAM = 10000
BKWS = { "m": 10, "kmax": KMAX }
CKWS = { "exact": False, "n_samples": NSAM, "length": KMAX, "parallel": True }


# Get network datasets --------------------------------------------------------

RESULTS = { "H": [], "S": [] }
for chamber in tqdm(("H", "S")):
    for i in tqdm(range(93, 115)):
        G = ig.Graph.Read_GraphMLz(str(DATA/f"{chamber}{i}.graphml.gz"))
        B = Balance(G, **BKWS).k_balance()
        A = np.array([*G.get_adjacency(attribute="weight")])
        C = 1 - balance_ratio(A, **CKWS)
        rec = { "name": G["name"], "dob": B, "cycles": C }
        RESULTS[chamber].append(rec)


# Save data -------------------------------------------------------------------

with gzip.open(HERE/f"cycles-{NSAM}.pkl.gz", "wb") as fh:
    pickle.dump(RESULTS, fh)

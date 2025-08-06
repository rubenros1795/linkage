# Analyze topic model dynamics: modularity, density, effect sizes, and community stability (1945–1994).

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, entropy
from cliffs_delta import cliffs_delta
from scipy.stats import mannwhitneyu
import networkx as nx
from cdlib import algorithms
from cdlib.evaluation import normalized_mutual_information
from cdlib.model import TemporalClustering

from config import *
from helpers import metrics as mts
from helpers.afa import adaptive_filter
from helpers.dataloader import load, load_diag
from helpers.linkage import mutual_information_smooth
from helpers.pathtools import *
from helpers.style import load_style
from helpers.visuals import add_cabinet_periods, plot_trend

# Style
load_style(SMALL_SIZE=18)

# Density vs. Modularity
di_path = os.path.join(DATA_DIR_REVISIONS, f"doc-topics-1945-1994-250.tsv")
da_path = os.path.join(DATA_DIR_REVISIONS, "data.tsv")
ks_path = os.path.join(DATA_DIR_REVISIONS, f"topic-keys-1945-1994-250.tsv")

dists, dat, coltrans, keys, dz = load(
    dist_path = di_path,
    dat_path = da_path,
    keys_path = ks_path,
    filter_thematic = True,
    zscore_filter = False
)

dists = dists[dists.index.year > 1945]

r = []
networks = {}

for cd, (date, _) in enumerate(dists.groupby(dists.index)):
    rij, ri, m = mutual_information_smooth(theta = _.to_numpy())
    rij = pd.DataFrame(rij).stack().reset_index().rename(columns={"level_0": "s", "level_1": "t", 0: "pmi"})
    rij = rij[(rij.s != rij.t) & (rij.pmi > 0)]
    g = nx.from_pandas_edgelist(df = rij, source='s', target='t', edge_attr='pmi')
    networks[date] = g

f, a = plt.subplots(1, 1, figsize=(6, 4))
tc = TemporalClustering()
for cc, (c, g) in enumerate(sorted(networks.items(), key=lambda n: n[0])):
    coms = algorithms.louvain(g, weight='pmi', resolution=2.5)
    tc.add_clustering(coms, cc)

y = tc.clustering_stability_trend(method=normalized_mutual_information)
x = pd.Series(sorted(networks.keys())[1:])
dat = pd.Series(y, index=x)

a.plot(dat.index, dat.rolling(window=4, center=True).mean(), color='black', lw=2)
plot_trend(x=x, y=y, color='darkgrey', ls='--', ax=a, alpha=1)
add_cabinet_periods(ax=a, min_time=1948, max_time=1994, text=False, color='lightgrey', linestyle='-')

a.set_ylabel('Normalized Mutual\nInformation', labelpad=10, fontsize=20)
a.set_xlabel('6-Month Periods (1945 — 1994)', color='black', fontsize=20, labelpad=10)

plt.subplots_adjust(hspace=0.5)
plt.savefig('figs/clustering-stability.pdf', dpi=400, bbox_inches='tight')
plt.close()

# Analyze and visualize average topic entropy over time from a topic model (1945–1994).

import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy

from config import *
from helpers import style
from helpers.afa import adaptive_filter
from helpers.dataloader import load
from helpers.visuals import add_cabinet_periods

# Apply custom plot style
style.load_style()

# Load topic model data
di_path = os.path.join(DATA_DIR_REVISIONS, "doc-topics-1945-1994-250.tsv")
da_path = os.path.join(DATA_DIR_REVISIONS, "data.tsv")
ks_path = os.path.join(DATA_DIR_REVISIONS, "topic-keys-1945-1994-250.tsv")

dists, dat, coltrans, keys = load(
    dist_path = di_path,
    dat_path = da_path,
    keys_path = ks_path,
    filter_thematic = True,
    zscore_filter = False
)

# Compute mean entropy over time
y = dists.apply(entropy, axis=1).groupby(dists.index).mean()

# Plot entropy trend
_, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    y.index,
    adaptive_filter(y.tolist(), span=50),
    color='black',
    alpha=1,
    lw=2,
    label='Modularity'
)

ax.set_ylabel('Average Entropy', color='black', fontsize=20, labelpad=10)
ax.set_xlabel('6-Month Periods (1945 — 1994)', color='black', fontsize=20, labelpad=10)

add_cabinet_periods(ax=ax, min_time=1948, max_time=1994, text=False, color='lightgrey', linestyle='-')
plt.savefig('figs/topic-dist-entropy.pdf', dpi=400, bbox_inches='tight')
plt.show()

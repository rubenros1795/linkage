# Code for Plotting TNS (Topic Neighbourhood Stability)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from matplotlib.gridspec import GridSpec
from itertools import combinations

from helpers import style
from helpers.afa import adaptive_filter
from helpers.linkage import mutual_information_smooth
from helpers.metrics import overlap_coefficient
from helpers.visuals import plot_trend, add_cabinet_periods
from helpers.dataloader import load
from config import *

style.load_style()

# Load data
di_path = os.path.join(DATA_DIR_REVISIONS, f"doc-topics-1945-1994-250.tsv")
da_path = os.path.join(DATA_DIR_REVISIONS, "data.tsv")
ks_path = os.path.join(DATA_DIR_REVISIONS, f"topic-keys-1945-1994-250.tsv")

dists, dat, coltrans, keys = load(
                                    dist_path = di_path,
                                    dat_path = da_path,
                                    keys_path = ks_path,
                                    filter_thematic = True,
                                    zscore_filter = False
                                )

# Build topic networks
networks = {}

for cd, (date, _) in enumerate(dists.groupby(dists.index)):
    rij, ri, m = mutual_information_smooth(theta = _.to_numpy())
    rij = pd.DataFrame(rij).stack().reset_index().rename(columns={"level_0": "s", "level_1": "t", 0: "pmi"})
    rij = rij[(rij.s != rij.t) & (rij.pmi > 0)]
    rij['t'] = rij.t.astype(int).apply(lambda x: str(x) + ' - ' + coltrans[x])
    rij['s'] = rij.s.astype(int).apply(lambda x: str(x) + ' - ' + coltrans[x])
    g = nx.from_pandas_edgelist(df = rij, source = 's', target = 't', edge_attr = 'pmi')
    networks[date] = g

# Neighborhood similarity with overlap coefficient
r = []
dates = sorted(networks.keys())

for c, date in enumerate(dates):
    if c < 4:
        continue

    network_current = networks[date]
    ovls = {}

    for i in range(4):
        network_previous = networks[dates[c - i]]

        for node in set(network_current.nodes).intersection(network_previous.nodes):
            nb_c = set(network_current.neighbors(node))
            nb_p = set(network_previous.neighbors(node))
            ovl = overlap_coefficient(nb_c, nb_p)
            ovls.setdefault(node, []).append(ovl)

    for node, o in ovls.items():
        r.append({"p1": date, "topic": node, "o": np.mean(o)})

rd = pd.DataFrame(r)

# Plotting figure with 3 sample topics
fig = plt.figure(layout="constrained", figsize=(7, 5))
gs = GridSpec(3, 9, figure=fig)

# Define axes
ax1 = fig.add_subplot(gs[1:, :])
ax2 = fig.add_subplot(gs[0, 0:3])
ax3 = fig.add_subplot(gs[0, 3:6], sharey=ax2)
ax4 = fig.add_subplot(gs[0, 6:9], sharey=ax2)

# Filter topic-specific data
env_data = rd[rd.topic.str.startswith('35 ')]
conflict_data = rd[rd.topic.str.startswith('51 ')]
broadcast_data = rd[rd.topic.str.startswith('11 ')]

# Smoothed lines
env_smoothed = env_data['o'].ewm(span=4).mean()
conflict_smoothed = conflict_data['o'].ewm(span=4).mean()
broadcast_smoothed = broadcast_data['o'].ewm(span=4).mean()

# Subplot lines
ax2.plot(env_data['p1'], env_smoothed, color='black')
ax3.plot(conflict_data['p1'], conflict_smoothed, color='black')
ax4.plot(broadcast_data['p1'], broadcast_smoothed, color='black')

# Horizontal mean lines
ax2.axhline(env_smoothed.mean(), color='black', linestyle='--', linewidth=0.5)
ax3.axhline(conflict_smoothed.mean(), color='black', linestyle='--', linewidth=0.5)
ax4.axhline(broadcast_smoothed.mean(), color='black', linestyle='--', linewidth=0.5)

# Titles
ax2.set_title('Environmental\nManagement', fontsize=15)
ax3.set_title('International\nConflict', fontsize=15)
ax4.set_title('Public\nBroadcasting', fontsize=15)

# Format axes
for ax in [ax2, ax3, ax4]:
    ax.set_ylim(0.45, 0.85)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_xlabel('')
    for label in ax.get_xticklabels():
        label.set_rotation(0)

for ax in [ax3, ax4]:
    ax.tick_params(labelleft=False)

# Main panel
sns.lineplot(data = rd, x = 'p1', y = 'o', ax = ax1, color = 'black')
ax1.get_children()[1].set_color('grey')  # Fill
add_cabinet_periods(ax1, min_time=1950, max_time=1994, text=False, color='lightgrey', linestyle='-')

ax1.set_xlabel('6-MONTH PERIODS (1945 — 1994)', labelpad=10)
ax1.set_ylabel('OVERLAP RATIO (W=4)', fontsize=20, labelpad=10)

plt.subplots_adjust(hspace=0.5)
plt.savefig('figs/nbh-similarity.pdf', dpi=400, bbox_inches='tight')
plt.close()

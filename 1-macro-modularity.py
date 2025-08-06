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
r = []
for k in [50, 100, 150, 200, 250]:
    di_path = os.path.join(DATA_DIR_REVISIONS, f"doc-topics-1945-1994-{k}.tsv")
    da_path = os.path.join(DATA_DIR_REVISIONS, "data.tsv")
    ks_path = os.path.join(DATA_DIR_REVISIONS, f"topic-keys-1945-1994-{k}.tsv")

    dists, dat, coltrans, keys, dz = load(
        dist_path = di_path,
        dat_path = da_path,
        keys_path = ks_path,
        filter_thematic = True,
        zscore_filter = True
    )

    dists = dists[dists.index.year > 1945]
    density = dists.groupby(dists.index).apply(lambda g: mts.density(g.to_numpy()))
    modularity = dists.groupby(dists.index).apply(lambda g: mts.modularity(g.to_numpy()))
    df = pd.concat([density, modularity], axis=1).rename(columns={0: 'density', 1: 'modularity'})
    df = df.assign(k = k)
    r.append(df)

rd = pd.concat(r)

# Plot Macro Trends
_, a = plt.subplots(1, 1, figsize=(6, 4))
aa = a.twinx()
pal = sns.color_palette('Greys', 6)[1:]

a.plot(
    modularity.index,
    adaptive_filter(modularity, span=30),
    color='black',
    alpha=0.25 if k != 250 else 1,
    lw=0.5 if k != 250 else 2,
    label='Modularity'
)

aa.plot(
    density.index,
    adaptive_filter(density, span=30),
    color='darkgrey',
    alpha=0.25 if k != 250 else 1,
    lw=0.5 if k != 250 else 2,
    linestyle='--',
    label='Density'
)

a.set_ylabel('Modularity (K = 250)', color='black', fontsize=20, labelpad=10)
a.set_xlabel('6-Month Periods (1945 — 1994)', color='black', fontsize=20, labelpad=10)
aa.set_ylabel('Density', color='darkgrey', fontsize=20, labelpad=10)
aa.yaxis.set_label_position("right")
aa.spines['right'].set_color('darkgrey')
aa.tick_params(colors='darkgrey')

a.spines['top'].set_alpha(0)
aa.spines['top'].set_alpha(0)

aim = a.inset_axes([0, 1.1, .45, .35])
aid = a.inset_axes([.55, 1.1, .45, .35])

for cc, (k, d) in enumerate(rd.groupby('k')):
    x = d.index
    aim.plot(x, adaptive_filter(zscore(d.modularity), span=35), color=pal[cc], alpha=1, label=f'{k}', lw=1)
    aid.plot(x, adaptive_filter(zscore(d.density), span=35), color=pal[cc], alpha=1, linestyle='--', label=f'{k}', lw=1)

    aim.set_ylim(-2, 2)
    aid.set_ylim(-2, 2)
    aim.xaxis.set_tick_params(labelsize=8, size=0.1)
    aim.yaxis.set_tick_params(labelsize=8, size=0.1)
    aid.xaxis.set_tick_params(labelsize=8, size=0.1)
    aid.yaxis.set_tick_params(labelsize=8, size=0.1)

aim.set_ylabel('Z-Scored Modularity', fontsize=10)
aid.set_ylabel('Z-Scored Density', fontsize=10)

l = aim.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), framealpha=1, fancybox=True, fontsize=10,
               handlelength=0.5, ncol=5, edgecolor='black', columnspacing=0.5, handletextpad=0.3)
for legline in l.get_lines():
    legline.set_linewidth(3)

l = aid.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), framealpha=1, fancybox=True, fontsize=10,
               handlelength=0.5, ncol=5, edgecolor='black', columnspacing=0.5, handletextpad=0.3)
for legline in l.get_lines():
    legline.set_linewidth(3)

add_cabinet_periods(ax=a, min_time=1948, max_time=1994, text=False, color='lightgrey', linestyle='-')

plt.savefig('figs/macro-mod-dens.pdf', dpi=400, bbox_inches='tight')
plt.show()

# Effect Size of Modularity
def interpret_cliffs_delta(delta):
    abs_delta = abs(delta)
    if abs_delta < 0.15:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.47:
        return "medium"
    else:
        return "large"

def calculate_cliffs_delta(x1, x2):
    delta_value, _ = cliffs_delta(x1, x2)
    return delta_value, interpret_cliffs_delta(delta_value)

def mann_whitney_u(x1, x2):
    stat, p = mannwhitneyu(x1, x2, alternative="two-sided")
    return stat, p

def analyze_modularity_nonparametric(modularity_series: pd.Series):
    modularity_series = modularity_series.dropna()
    if not isinstance(modularity_series.index, pd.DatetimeIndex):
        raise ValueError("Input series must have a DatetimeIndex.")
    modularity_series = modularity_series.sort_index()

    decade_data = {}
    for dt, val in modularity_series.items():
        decade = (dt.year // 5) * 5
        decade_data.setdefault(decade, []).append(val)

    decades = sorted(decade_data.keys())
    for i in range(len(decades) - 1):
        mod1 = decade_data[decades[i]]
        mod2 = decade_data[decades[i + 1]]

        cliffs_d, interp = calculate_cliffs_delta(mod2, mod1)
        u_stat, p_value = mann_whitney_u(mod1, mod2)

        print(f"{decades[i]}s → {decades[i + 1]}s:")
        print(f"  Cliff's δ: {cliffs_d:.4f} ({interp})")
        print(f"  Mann-Whitney U: {u_stat:.1f} (p = {p_value:.4f})")

    vals = modularity_series.values
    midpoint = len(vals) // 2
    d1, d2 = vals[:midpoint], vals[midpoint:]
    cliffs_d, interp = calculate_cliffs_delta(d2, d1)
    u_stat, p_value = mann_whitney_u(d1, d2)

    print("\n=== Overall Comparison ===")
    print(f"  Cliff's δ: {cliffs_d:.4f} ({interp})")
    print(f"  Mann-Whitney U: {u_stat:.1f} (p = {p_value:.4f})")

mod_signal = rd[rd.k == 250].modularity
analyze_modularity_nonparametric(mod_signal)

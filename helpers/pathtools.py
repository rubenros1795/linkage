import random
from tqdm import tqdm
import networkx as nx
from itertools import combinations
from collections import Counter
from cdlib import LifeCycle, TemporalClustering, algorithms
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import random

def overlap_coefficient(set_a, set_b):
  intersection = len(set_a.intersection(set_b))
  smaller_set = min(len(set_a), len(set_b))
  if smaller_set == 0:
    return 0
  elif set_a <= set_b or set_b <= set_a:
    return 1
  else:
    return intersection / smaller_set
    
def jaccard_similarity(list1, list2):
    return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))

def chain2topic(communities,chain,s=4):
    topics = []
    for node in chain:
        x = int(node.split('-')[0])
        topics += list(communities[node])
    return ', '.join([f"{t} ({c})" for t,c in Counter(topics).most_common(s)])

# Function to create chains of communities
def create_community_chains(events, min_chain_len = 6):
    chains = []
    visited = set()
    
    def follow_chain(start):
        chain = [start]
        current = start
        while current in events and events[current]['out_flow']:
            next_community = list(events[current]['out_flow'].keys())[0]
            chain.append(next_community)
            current = next_community
        return chain

    for community in events.keys():
        if community not in visited:
            chain = follow_chain(community)
            chains.append(chain)
            visited.update(chain)

    return [c for c in chains if len(c) > min_chain_len]

def get_tc(networks, size_dict=None, overlap_threshold=.35, min_chain_len=2, louvain_res=1, verbose=True):
    tc = TemporalClustering()

    for cc,(c,g) in enumerate(sorted(networks.items(),key = lambda n: n[0])):
        if louvain_res == None:
            coms = algorithms.leiden(g,weights='pmi')
        else:
            coms = algorithms.louvain(g,weight='pmi',resolution=louvain_res)
        tc.add_clustering(coms, cc)

    # Sizes
    if size_dict:
        sizes = {i:{cc:max([(t,s) for t,s in size_dict[i] if t in c],key=lambda x:x[1])[0] 
            for cc,c in enumerate(json.loads(tc.get_clustering_at(i).to_json())['communities'])} 
            for i in tc.get_observation_ids()}

    # Get 'Flows' of communities
    lc = LifeCycle(tc)
    overlap_coefficient = lambda x, y: len(set(x) & set(y)) / min(len(set(x)), len(set(y)))
    lc.compute_events_with_custom_matching(overlap_coefficient, threshold=overlap_threshold, two_sided=False)
    flows = {k:v.to_json() for k,v in lc.events.items()}

    # Create chains of communities
    paths = create_community_chains(flows, min_chain_len = min_chain_len)
    if verbose == True:
        print('average n. of comms per snapshot',np.mean([len(tc.get_clustering_at(d).communities) for d in range(cc)]))
        print(len(paths), 'found in temporal networks')
    
    g = lc.polytree()
    metadata = pd.DataFrame(list(g.nodes),columns=['node'])
    metadata['in_degree'] = metadata.node.map(dict(g.in_degree()))
    metadata['out_degree'] = metadata.node.map(dict(g.out_degree()))
    metadata['in_chain'] = metadata.node.apply(lambda n: True if n in set([item for items in paths for item in items]) else False)
    metadata['topics'] = metadata.node.apply(lambda n: ', '.join(tc.get_community(n)))
    metadata['time'] = metadata.node.apply(lambda n: sorted(list(networks.keys()))[int(n.split('_')[0])])
    metadata['is_path_start'] = metadata.node.apply(lambda n: True if n in [p[0] for p in paths] else False)
    metadata['is_path_end'] = metadata.node.apply(lambda n: True if n in [p[-1] for p in paths] else False)

    return tc, g, sizes, paths, metadata

def plot_flows(g, tc, paths, networks, sizes, cmap='hsv', annotate_max_topic=True, add_non_path_nodes = True, figsize=(60,10), save = 'figs/chains.pdf'):

    cids = [[f"{i}_{c}" for c in range(len(cc.communities))] for i,cc in tc.clusterings.items()]
    cids = [item for items in cids for item in items]

    if add_non_path_nodes == True:
        for cid in cids:
            if cid not in g.nodes:
                g.add_node(cid)

    paths = sorted(paths, key=lambda x: int(x[0].split('_')[0]))
    layer_mapping = {node: int(node.split('_')[0]) for node in g.nodes}
    for node, layer in layer_mapping.items():
        g.nodes[node]['layer'] = layer

    # Create multipartite layout
    pos = nx.multipartite_layout(g, subset_key='layer')

    # Extract node and edge positions
    x_nodes = [pos[node][0] for node in g.nodes]
    y_nodes = [pos[node][1] for node in g.nodes]
    edges = list(g.edges())
    x_edges = [[pos[edge[0]][0], pos[edge[1]][0]] for edge in edges]
    y_edges = [[pos[edge[0]][1], pos[edge[1]][1]] for edge in edges]

    # Create a color palette
    colors = list(sns.color_palette(cmap, len(paths))) if cmap.startswith('c') == False else [cmap[1:]] * len(paths)
    # random.shuffle(colors)

    # Track node colors and chains
    node_chain_count = {node: 0 for node in g.nodes}
    node_colors = {node: "lightgrey" for node in g.nodes}  # Default to gray with alpha 0.25
    edge_alphas = {edge: 0.1 for edge in g.edges}
    node_sizes = {node: 50 for node in g.nodes}  # Default size for out-chain nodes

    # Increment node count for each chain
    for chain in paths:
        for node in chain:
            if node in node_chain_count:
                node_chain_count[node] += 1

    # Create subplots
    fig, ax = plt.subplots(figsize=figsize)

    # Highlight edges and nodes with unique colors for each chain
    for idx, chain in enumerate(paths):
        color = colors[idx]
        for i in range(len(chain) - 1):
            if g.has_edge(chain[i], chain[i+1]):
                ax.plot([pos[chain[i]][0], pos[chain[i+1]][0]], [pos[chain[i]][1], pos[chain[i+1]][1]], color=color, zorder=0)
                edge_alphas[(chain[i], chain[i+1])] = 1.0  # Set edge alpha to 1.0

        for node in chain:
            node_colors[node] = color  # Set node color with alpha 1.0
            node_sizes[node] = 750  # In-chain node size
    
    node_sizes = [node_sizes[node] for node in g.nodes]
    node_colors = [node_colors[node] for node in g.nodes]

    for n, x,y,s,c in zip(g.nodes, x_nodes, y_nodes, node_sizes, node_colors):
        is_start = True if n in [p[0] for p in paths] else False
        is_end = True if n in [p[-1] for p in paths] else False
        linestyle = 'dashed' if g.in_degree[n] > 1 else 'dashdot' if g.out_degree[n] > 1 else 'solid'
        ax.scatter([x],[y],s=s,c=c,edgecolors="black" if linestyle != 'solid' else 'white',zorder=3)

        if is_start:
            ax.scatter([x],[y], s=s * .95, c="black", edgecolors=None, alpha=.1, zorder = 4, marker=MarkerStyle("o", fillstyle="left"))
        if is_end:
            ax.scatter([x],[y], s=s * .95, c="black", edgecolors=None, alpha=.1, zorder = 4, marker=MarkerStyle("o", fillstyle="right"))

    # Draw edges with plot
    # for edge, alpha in edge_alphas.items():
    #     ax.plot([pos[edge[0]][0], pos[edge[1]][0]], [pos[edge[0]][1], pos[edge[1]][1]], color='gray', alpha=alpha, zorder=1)

    # Draw labels with annotate
    for node in g.nodes:
        if add_non_path_nodes == True:
            if node_chain_count[node] == 0:
                continue
        if node_chain_count[node] > 0:  # Annotate only in-chain nodes
            period, com = (int(x) for x in node.split('_'))
            if annotate_max_topic == True:
                txt = node + '\n' + sizes[period][com].replace(' ','\n').upper()
            else:
                txt = node
            ax.annotate(txt, xy=pos[node], xytext=(0, 0), textcoords='offset points', fontsize=4, color='black', ha='center', va='center', zorder=5,fontweight='bold')
            
    # Remove axis
    ax.axis('off')

    # Ticks
    for ix,date in {c:k for c,k in enumerate(sorted(networks.keys())) if c % 5 == 0}.items():
        y = min(y_nodes) + (min(y_nodes) * .1)
        x = sorted(list(set(x_nodes)))[ix]
        ax.annotate(text=date.year,xy=(x,y),fontsize=50,ha='center',va='center')

    plt.savefig(save,bbox_inches='tight',dpi=400)
    plt.show()


def coverage(tc, paths, networks):
    nw_topic_size = {k:len(v.nodes) for k,v in networks.items()}
    cl_comm_size = {sorted(networks.keys())[i]:len(tc.get_clustering_at(i).communities) for i in tc.get_observation_ids()}

    topic_coverage = []
    community_coverage = []

    for c in paths:
        for co in c:
            period, cluster = co.split('_')
            period, cluster = int(period), int(cluster)
            community_coverage.append({"period":sorted(networks.keys())[period],"cluster":cluster})
            for t in tc.get_community(co):
                topic_coverage.append({"period":sorted(networks.keys())[period],"topic":t})

    topic_coverage = pd.DataFrame(topic_coverage).groupby('period').topic.nunique().reset_index(name='n_topics')
    topic_coverage['nn_topics'] = topic_coverage.n_topics / topic_coverage.period.map(nw_topic_size)

    community_coverage = pd.DataFrame(community_coverage).groupby('period').cluster.nunique().reset_index(name='n_comms')
    community_coverage['nn_comms'] = community_coverage.n_comms / community_coverage.period.map(cl_comm_size)

    coverage = pd.merge(topic_coverage, community_coverage, on = 'period')
    return coverage
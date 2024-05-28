import igraph
import networkx as nx
import pandas as pd
import numpy as np
import sys, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import binom


def flatten_mi_array(theta):
    m,n = theta.shape
    theta[:] = np.where(np.arange(m)[:,None] >= np.arange(n),np.nan,theta)
    ta_ = theta.flatten()
    ix = np.array(list(np.ndindex(theta.shape)))
    na = np.column_stack((ix[:,0],ix[:,1],ta_))
    return pd.DataFrame(na).dropna()


def get_network_from_mi_theta(mi_theta=None,dis_filter=None,weight_threshold=None,weighted=False,words=None,labels=None,node_text=None,cluster_method='louvain'):

    edge_df = flatten_mi_array(mi_theta)
    edge_df.columns = ['source','target','weight']

    # Apply disparity filter and weight threshold, possibly both
    if dis_filter != None and weight_threshold == None:
        edge_df.columns = ['src','trg','nij']
        edge_df = disparity_filter(edge_df,undirected=True)
        edge_df = edge_df[edge_df.score > dis_filter]
        edge_df = edge_df.drop(columns=['score','variance'])
        edge_df.columns = ['source','target','weight']

    elif dis_filter != None and weight_threshold != None:
        edge_df = edge_df[edge_df['weight'] > weight_threshold]
        edge_df.columns = ['src','trg','nij']
        edge_df = disparity_filter(edge_df,undirected=True)
        edge_df = edge_df[edge_df.score > dis_filter]
        edge_df = edge_df.drop(columns=['score','variance'])
        edge_df.columns = ['source','target','weight']

    elif dis_filter == None and weight_threshold != None:
        edge_df.columns = ['source','target','weight']
        edge_df = edge_df[edge_df['weight'] > weight_threshold]
    else:
        pass
        
    if node_text == 'labels':
        edge_df['source']  = edge_df['source'].apply(lambda x: labels[x])
        edge_df['target']  = edge_df['target'].apply(lambda x: labels[x])
        
    if node_text == 'words':
        edge_df['source']  = edge_df['source'].apply(lambda x: ' '.join(words[x].split(' ')[:4]))
        edge_df['target']  = edge_df['target'].apply(lambda x: ' '.join(words[x].split(' ')[:4]))
    
    if node_text == 'words_n':
        edge_df['source']  = edge_df['source'].apply(lambda x: '\n'.join(words[x].split(' ')[:3]))
        edge_df['target']  = edge_df['target'].apply(lambda x: '\n'.join(words[x].split(' ')[:3]))
    

    edge_df.columns = ['source','target','weight']
    edge_df = edge_df[edge_df.source != edge_df.target]


    G = igraph.Graph.TupleList(edge_df.itertuples(index=False), directed=False, weights=weighted)

    clustering = G.community_multilevel() if cluster_method == 'louvain' else G.community_leiden()
    return G, clustering, edge_df

def get_betweenness_values(g):
    return list(zip([v['name'] for v in g.vs()],g.betweenness()))

def get_degree_values(g):
    return list(zip([v['name'] for v in g.vs()],g.degree()))

def get_centralization_of_degree(g, weight="weight"):
    N=g.order()
    indegrees = dict(g.in_degree()).values()
    max_in = max(indegrees)
    centralization = float((N*max_in - sum(indegrees)))/(N-1)**2
    return centralization

def betweenness_centralization(G):
    vnum = G.vcount()
    if vnum < 3:
        raise ValueError("graph must have at least three vertices")
    denom = (vnum - 1) * (vnum - 2)

    temparr = [2 * i / denom for i in G.betweenness()]
    max_temparr = max(temparr)
    return sum(max_temparr - i for i in temparr) / (vnum - 1)


def get_network_statistics(dict_date_theta,weight_threshold=None,df_filter=None,weighted=False,cluster_method='louvain'):
    r = []

    for k,v in dict_date_theta.items():
        tmp_ = {"date":k}
        G, clustering, edge_df = get_network_from_mi_theta(mi_theta=v,dis_filter=df_filter,weight_threshold=weight_threshold,weighted=weighted,cluster_method=cluster_method)
        tmp_['modularity'] = G.modularity(clustering,weights=G.es['weight']) if weighted == True else G.modularity(clustering)
        tmp_['average_degree'] = igraph.mean(G.degree())
        r.append(tmp_)
    return pd.DataFrame(r)

def disparity_filter(table, undirected = False, return_self_loops = False):
    table = table.copy()
    table_sum = table.groupby(table["src"]).sum().reset_index()
    table_deg = table.groupby(table["src"]).count()["trg"].reset_index()
    table = table.merge(table_sum, on = "src", how = "left", suffixes = ("", "_sum"))
    table = table.merge(table_deg, on = "src", how = "left", suffixes = ("", "_count"))
    table["score"] = 1.0 - ((1.0 - (table["nij"] / table["nij_sum"])) ** (table["trg_count"] - 1))
    table["variance"] = (table["trg_count"] ** 2) * (((20 + (4.0 * table["trg_count"])) / ((table["trg_count"] + 1.0) * (table["trg_count"] + 2) * (table["trg_count"] + 3))) - ((4.0) / ((table["trg_count"] + 1.0) ** 2)))
    if not return_self_loops:
        table = table[table["src"] != table["trg"]]
    if undirected:
        table["edge"] = table.apply(lambda x: "%s-%s" % (min(x["src"], x["trg"]), max(x["src"], x["trg"])), axis = 1)
        table_maxscore = table.groupby(by = "edge")["score"].max().reset_index()
        table_minvar = table.groupby(by = "edge")["variance"].min().reset_index()
        table = table.merge(table_maxscore, on = "edge", suffixes = ("_min", ""))
        table = table.merge(table_minvar, on = "edge", suffixes = ("_max", ""))
        table = table.drop_duplicates(subset = ["edge"])
        table = table.drop("edge", 1)
        table = table.drop("score_min", 1)
        table = table.drop("variance_max", 1)
    return table[["src", "trg", "nij", "score", "variance"]]

def noise_corrected(table, undirected = False, return_self_loops = False, calculate_p_value = False):
    sys.stderr.write("Calculating NC score...\n")
    table = table.copy()
    src_sum = table.groupby(by = "src").sum()[["nij"]]
    table = table.merge(src_sum, left_on = "src", right_index = True, suffixes = ("", "_src_sum"))
    trg_sum = table.groupby(by = "trg").sum()[["nij"]]
    table = table.merge(trg_sum, left_on = "trg", right_index = True, suffixes = ("", "_trg_sum"))
    table.rename(columns = {"nij_src_sum": "ni.", "nij_trg_sum": "n.j"}, inplace = True)
    table["n.."] = table["nij"].sum()
    table["mean_prior_probability"] = ((table["ni."] * table["n.j"]) / table["n.."]) * (1 / table["n.."])
    if calculate_p_value:
        table["score"] = binom.cdf(table["nij"], table["n.."], table["mean_prior_probability"])
        return table[["src", "trg", "nij", "score"]]
    table["kappa"] = table["n.."] / (table["ni."] * table["n.j"])
    table["score"] = ((table["kappa"] * table["nij"]) - 1) / ((table["kappa"] * table["nij"]) + 1)
    table["var_prior_probability"] = (1 / (table["n.."] ** 2)) * (table["ni."] * table["n.j"] * (table["n.."] - table["ni."]) * (table["n.."] - table["n.j"])) / ((table["n.."] ** 2) * ((table["n.."] - 1)))
    table["alpha_prior"] = (((table["mean_prior_probability"] ** 2) / table["var_prior_probability"]) * (1 - table["mean_prior_probability"])) - table["mean_prior_probability"]
    table["beta_prior"] = (table["mean_prior_probability"] / table["var_prior_probability"]) * (1 - (table["mean_prior_probability"] ** 2)) - (1 - table["mean_prior_probability"])
    table["alpha_post"] = table["alpha_prior"] + table["nij"]
    table["beta_post"] = table["n.."] - table["nij"] + table["beta_prior"]
    table["expected_pij"] = table["alpha_post"] / (table["alpha_post"] + table["beta_post"])
    table["variance_nij"] = table["expected_pij"] * (1 - table["expected_pij"]) * table["n.."]
    table["d"] = (1.0 / (table["ni."] * table["n.j"])) - (table["n.."] * ((table["ni."] + table["n.j"]) / ((table["ni."] * table["n.j"]) ** 2)))
    table["variance_cij"] = table["variance_nij"] * (((2 * (table["kappa"] + (table["nij"] * table["d"]))) / (((table["kappa"] * table["nij"]) + 1) ** 2)) ** 2) 
    table["sdev_cij"] = table["variance_cij"] ** .5
    if not return_self_loops:
        table = table[table["src"] != table["trg"]]
    if undirected:
        table = table[table["src"] <= table["trg"]]
    return table[["src", "trg", "nij", "score", "sdev_cij"]]

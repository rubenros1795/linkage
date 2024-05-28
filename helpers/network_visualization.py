import igraph
import matplotlib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


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

def mutual_information(theta, topn=None):
    """
    theta (np.array): numpy array with rows as document-topic mixtures:
    
    returns:
    R_ij: np.array of linkage scores as measured with mutual information
    """
    if topn:
        theta = np.where(np.argsort(np.argsort(theta)) >= theta.shape[1]-topn, theta, 0.0000000001)

    p_ij = theta[:,:,None] * theta[:,None,:]
    p_ij = p_ij.sum(axis=0) / p_ij.sum()
    pt_i = theta.sum(axis=0)
    pt_i = pt_i / pt_i.sum()
    R_ij = np.log2(p_ij / (np.outer(pt_i.ravel(), pt_i.ravel())))
    ptj_i = p_ij / pt_i
    Ri = (R_ij * ptj_i).sum(axis=0)
    M = (Ri * pt_i).sum(axis=0)  
    Mm = (Ri * pt_i)
    return R_ij, Ri, M

def diachronic_linkage(dict_date_theta):
    """
    dict_day_theta: a dictionary with dates as keys and associated np.arrays with doc-topic mixtures as values.
    """

    stats = []
    mi_arrays = {}

    for date, theta in tqdm(dict_date_theta.items()):
        mi_, r, m = mutual_information(theta=theta)
        stats.append({"date":date,"mi_mu":mi_.mean(),"mi_sigma":mi_.std(),"r_mu":r.mean(),"r_sigma":r.std(),"m":m})
        mi_arrays.update({date:mi_})
    return stats, mi_arrays

def flatten_mi_array(theta):
    m,n = theta.shape
    theta[:] = np.where(np.arange(m)[:,None] >= np.arange(n),np.nan,theta)
    ta_ = theta.flatten()
    ix = np.array(list(np.ndindex(theta.shape)))
    na = np.column_stack((ix[:,0],ix[:,1],ta_))
    return pd.DataFrame(na).dropna()

def default_style(g,edge_color='grey',node_color='darkgrey'):
    visual_style = dict()
    visual_style["edge_curved"] = False
    # visual_style["edge_color"] = [igraph.color_name_to_rgba(edge_color) for e in g.es] if isinstance(edge_color,str) else edge_color
    visual_style["vertex_label"] = g.vs["name"]
    visual_style['vertex_color'] = [igraph.color_name_to_rgba(node_color) for e in g.vs] if isinstance(node_color,str) else node_color
    
    return visual_style

def style_community_patches(ax,edge_color='whitesmoke',face_color='lightgrey',alpha=.25):
    for i in ax.get_children():
        if isinstance(i,matplotlib.patches.PathPatch):
            i.set_edgecolor(igraph.color_name_to_rgba(edge_color))
            i.set_facecolor(igraph.color_name_to_rgba(face_color))
            i.set_alpha(alpha)

def style_nodes(ax,edge_color='dimgrey',face_color='grey',alpha=.25):
    for i in ax.get_children():
        if isinstance(i,matplotlib.patches.Circle):
            i.set_edgecolor(igraph.color_name_to_rgba(edge_color))
            i.set_facecolor(igraph.color_name_to_rgba(face_color))
            i.set_alpha(alpha)

def style_text(ax,color='black',alpha=.25):
    for i in ax.get_children():
        if isinstance(i,matplotlib.text.Text):
            i.set_color(igraph.color_name_to_rgba(color))
            i.set_alpha(alpha)

## Full plotters

def plot_network(dists_subset,ax,labels,weight_threshold,df_filter,title,show_labels=False,top_btw_color=3,topn=None):
    """
    dists_subset: speech-topic distributions
    ax: plt. Axes
    labels: dict of index-label pairs
    weight_threshold: minimum weight, commonly a minimum of 0
    df_filter: threshold for disparity filter, usually .9 or .95 works best
    title: subplot title
    no_labels: remove labels or not
    top_btw_color: number of top betweenness nodes to color
    """

    theta,_,__ = mutual_information(dists_subset,topn)
    g,comm,_ = get_network_from_mi_theta(mi_theta=theta,node_text='labels',labels=labels,dis_filter=df_filter,weight_threshold=weight_threshold)
    betweenness_values = get_betweenness_values(g)
    
    if top_btw_color not in [None,0]:
        grey_pal_light_dark = sns.color_palette('Greys',4)
        visual_style = dict()
        top_btw = Counter(dict(betweenness_values)).most_common(top_btw_color)
        visual_style["vertex_size"] = [.5 if e['name'] not in dict(top_btw).keys() else 1 for e in g.vs]
        visual_style['vertex_color'] = [grey_pal_light_dark[-1] if e['name'] not in dict(top_btw).keys() else 'red' for e in g.vs]
   
    else:
        visual_style = dict()
        visual_style["vertex_size"] = 1
        visual_style['vertex_color'] = grey_pal_light_dark[-1] 
        
    ax.set_title(title,fontsize=(13))
    layout = g.layout_fruchterman_reingold()
    igraph.plot(comm,mark_groups = True,target=ax,layout=layout,**visual_style)

    for i in ax.get_children():
        if isinstance(i,matplotlib.patches.PathPatch):
            i.set_edgecolor(grey_pal_light_dark[1])
            i.set_facecolor(grey_pal_light_dark[0])
        elif isinstance(i,matplotlib.text.Text):
            if len(i.get_text()) != 4 and i.get_text() != title:
                i.set_alpha(0)
            if show_labels == True:
                i.set_alpha(1)


def plot_speaker(ax,data,dists,labels,id='nl.m.01611',min_date=None,max_date=None,remove_nonsem=True,topic_prop_threshold=1,df_threshold=.95,title='',topn=None):
    
    """
    ax: plt. Axes
    data: DataFrame with metadata (and member-ref column)
    dists: topic-speech distribution in DataFrame format
    labels: dict of index-label pairs
    id: speaker id
    min_date: timestamp, pd.Timestamp(year=1945,month=1,day=1)
    max_date: timestamp, pd.Timestamp(year=1945,month=1,day=1)
    remove_nonsem: whether to remove rhet/proc/nonsem topics
    topic_prop_threshold: threshold for dropping topic irrelevant to a speaker, usually in range .1-5
    df_threshold: threshold for disparity filter, usually .9 or .95 works best
    """

    labels_reverse =dict(zip(labels.values(),labels.keys()))
    indices_nonsem = [k for k,v in labels.items() if 'rhet' in v or 'proc' in v]

    data_speaker = data[data['member-ref'] == id]
    if min_date != None and max_date != None:
        mask = (data.date > min_date) & (data.date <= max_date)
        data_speaker = data_speaker.loc[mask]
    speaker_theta = dists.reset_index().loc[data_speaker.index].set_index('date')

    # Get Normalized Topic Proportions for the speaker
    speaker_theta_prop = speaker_theta.mean(axis=0).reset_index()
    speaker_theta_prop.columns = ['topic_index','topic_prop_speaker']
    dists = dists.mean(axis=0).reset_index()
    dists.columns = ['topic_index','topic_prop']
    speaker_theta_prop = pd.merge(dists, speaker_theta_prop, on='topic_index', how='outer')
    speaker_theta_prop['norm_topic_prop_speaker'] = speaker_theta_prop['topic_prop_speaker'] / speaker_theta_prop['topic_prop']
    speaker_prop = dict(zip(speaker_theta_prop.topic_index,speaker_theta_prop.norm_topic_prop_speaker))

    speaker_topic_selection = [k for k,v in speaker_prop.items() if v > 1]

    theta = speaker_theta.to_numpy()
    theta,_,__ = mutual_information(theta)
    theta_flat = flatten_mi_array(theta)
    theta_flat.columns = ['source','target','weight']

    if remove_nonsem == True:
        theta_flat = theta_flat[~theta_flat.source.isin(indices_nonsem)]
        theta_flat = theta_flat[~theta_flat.target.isin(indices_nonsem)]
    if topic_prop_threshold != None:
        theta_flat = theta_flat[~theta_flat.source.isin(speaker_topic_selection)]
        theta_flat = theta_flat[~theta_flat.target.isin(speaker_topic_selection)]

    g,comm,_ = get_network_from_edge_dataframe(edge_df=theta_flat,dis_filter=df_threshold,weight_threshold=0,node_text='labels',labels=labels)
    layout = g.layout_kamada_kawai()

    visual_style = default_style(g)
    visual_style['vertex_size'] = [speaker_prop[labels_reverse[i['name']]] * .75 for i in g.vs]
    visual_style['label_size'] = [speaker_prop[labels_reverse[i['name']]] * .75 for i in g.vs]
    visual_style['bbox'] = (1200,400)
    igraph.plot(comm,mark_groups = True,target=ax,layout=layout,**visual_style)

    style_community_patches(ax)
    style_nodes(ax,alpha=1)

    for i in ax.get_children():
        if isinstance(i,matplotlib.text.Text):
            if i.get_text() in labels_reverse.keys():
                if labels_reverse[i.get_text()] in speaker_prop.keys():
                    i.set_fontsize(speaker_prop[labels_reverse[i.get_text()]] * 15)
                
    ax.set_title(title)
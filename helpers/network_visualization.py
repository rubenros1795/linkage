import igraph
import matplotlib
from .linkage import mutual_information, flatten_mi_array
from .networks import disparity_filter, get_network_from_mi_theta, get_betweenness_values, get_network_from_edge_dataframe
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

def plot_network(dists_subset,ax,labels,weight_threshold,df_filter,title,show_labels=False,top_btw_color=3):
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

    theta,_,__ = mutual_information(dists_subset)
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


def plot_speaker(ax,data,dists,labels,id='nl.m.01611',min_date=None,max_date=None,remove_nonsem=True,topic_prop_threshold=1,df_threshold=.95,title=''):
    
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
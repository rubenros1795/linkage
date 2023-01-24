import igraph
import matplotlib

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


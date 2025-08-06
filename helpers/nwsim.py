import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigvalsh
from time import sleep
from tqdm.notebook import tqdm
import seaborn as sb
import matplotlib.pyplot as plt
import scipy
import random
import time


# # Similarity Measures

# In[5]:


sim_dict = dict()


# ## 3.1.1 Jaccard index

# In[6]:


from iteration_utilities import first

def jaccard(G1,G2):
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    return len(E1&E2)/len(E1|E2)

def jaccard_weighted(G1,G2, attr="weight"):
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    if (len(E1)>0 and attr in G1[first(E1)[0]][first(E1)[1]]) or (len(E2)>0 and attr in G2[first(E2)[0]][first(E2)[1]]):
        E_overlap = E1&E2
        n1, n2 = 0, 0
        for edge in E_overlap:
            n1+=min(G1[edge[0]][edge[1]][attr], G2[edge[0]][edge[1]][attr])
            n2+=max(G1[edge[0]][edge[1]][attr], G2[edge[0]][edge[1]][attr])
        E1u, E2u = E1 - E_overlap, E2 - E_overlap
        for edge in E1u:
            n2+=G1[edge[0]][edge[1]][attr]
        for edge in E2u:
            n2+=G2[edge[0]][edge[1]][attr]
        if n2 != 0:
            return n1/n2
        else:
            return None
    else:
        return None


# ## 3.1.2 Graph Edit Distance

# In[8]:


def ged(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    sim = len(V1)+len(V2)+len(E1)+len(E2)-2*(len(V1&V2) + len(E1&E2))
    return sim


# In[9]:


def ged_norm(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    sim = 1 - (len(V1&V2) + len(E1&E2))/(len(V1|V2) + len(E1|E2))
    return sim


# ## 3.1.3 Vertex-Edge Overlap

# In[11]:


def vertex_edge_overlap(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    V_overlap, E_overlap = V1&V2, E1&E2
    sim = 2*(len(V_overlap) + len(E_overlap)) / (len(V1)+len(V2)+len(E1)+len(E2))
    return sim


# ## 3.1.4 k-hop Nodes Neighborhood

# In[13]:


def get_neigbors(G, v, k):
    neighbors = set()
    k_hop_n = []
    for i in range(k):
        if i==0:
            k_hop_n.append(set(G.neighbors(v)))
            neighbors |= set(k_hop_n[i])
        else:
            nn = set()
            for node in k_hop_n[i-1]:
                nn |= set(G.neighbors(node))
            k_hop_n.append(set(nn - neighbors - set([v])))
            neighbors |= k_hop_n[i]
    return neighbors


def nodes_neighborhood(G1,G2, k):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    V_overlap = V1&V2
    sim = 0
    for v in V_overlap:
        n1, n2 = get_neigbors(G1, v, k), get_neigbors(G2, v, k)
        if len(n1|n2)>0:
            sim+=len(n1&n2)/len(n1|n2)
    return sim/(len(V1|V2))


# ## 3.1.5 Maximum Common Subgraph Distance (MCS)

# In[15]:


def getMCS(g1, g2):
    matching_graph=nx.Graph()
    mcs = 0

    for n1,n2 in g2.edges():
        if g1.has_edge(n1, n2):
            matching_graph.add_edge(n1, n2)
    if not nx.is_empty(matching_graph):
        components = (nx.connected_components(matching_graph))
        largest_component = max(components, key=len)
        mcs = len(largest_component)
    return mcs


# ## 3.1.7 Vector Similarity Algorithm

# In[18]:


from scipy.stats import rankdata

def construct_vsgraph(G):
    Gm = nx.DiGraph()
    q = nx.pagerank(G)
    label = 'weight' if nx.is_weighted(G) else None
    degree = {}
    if isinstance(G, nx.Graph):
        degree = G.degree(weight=label)
    else:
        degree = G.out_degree(weight=label)

    #reconstruct graph
    E = list(G.edges(data=True))
    for edge in E:
        w = 1 if label is None else edge[2][label]
        if isinstance(G, nx.Graph):
            Gm.add_edge(edge[0], edge[1], weight=w*q[edge[0]]/degree[edge[0]])
            Gm.add_edge(edge[1], edge[0], weight=w*q[edge[1]]/degree[edge[1]])
        else:
            Gm.add_edge(edge[0], edge[1], weight=w*q[edge[0]]/degree[edge[0]])
    return Gm


def compare_graph_weghts(G1,G2, attr="weight"):
    E1,E2 = [set(G.edges()) for G in [G1,G2]]
    E_union = E1|E2
    sim = 0
    for edge in E_union:
        if G1.has_edge(*edge):
            if G2.has_edge(*edge):
                sim+=abs(G1[edge[0]][edge[1]][attr]-G2[edge[0]][edge[1]][attr])/max(G1[edge[0]][edge[1]][attr],G2[edge[0]][edge[1]][attr])
            else:
                sim+=1
        else:
            sim+=1
    return sim/len(E_union)


def vector_similarity(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    V_overlap = V1&V2
    num = len(V1|V2)
    G1m, G2m = construct_vsgraph(G1), construct_vsgraph(G2)
    return 1-compare_graph_weghts(G1m, G2m)


# ## 3.1.12 Vertex Ranking

# In[25]:


from scipy.stats import rankdata

def vertex_ranking(G1,G2):
    V1,V2 = [set(G.nodes()) for G in [G1,G2]]
    V_overlap = V1&V2
    num = len(V1|V2)
    #find maximal denominator
    d = 0
    for i in range(num):
        d += (i-(num-i-1))**2

    #compute centralities
    pi1,pi2 = nx.pagerank(G1), nx.pagerank(G2)
    for v in (V1-V_overlap):
        pi2[v]=0
    for v in (V2-V_overlap):
        pi1[v]=0
    ranks1, ranks2 = rankdata(list(pi1.values()), method='min'),rankdata(list(pi2.values()), method='min')

    #compute similarity
    sim = 0
    for i in range(num):
        sim+=(ranks1[i]-ranks2[i])**2

    return 1-2*sim/d


# ## 3.1.13 Degree Jenson-Shannon divergence

# In[27]:


from collections import Counter

def degree_vector_histogram(graph):
        """Return the degrees in both formats.

        max_deg is the length of the histogram, to be padded with
        zeros.

        """
        vec = np.array(list(dict(graph.degree()).values()))
        if next(nx.selfloop_edges(graph), False):
            max_deg = len(graph)
        else:
            max_deg = len(graph) - 1
        counter = Counter(vec)
        hist = np.array([counter[v] for v in range(max_deg+1)])
        return vec, hist


# def degreeJSD(G1,G2):
#     deg1, hist1 = degree_vector_histogram(G1)
#     deg2, hist2 = degree_vector_histogram(G2)
#     max_len = max(len(hist1), len(hist2))
#     p1 = np.pad(hist1, (0, max_len - len(hist1)), 'constant', constant_values=0)
#     p2 = np.pad(hist2, (0, max_len - len(hist2)), 'constant', constant_values=0)
#     if sum(hist1)>0:
#         p1 = p1/sum(p1)
#     if sum(hist2)>0:
#         p2 = p2/sum(p2)
#     return netrd.utilities.entropy.js_divergence(p1,p2)**(1/2)



# ## 3.1.15 Communicability Sequence Entropy

# In[29]:


def create_comm_matrix(C, dictV):
    N = len(dictV)
    Ca = np.zeros((N, N))
    for v in C:
        for v2 in C[v]:
            Ca[dictV[v]][dictV[v2]] = C[v][v2]
    return Ca


# def CommunicabilityJSD(G1, G2):
#     dist = 0
#     V1,V2 = [set(G.nodes()) for G in [G1,G2]]
#     N1, N2 = len(V1), len(V2)
#     V = list(V1|V2)
#     N = len(V)
#     dictV = dict(zip(V, list(range(N))))

#     Ca1 = create_comm_matrix(nx.communicability_exp(G1), dictV)
#     Ca2 = create_comm_matrix(nx.communicability_exp(G2), dictV)

#     lil_sigma1 = np.triu(Ca1).flatten()
#     lil_sigma2 = np.triu(Ca2).flatten()

#     big_sigma1 = sum(lil_sigma1[np.nonzero(lil_sigma1)[0]])
#     big_sigma2 = sum(lil_sigma2[np.nonzero(lil_sigma2)[0]])

#     P1 = lil_sigma1 / big_sigma1
#     P2 = lil_sigma2 / big_sigma2
#     P1 = np.array(sorted(P1))
#     P2 = np.array(sorted(P2))

#     dist = netrd.utilities.entropy.js_divergence(P1, P2)
#     return dist


# ## 3.1.19 λ-distances

# In[34]:


def lambda_distances(G1, G2):
        d=dict()
        labels = ["λ-d Adj.","λ-d Lap.","λ-d N.L."]
        # Get adjacency matrices
        try:
            A1, A2 = nx.to_numpy_array(G1), nx.to_numpy_array(G2)
            # List of matrix variations
            l1 = [A1, laplacian(A1), laplacian(A1, normed=True)]
            l2 = [A2, laplacian(A2), laplacian(A2, normed=True)]

            for l in range(len(l1)):
                ev1 = np.abs(eigvalsh(l1[l]))
                ev2 = np.abs(eigvalsh(l2[l]))
                d[labels[l]]= np.linalg.norm(ev1 - ev2)
        except Exception:
            for l in range(len(labels)):
                d[labels[l]]= "None"
        return d


# ## 3.1.26 Signature similarity (SS)

# In[44]:


import hashlib
from scipy.spatial.distance import hamming

def compute_features(G):
    features = []
    q = nx.pagerank(G)
    Gm = construct_vsgraph(G)

    for row in q:
        features.append([str(row), q[row]])
        #features[(str(row))]=q[row]

    E = set(Gm.edges())
    for edge in E:
        t = str(edge[0])+"_"+str(edge[1])
        w = Gm[edge[0]][edge[1]]['weight']
        #features[t]=w
        features.append([t, w])
    return features

def encode_fingerprint(text):
    return bin(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16))[2:]

def text2hash(h):
    for i in range(len(h)):
        h[i][0]=encode_fingerprint(h[i][0])
    return h

def get_fingerprint(h):
    arr = np.zeros(128)
    for i in range(len(h)):
        for j in range(len(h[i][0])):
            w = h[i][1] if h[i][0][j]=='1' else -h[i][1]
            arr[j] += w
    for j in range(len(arr)):
        if arr[j]>0:
            arr[j]=1
        else:
            arr[j]=0
    return arr



def signature_similarity(G1, G2):
    h1 = get_fingerprint(text2hash(compute_features(G1)))
    h2 = get_fingerprint(text2hash(compute_features(G2)))
    dist = 1 - hamming(h1,h2)/len(h1)

    return dist



# ## 3.1.28 LD-measure

# In[47]:


def get_transition_distr(G, node, v_dict):
    arr = np.zeros(len(v_dict))
    neighbors = list(G.neighbors(node))
    if len(neighbors)>0:
        for el in G.neighbors(node):
            arr[v_dict[el]] = 1
        arr =  len(neighbors)
    return arr


def node_distance(G):
    """
    Return an NxN matrix that consists of histograms of shortest path
    lengths between nodes i and j. This is useful for eventually taking
    information theoretic distances between the nodes.

    Parameters
    ----------
    G (nx.Graph): the graph in question.

    Returns
    -------
    out (np.ndarray): a matrix of binned node distance values.

    """

    N = G.number_of_nodes()
    a = np.zeros((N, N))

    dists = nx.shortest_path_length(G)
    for idx, row in enumerate(dists):
        counts = Counter(row[1].values())
        a[idx] = [counts[l] for l in range(1, N + 1)]

    return a / (N - 1)

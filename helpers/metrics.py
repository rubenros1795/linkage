import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx

def mutual_information_smooth(theta):
    """
    theta (np.array): numpy array with rows as document-topic mixtures.

    Returns:
    R_ij: np.array of linkage scores as measured with mutual information.
    Ri: np.array of average mutual information for each topic.
    M: float, overall mutual information.
    """
    import math

    def _log2(n):
        return math.log2(n)

    def pmi_delta_score(ab, a, b, cz, factor):
        weight = (ab / (ab + 1)) * (min(a, b) / (min(a, b) + 1))
        return factor * weight * (_log2(ab * cz) - _log2(a * b))

    # p_ij = P(t_i, t_j) - joint probability of topics t_i and t_j
    p_ij = theta[:,:,None] * theta[:,None,:]
    p_ij = p_ij.sum(axis=0) / p_ij.sum()

    # pt_i = P(t_i) - marginal probability of topic t_i
    pt_i = theta.sum(axis=0)
    pt_i = pt_i / pt_i.sum()

    # Calculate the corpus size and the factor
    cz = theta.sum()
    factor = 1  # Assuming factor is 1 if not specified

    # Calculate R_ij using PMI-DELTA score
    R_ij = np.zeros_like(p_ij)
    for i in range(p_ij.shape[0]):
        for j in range(p_ij.shape[1]):
            if pt_i[i] > 0 and pt_i[j] > 0 and p_ij[i, j] > 0:
                ab = p_ij[i, j] * cz
                a = pt_i[i] * cz
                b = pt_i[j] * cz
                R_ij[i, j] = pmi_delta_score(ab, a, b, cz, factor)

    # ptj_i = P(t_j|t_i) - conditional probability of topic t_j given topic t_i
    ptj_i = p_ij / pt_i

    # Ri = Average mutual information for each topic
    Ri = (R_ij * ptj_i).sum(axis=0)

    # M = Overall mutual information
    M = (Ri * pt_i).sum()

    return R_ij, Ri, M


def shannon_entropy(p):
    """Calculates shannon entropy in bits.
    Parameters
    ----------
    p : np.array
        array of probabilities
    Returns
    -------
    shannon entropy in bits
    """
    return -np.sum(np.where(p!=0, p * np.log2(p), 0))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def kld(p, q):
    """ KL-divergence for two probability distributions
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, (p-q) * np.log10(p / q), 0))


def jsd(p, q, base=2):
    '''Pairwise Jensen-Shannon Divergence for two probability distributions
    '''
    # convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    # normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return stats.entropy(p, m, base=base)/2. + stats.entropy(q, m, base=base)/2.


def cosine_distance(p, q):
    '''Cosine distance for two vectors
    '''

    p, q = np.asarray(p), np.asarray(q)

    dot_prod = np.dot(p, q)
    magnitude = np.sqrt(p.dot(p)) * np.sqrt(q.dot(q))
    cos_sim = dot_prod / magnitude
    cos_dist = 1 - cos_sim

    return cos_dist

def make_foote(quart=5):
    tophalf = [-1] * quart + [1] * quart
    bottomhalf = [1] * quart + [-1] * quart
    foote = list()
    for i in range(quart):
        foote.append(tophalf)
    for i in range(quart):
        foote.append(bottomhalf)
    foote = np.array(foote)
    return foote

def foote_novelty(distdf, foote_size=5):
    foote=make_foote(foote_size)
    distmat = distdf.values if type(distdf)==pd.DataFrame else distdf

    axis1, axis2 = distmat.shape
    assert axis1 == axis2
    distsize = axis1
    axis1, axis2 = foote.shape
    assert axis1 == axis2
    halfwidth = axis1 / 2
    novelties = []
    for i in range(distsize):
        start = int(i - halfwidth)
        end = int(i + halfwidth)
        if start < 0 or end > (distsize - 1):
            novelties.append(0)
        else:
            novelties.append(np.sum(foote * distmat[start: end, start: end]))
    return novelties

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

## Modularity & Density
def to_edge_list(rij):
    # rij is a 2D NumPy array
    edges = []
    n = rij.shape[0]
    for i in range(n):
        for j in range(n):
            w = rij[i, j]
            if w > 0:
                edges.append((i, j, w))
    return edges

def modularity(v):
    rij, ri, m = mutual_information_smooth(v)
    edf = pd.DataFrame(rij).stack().reset_index()
    edf = edf[edf[0] > 0]
    G = nx.from_pandas_edgelist(df = edf, source = 'level_0', target = 'level_1', edge_attr=0)
    comms = nx.community.louvain_communities(G,weight=0,resolution=2.5)
    return nx.community.quality.modularity(G,communities=comms)

## Density
def density(v):
    rij, ri, m = mutual_information_smooth(v)
    edf = pd.DataFrame(rij).stack().reset_index()
    edf = edf[edf[0] > 0]
    G = nx.from_pandas_edgelist(df = edf, source = 'level_0', target = 'level_1', edge_attr=0)
    return nx.density(G)

'''
Relative entropy measures
    taken from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/tekisuto/metrics/entropies.py
    commit 1fb16bc91b99716f52b16100cede99177ac75f55
'''

import numpy as np
import pandas as pd
from scipy import stats

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
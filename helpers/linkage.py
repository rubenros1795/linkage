import numpy as np
from tqdm import tqdm
import pandas as pd
from numpy import inf


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
    return R_ij, Ri, M

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
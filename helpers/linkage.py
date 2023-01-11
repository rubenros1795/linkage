import numpy as np
from tqdm import tqdm
import pandas as pd

def mutual_information(theta):
    """
    theta (np.array): numpy array with rows as document-topic mixtures:
    
    returns:
    R_ij: np.array of linkage scores as measured with mutual information
    """
    p_ij = theta[:,:,None] * theta[:,None,:]
    p_ij = p_ij.sum(axis=0) / p_ij.sum()
    pt_i = theta.sum(axis=0)
    pt_i = pt_i / pt_i.sum()
    R_ij = np.log2(p_ij / (np.outer(pt_i.ravel(), pt_i.ravel())))
    ptj_i = p_ij / pt_i
    Ri = (R_ij * ptj_i).sum(axis=0)
    M = (Ri * pt_i).sum(axis=0)  
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
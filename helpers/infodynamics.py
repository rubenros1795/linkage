"""
Class for estimation of information dynamics of time-dependent probabilistic document representations
    taken from https://github.com/centre-for-humanities-computing/newsFluxus/blob/master/src/tekisuto/models/infodynamics.py
    commit 1fb16bc91b99716f52b16100cede99177ac75f55
"""
import json
import numpy as np
from scipy import stats
import pandas as pd
from tqdm import tqdm

def diachronic_correlations_lookback(timestamps,theta,scale):
    mean_correlations = []
    for i in tqdm(range(scale,theta.shape[0])):
        lookback_window = theta[i:i+scale,:]
        m = np.array([np.correlate(theta[i,:],r) for r in lookback_window]).mean()
        mean_correlations.append(m)
    cdf = pd.DataFrame(list(zip(timestamps[scale:],mean_correlations)),columns=['date','corr'])
    return cdf


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

class InfoDynamics:
    def __init__(self, data, time, window=3, weight=0, sort=False, normalize=False):
        """
        - data: list/array (of lists), bow representation of documents
        - time: list/array, time coordinate for each document (identical order as data)
        - window: int, window to compute novelty, transience, and resonance over
        - weight: int, parameter to set initial window for novelty and final window for transience
        - sort: bool, if time should be sorted in ascending order and data accordingly
        - normalize: bool, make row sum to 1
        """
        self.window = window
        self.weight = weight

        if sort:
            self.data = np.array([text for _,text in sorted(zip(time, data))])
            self.time = sorted(time)
        else:
            self.data = np.array(data)
            self.time = time

        self.m = self.data.shape[0]
    
        if normalize:
            data = data / data.sum(axis=1, keepdims=True)
        
    def novelty(self, meas=kld):
        N_hat = np.zeros(self.m)
        N_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[(i - self.window):i,]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window]) + self.weight
            
            N_hat[i] = np.mean(tmp)
            N_sd[i] = np.std(tmp)

        self.nsignal = N_hat
        self.nsigma = N_sd
    
    def transience(self, meas=kld):
        T_hat = np.zeros(self.m)
        T_sd = np.zeros(self.m)
        for i, x in enumerate(self.data):
            submat = self.data[i+1:(i + self.window + 1),]
            tmp = np.zeros(submat.shape[0])
            if submat.any():
                for ii, xx in enumerate(submat):
                    tmp[ii] = meas(x, xx)
            else:
                tmp = np.zeros([self.window])
            
            T_hat[i] = np.mean(tmp)
            T_hat[-self.window:] = np.zeros([self.window]) + self.weight
            T_sd[i] = np.std(tmp)
        
        self.tsignal = T_hat  
        self.tsigma = T_sd

    def resonance(self, meas=kld):
        self.novelty(meas)
        self.transience(meas)
        self.rsignal = self.nsignal - self.tsignal
        self.rsignal[:self.window] = np.zeros([self.window]) + self.weight
        self.rsignal[-self.window:] = np.zeros([self.window]) + self.weight
        self.rsigma = (self.nsigma + self.tsigma) / 2
        self.rsigma[:self.window] = np.zeros([self.window]) + self.weight
        self.rsigma[-self.window:] = np.zeros([self.window]) + self.weight

    def slice_zeros(self):
        self.nsignal = self.nsignal[self.window:-self.window]
        self.nsigma = self.nsigma[self.window:-self.window]
        self.tsignal = self.tsignal[self.window:-self.window]
        self.tsigma = self.tsigma[self.window:-self.window]
        self.rsignal = self.rsignal[self.window:-self.window]
        self.rsigma = self.rsigma[self.window:-self.window]

    def fit(self, meas, slice_w=False):
        self.novelty(meas)
        self.transience(meas)
        self.resonance(meas)
        if slice_w:
            self.slice_zeros()
            
        out = {
            'novelty': self.nsignal.tolist(),
            'novelty_sigma': self.nsigma.tolist(),
            'transience': self.tsignal.tolist(),
            'transience_sigma': self.tsigma.tolist(),
            'resonance': self.rsignal.tolist(),
            'resonance_sigma': self.rsigma.tolist(),
        }
        
        self.results = out

    def fit_save(self, meas, path, slice_w=False):
        self.fit(meas, slice_w)

        out = {
            'novelty': self.nsignal.tolist(),
            'novelty_sigma': self.nsigma.tolist(),
            'transience': self.tsignal.tolist(),
            'transience_sigma': self.tsigma.tolist(),
            'resonance': self.rsignal.tolist(),
            'resonance_sigma': self.rsigma.tolist(),
        }

        with open(path, 'w') as f:
            json.dump(out, f)

    def fit_return(self, meas, slice_w=False):
        self.fit(meas, slice_w)

        out = {
            'novelty': self.nsignal.tolist(),
            'novelty_sigma': self.nsigma.tolist(),
            'transience': self.tsignal.tolist(),
            'transience_sigma': self.tsigma.tolist(),
            'resonance': self.rsignal.tolist(),
            'resonance_sigma': self.rsigma.tolist(),
        }

        return out
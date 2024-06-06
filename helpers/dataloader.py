import pandas as pd
import numpy as np
import os
from scipy.stats import zscore
from helpers.metrics import softmax

def load(lda_path = '/home/rb/Documents/Data/models/lda/postwar-v3/',
         topic_mb_agg = True,
         time_agg = "6M",
         filter_thematic = True,
         zscore_filter = True,
         plenary_filter = True,
         quick_return = False
         ):

    dists = pd.read_csv(os.path.join(lda_path, 'dist.tsv'),sep='\t')
    dat = pd.read_csv(os.path.join(lda_path, 'data.tsv'),sep='\t',parse_dates=['date'])
    topic_dates = dict(zip(dat.topic_id,dat.date))
    topic_sesst = dict(zip(dat.topic_id,dat.sess_type))
    keys = pd.read_csv(os.path.join(lda_path, 'keys.tsv'),sep='\t')
    keys = dict(zip(keys.ix,keys.label))
    
    if quick_return == True:
        return dists, dat, keys

    # Average Member Speeches per Session
    if topic_mb_agg == True:
        dists = dists.groupby(dat[['topic_id','member-ref']].astype(str).agg('_'.join,axis=1)).mean()
        dists.index = dists.index.str.split('_').str[0]
    else:
        dists.index = dat['topic_id']

    # Filter Plenary sessions
    if plenary_filter == True:
        dists = dists[dists.index.map(topic_sesst) == 'plenary']
        dat = dat[dat.sess_type == 'plenary']

    # Aggregate on time period
    if time_agg == '6M':
        topic_dates = {topic:pd.Timestamp(year = _.year, month = 1 if _.month < 7 else 6, day = 1) for topic,_ in topic_dates.items()}
    elif time_agg == '1M':
        topic_dates = {topic:pd.Timestamp(year = _.year, month = _.month, day = 1) for topic,_ in topic_dates.items()}
    if time_agg == 'Y':
        topic_dates = {topic:_.year for topic,_ in topic_dates.items()}
    else:
        topic_dates = topic_dates

    dists.index = dists.index.map(topic_dates)

    # Filter Thematic Topics (and normalize again)
    ## Because MI function resets column indices, save original link between topic id - labels
    if filter_thematic:
        dists = dists[[v for v in dists.columns if 'rhet' not in keys[int(v)] and 'nonse' not in keys[int(v)] and 'proc' not in keys[int(v)]]]
        dists = dists.div(dists.sum(axis=1), axis=0)
        coltrans = {c:keys[int(cc)] for c,cc in enumerate(dists.columns)}

    if zscore_filter:
        # Keep only topics that have an above-average score (from the persp. of their diachronic evolution)
        dz = dists.apply(zscore,axis=0)
        dists = dists.where(dz >= 0, 0.0000000000001)
        dists = dists.div(dists.sum(axis=1), axis=0)
    
    return (dists, dat, coltrans, keys) if zscore_filter == False else (dists, dat, coltrans, keys, dz)
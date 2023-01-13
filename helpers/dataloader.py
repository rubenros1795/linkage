import pandas as pd
from top2vec import Top2Vec
from pathlib import Path
import yaml

cf = yaml.safe_load(Path('config.yml').read_text())

def load(cf=cf,lmodel=False,ldata=True,llabels=True,lwords=True,ldists=True,ltranslator=True):
    opd = dict()
    # Load model
    if lmodel == True:
        opd['model'] = Top2Vec.load(cf['model_path'])

    # Load Data
    if ldata == True:
        data = pd.read_csv(cf['data_path'],sep='\t',usecols=['date','speech_id','role','member-ref'])
        data['date'] = pd.to_datetime(data.date,infer_datetime_format=True)
        opd['data'] = data

    # Model Labels (to prune non-semantic, procedural and rhetorical topics)
    if llabels == True or lwords == True:
        keys = pd.read_csv(cf['keys_path'],sep='\t')
        labels = dict(zip(keys.topic_index,keys.label))
        words = dict(zip(keys.topic_index,keys.words))
        opd['labels'] = labels
        opd['words'] = words

    # Distributions (prune irrelevant topics)
    if ldists == True:
        dists = pd.read_csv(cf['dist_path_raw'],sep='\t',header=None)
        dists = dists[[t for t,l in labels.items() if l not in ['nonsem','rhetoric','proc']]]
        dists = dists.set_index(data.date)
        dists = dists + dists.min().abs()
        dists = dists.div(dists.sum(axis=1), axis=0)
        # Drop Chairs
        nonchair_indices = data[data.role != 'chair'].index
        dists = dists.reset_index().loc[nonchair_indices].set_index('date')
        opd['dists'] = dists

    # Create a dictionary with the indices of the semantic topics (for converting between pandas / numpy matrices)
    if ltranslator == True:
        sem_col_translator = {c:int(k) for c,k in enumerate(dists.columns)}
        opd['translator'] = sem_col_translator

    return opd

def load_lda(cf=cf,agg_level="speech",ldata=True,lwords=True,ldists=True):
    """
    cf: config file (yaml)
    agg_level: model type, aggregated on "speech" or "day"
    ldata: Bool, whether to load data
    lwords: Bool, whether to load topic keys
    ldists: Bool, whether to load topic-doc distributions
    """

    opd = dict()
    if ldata == True:
        data = pd.read_csv(cf[f'data_path_{agg_level}'],sep='\t')
        data['date'] = pd.to_datetime(data.date,infer_datetime_format=True)
        opd['data'] = data
    if lwords == True:
        keys = pd.read_csv(cf[f'lda_keys_path_{agg_level}'],sep='\t',header=None)
        keys.columns = ['i','size','words']
        words = dict(zip(keys.i,keys.words))
        opd['words'] = words
    if ldists == True:
        dists = pd.read_csv(cf[f'lda_dist_path_{agg_level}'],sep='\t',header=None).iloc[:,2:]
        dists.columns = range(len(dists.columns))
        dists = dists.set_index(data.date)
        dists = dists + dists.min().abs()
        dists = dists.div(dists.sum(axis=1), axis=0)
        opd['dists'] = dists
    return opd
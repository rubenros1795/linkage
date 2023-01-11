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
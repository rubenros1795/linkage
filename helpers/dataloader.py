import pandas as pd
import numpy as np
from helpers.metrics import softmax

def load(cf,model_type='lda',agg_level="speech",remove_labels=[]):
    """
    cf: config file (yaml)
    model_type: model to use, "lda" or "t2v"
    agg_level: aggregation level of the model, "speech" or "day"
    remove_labels: topic labels to exclude from distributions, either empty or ["rhet", "proc", "nonsem"]
    """

    opd = dict()

    # Load data
    data = pd.read_csv(cf[f'data_path_{agg_level}'],sep='\t')
    data['date'] = pd.to_datetime(data.date,infer_datetime_format=True)
    opd['data'] = data

    # Load keys and labels
    keys = pd.read_csv(cf[f'{model_type}_keys_path_{agg_level}'],sep='\t')
    words = dict(zip(keys.index,keys.words))
    labels = dict(zip(keys.index,keys.label))
    opd['words'] = words
    opd['labels'] = labels

    # Load dists
    dists = pd.read_csv(cf[f'{model_type}_dist_path_{agg_level}'],sep='\t',header=None)
    if model_type == 'lda':
        dists = dists.iloc[:,2:]
    dists.columns = range(len(dists.columns))

    if len(remove_labels) != 0:
        dists = dists[[t for t,l in labels.items() if l not in remove_labels]]
        sem_col_translator = {c:int(k) for c,k in enumerate(dists.columns)}
        opd['translator'] = sem_col_translator

    dists = dists.set_index(data.date)

    if model_type == 't2v':
        # Transform top2vec topic similarities to positive floats for normalization
        # dists = dists + dists.min().abs()
        dists = dists.to_numpy()
        dists = np.apply_along_axis(softmax,1,dists)
        dists = dists.div(dists.sum(axis=1), axis=0)
    opd['dists'] = dists

    return opd
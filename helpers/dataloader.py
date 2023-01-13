import pandas as pd
from pathlib import Path
import yaml

cf = yaml.safe_load(Path('config.yml').read_text())

def load_lda(cf=cf,agg_level="speech",remove_labels=[]):
    """
    cf: config file (yaml)
    agg_level: model type, aggregated on "speech" or "day"
    """

    opd = dict()

    # Load data
    data = pd.read_csv(cf[f'data_path_{agg_level}'],sep='\t')
    data['date'] = pd.to_datetime(data.date,infer_datetime_format=True)
    opd['data'] = data

    # Load keys and labels
    keys = pd.read_csv(cf[f'lda_keys_path_{agg_level}'],sep='\t')
    words = dict(zip(keys.index,keys.words))
    labels = dict(zip(keys.index,keys.label))
    opd['words'] = words
    opd['labels'] = labels

    # Load dists
    dists = pd.read_csv(cf[f'lda_dist_path_{agg_level}'],sep='\t',header=None).iloc[:,2:]
    dists.columns = range(len(dists.columns))

    if len(remove_labels) != 0:
        dists = dists[[t for t,l in labels.items() if l not in remove_labels]]
        sem_col_translator = {c:int(k) for c,k in enumerate(dists.columns)}
        opd['translator'] = sem_col_translator

    dists = dists.set_index(data.date)
    dists = dists + dists.min().abs()
    dists = dists.div(dists.sum(axis=1), axis=0)
    opd['dists'] = dists

    return opd

def load_top2vec(cf=cf,remove_labels=[]):
    """
    cf: config file (yaml)
    agg_level: model type, aggregated on "speech" or "day"
    """

    opd = dict()

    # Load data
    data = pd.read_csv(cf[f't2v_data_path'],sep='\t')
    data['date'] = pd.to_datetime(data.date,infer_datetime_format=True)
    opd['data'] = data

    # Load keys and labels
    keys = pd.read_csv(cf[f't2v_keys_path'],sep='\t')
    words = dict(zip(keys.index,keys.words))
    labels = dict(zip(keys.index,keys.label))
    opd['words'] = words
    opd['labels'] = labels

    # Load dists
    dists = pd.read_csv(cf[f't2v_dist_path_raw'],sep='\t',header=None)

    if len(remove_labels) != 0:
        dists = dists[[t for t,l in labels.items() if l not in remove_labels]]
        sem_col_translator = {c:int(k) for c,k in enumerate(dists.columns)}
        opd['translator'] = sem_col_translator

    dists = dists.set_index(data.date)
    dists = dists + dists.min().abs()
    dists = dists.div(dists.sum(axis=1), axis=0)

    # Drop Chairs (for old models where chairs are included in model, in new models, chairs are dropped)
    nonchair_indices = data[data.role != 'chair'].index
    dists = dists.reset_index().loc[nonchair_indices].set_index('date')
    opd['dists'] = dists

    return opd
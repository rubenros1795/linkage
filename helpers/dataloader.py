import pandas as pd
import numpy as np
import os
from scipy.stats import zscore
from helpers.metrics import softmax
import xml.etree.ElementTree as ET

def load(lda_path = '/home/rb/Documents/Data/models/lda/postwar-v3/',
         dist_path = None,
         dat_path = None,
         keys_path = None,
         topic_mb_agg = True,
         time_agg = "6M",
         filter_thematic = True,
         zscore_filter = True,
         quick_return = False
         ):

    dist_path = os.path.join(lda_path, 'dist.tsv') if dist_path == None else dist_path
    dat_path = os.path.join(lda_path, 'data.tsv') if dat_path == None else dat_path
    keys_path = os.path.join(lda_path, 'keys.tsv') if keys_path == None else keys_path

    dists = pd.read_csv(dist_path,sep='\t')
    dat = pd.read_csv(dat_path,sep='\t',usecols=['date','member-ref','topic_id'], parse_dates=['date'])
    topic_dates = dict(zip(dat.topic_id,dat.date))

    keys_df = pd.read_csv(keys_path,sep='\t')

    if 'policy_label' in keys_df.columns:
        keys = dict(zip(keys_df.ix,keys_df['keys']))
    elif 'policy_label' not in keys_df.columns and filter_thematic == True:
        print('no labels found, thematic filtering not possible.')
        return

    if quick_return == True:
        return dists, dat, keys_df

    # Average Member Speeches per Session
    if topic_mb_agg == True:
        dists = dists.groupby(dat[['topic_id','member-ref']].astype(str).agg('_'.join,axis=1)).mean()
        dists.index = dists.index.str.split('_').str[0]
    else:
        dists.index = dat['topic_id']

    # Aggregate on time period
    if time_agg == '6M':
        topic_dates = {topic:pd.Timestamp(year = _.year, month = 1 if _.month < 7 else 6, day = 1) for topic,_ in topic_dates.items()}
    elif time_agg == '1M':
        topic_dates = {topic:pd.Timestamp(year = _.year, month = _.month, day = 1) for topic,_ in topic_dates.items()}
    if time_agg == 'Y':
        topic_dates = {tid:pd.Timestamp(year=int(tid.split('.')[4][:4]),month=1,day=1) for tid in dat.topic_id}
    else:
        topic_dates = topic_dates

    dists.index = dists.index.map(topic_dates)

    # Filter Thematic Topics (and normalize again)
    # ---> Because MI function resets column indices, save original link between topic id - labels

    coltrans = None

    if filter_thematic:
        policy_topics = keys_df[keys_df.policy_label.astype(int)==1].ix.astype(str).tolist()
        dists = dists[[str(c) for c in dists.columns if str(c) in policy_topics]]
        dists = dists.div(dists.sum(axis=1), axis=0)
        coltrans = {c:keys[int(cc)] for c,cc in enumerate(dists.columns)}

    if zscore_filter:
        # Keep only topics that have an above-average score (from the persp. of their diachronic evolution)
        dz = dists.apply(zscore,axis=0)
        dists = dists.where(dz >= 0, 0.0000000000001)
        dists = dists.div(dists.sum(axis=1), axis=0)

    return (dists, dat, coltrans, keys) if zscore_filter == False else (dists, dat, coltrans, keys, dz)



def load_diag(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    topic_data = []
    for topic in root.findall('topic'):
        topic_id = int(topic.attrib['id'])
        coherence = float(topic.attrib.get('coherence', 0))
        exclusivity = float(topic.attrib.get('exclusivity', 0))
        document_entropy = float(topic.attrib.get('document_entropy', 0))
        word_length = float(topic.attrib.get('word-length', 0))
        eff_num_words = float(topic.attrib.get('eff_num_words', 0))
        tokens = float(topic.attrib.get('tokens', 0))
        allocation_ratio = float(topic.attrib.get('allocation_ratio', 0))
        uniform_dist = float(topic.attrib.get('uniform_dist', 0))
        corpus_dist = float(topic.attrib.get('corpus_dist', 0))
        words = [word.text for word in topic.findall('word')]

        topic_data.append({
            'topic_id': topic_id,
            'words':' '.join(words),
            'coherence': coherence,
            'exclusivity': exclusivity,
            'document_entropy': document_entropy,
            'word_length': word_length,
            'effective_num_words': eff_num_words,
            'tokens': tokens,
            'allocation_ratio': allocation_ratio,
            'uniform_dist': uniform_dist,
            'corpus_dist': corpus_dist
        })

    return pd.DataFrame(topic_data)

from os import listdir
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from sklearn.metrics import average_precision_score
import itertools
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def load_feature_names(path):
    with open(os.path.join(path, 'feature_names.txt'), 'r') as f:
        return json.load(f)


def load_samples(path):
    """
    Load LTR samples and feature names from disk
    """
    cols = ['query_id', 'date', 'query', 'doc_id', 'target']
    feature_names = load_feature_names(path)
    cols.extend(feature_names)

    files = sorted([fn for fn in listdir(path) if fn.startswith("part-")])
    df = pd.concat([pd.read_csv(os.path.join(path, fn), header=None, sep='\t', encoding='utf-8')
                    for fn in files],
                   ignore_index=True)
    df.columns = cols
    assert df.index.duplicated().sum() == 0
    assert len(df[df.target == 1]) > 0

    # Confirm that all dates are constant within rows that share the same query_id
    assert len(df.query_id.unique()) == len(df.groupby('query_id')['date'].nunique())

    # Sort samples by date
    df = df.sort_values(['date', 'query_id']).reset_index(drop=True)

    print "Loaded samples from {}".format(path)
    print "# samples per class:"
    print df.target.value_counts()

    return df, feature_names


def calc_base_average_recall(samples_df):
    target_sums = samples_df.groupby('query_id')['target'].sum()
    return target_sums.sum() / float(len(target_sums))


def preprocess_samples(samples_df, feature_names):
    """
    Prep samples for training pointwise LTR model.
    1. Downsample negative class
    2. Split samples into train and test sets, in a date-aware manner
    """
    split_train_test(samples_df)
    mark_sampled(samples_df)

    print "train base recall:", calc_base_average_recall(samples_df[samples_df.is_train])
    print "test  base recall:", calc_base_average_recall(samples_df[samples_df.is_test])


def mark_sampled(samples_df,
                 min_samples_per_query=5,
                 num_samples_per_query=10):
    """
    Downsample negative class
    Samples aren't actually removed from the dataframe.
    Instead, a column called 'sampled' is added.  IF 'sampled' == True, then the sample
    should be included in training.

    Sampling logic
    All query_id's with < min_samples_per_query are excluded. We are assuming that queries which yield such few hits
    will not benefit from re-ranking and should be excluded from training. ALso, some rankers optimize pairwise loss
    on a per-query basis so queries with so few hits will probably add noise to the training process.

    For each query_id
    - All rows with target = 1 are included
    - `num_samples_per_query` rows are included
    """

    hits_per_query_id = samples_df[samples_df.is_train].query_id.value_counts()
    low_hit_count_queries = set(hits_per_query_id[hits_per_query_id < min_samples_per_query].index.values)
    print len(low_hit_count_queries), "queries with < {} hits, removing from training set".format(min_samples_per_query)

    sampled_mask = []
    for query_id, group in samples_df.groupby('query_id'):
        n = num_samples_per_query - group['target'].sum()
        assert n > 0

        if query_id in low_hit_count_queries:
            continue

        sampled_mask.extend(group[group.target == 1].index.values)

        if len(group) > n:
            sampled_negative = group[group.target == 0].sample(n=n,
                                                               replace=False)
        else:
            sampled_negative = group

        sampled_mask.extend(sampled_negative.index.values)

    samples_df['sampled'] = False
    samples_df.loc[sampled_mask, 'sampled'] = True
    assert min(samples_df[samples_df.sampled & samples_df.is_train].query_id.value_counts()) >= min_samples_per_query

    return samples_df


def split_train_test(samples_df, train_test_split_pct=0.66):
    """
    Split samples into train and test sets.
    This function simply adds boolean columns is_test and is_train to samples_df
    IMPORTANT: This function assumes that samples are sorted in date order
    """
    split_idx = int(len(samples_df) * train_test_split_pct)
    query_id_at_split = samples_df.iloc[split_idx].query_id

    samples_df['is_test'] = False
    samples_df['is_train'] = False
    samples_df.loc[0:split_idx, 'is_train'] = True
    samples_df.loc[split_idx:, 'is_test'] = True

    # split_idx might be within a single query's pairs' boundary.
    samples_df.loc[samples_df.query_id == query_id_at_split, 'is_test'] = True
    samples_df.loc[samples_df.query_id == query_id_at_split, 'is_train'] = False

    assert len(samples_df[samples_df.is_test & samples_df.is_train]) == 0

    # assert that that samples' dates do not overlap between train/test sets
    test_min_date = samples_df[samples_df.is_test]['date'].min()
    train_max_date = samples_df[samples_df.is_train]['date'].max()
    assert test_min_date >= train_max_date


def normalize_features(samples_df, feature_names):
    """
    z-score normalize all samples w.r.t. training set
    """
    feature_value_mean_train = samples_df[samples_df.is_train][list(feature_names)].mean()
    feature_value_std_train = samples_df[samples_df.is_train][list(feature_names)].std()
    samples_df[list(feature_names)] = (samples_df[list(feature_names)] -
                                       feature_value_mean_train) / feature_value_std_train

    return feature_value_mean_train, feature_value_std_train


def normalize_X(X_train, X_test):
    """
    z-score normalize all samples w.r.t. training set
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def apply_scores_and_rank(samples_df, model_type, scores_test):
    assert len(samples_df.loc[samples_df.is_test]) == len(scores_test)

    rank_column = 'rank_{}'.format(model_type)
    score_column = 'score_{}'.format(model_type)

    samples_df.loc[samples_df.is_test, score_column] = scores_test

    samples_df[rank_column] = samples_df.groupby(
        'query_id')[score_column].rank(ascending=False, method='first')

    return rank_column, score_column


def calculate_recall_at_k(samples_df,
                          rank_column,
                          max_k=10):
    recall_at_k = []
    for k in range(1, max_k + 1):
        target_sums_by_query_id = samples_df[samples_df[rank_column] <= k].groupby('query_id')['target'].sum()
        recall_at_k.append(target_sums_by_query_id.sum() / float(len(target_sums_by_query_id)))

    df = pd.DataFrame(recall_at_k)
    df.columns = ['recall']
    df.recall = df.recall.round(decimals=3)
    df.index.name = 'k'
    df.index = df.index + 1
    return df


def calculate_mean_reciprocal_rank(samples_df,
                                   rank_column):
    return (1 / samples_df[samples_df.target == 1][rank_column]).mean()


def _group_sizes(ids):
    return [len(list(group)) for key, group in groupby(ids)]


def get_Xy(samples_df, feature_names):
    # Train, sampled from the negative class
    X_train = samples_df[samples_df.is_train & samples_df.sampled][list(feature_names)].values
    y_train = samples_df[samples_df.is_train & samples_df.sampled].target.values
    qids_train = samples_df[samples_df.is_train & samples_df.sampled].query_id.values

    # Test, use all samples
    X_test = samples_df[samples_df.is_test][list(feature_names)].values
    y_test = samples_df[samples_df.is_test].target.values
    qids_test = samples_df[samples_df.is_test].query_id.values

    return (X_train,
            y_train,
            qids_train,
            _group_sizes(qids_train),
            X_test,
            y_test,
            qids_test,
            _group_sizes(qids_test))


def train_and_test(clf, X_train, X_test, y_train):
    clf.fit(X_train, y_train)
    probas_train = clf.predict_proba(X_train)
    probas_test = clf.predict_proba(X_test)
    return probas_train, probas_test


def styled_feature_weights(feature_weights_df):
    return feature_weights_df.sort_values(1, ascending=False).style.bar(subset=[1])

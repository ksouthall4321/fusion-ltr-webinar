import matplotlib.pylab as plt
import pandas as pd
from collections import OrderedDict
from itertools import product
from sklearn import grid_search
from helpers.train import *


def single_feature_sort_evaluator(model_type,
                                  model_description,
                                  feature_name):
    def evaluate(samples_df):
        return model_type, model_description, samples_df[samples_df.is_test][feature_name], None, None
    return evaluate


def sklearn_model_evaluator(model_type,
                            model_description,
                            clf,
                            feature_names,
                            feature_score_func,
                            zscore_normalize=False):
    def evaluate(samples_df):
        X_train, y_train, _, _, X_test, y_test, _, _ = get_Xy(samples_df, feature_names)
        if zscore_normalize:
            X_train, X_test = normalize_X(X_train, X_test)

        clf.fit(X_train, y_train)
        scores_test = clf.predict_proba(X_test)

        feature_weights_df = pd.DataFrame(zip(feature_names, feature_score_func(clf)))
        feature_weights_df.sort_values(1, ascending=False).style.bar(subset=[1])

        return model_type, model_description, scores_test[:, 1], feature_weights_df, clf
    return evaluate


def to_labeled_points(samples_df, feature_names):
    for row in samples_df[feature_names + ['target']].itertuples():
        yield LabeledPoint(row[-1], row[1:-1])


def to_vectors(samples_df, feature_names):
    for row in samples_df[feature_names + ['target']].itertuples():
        yield row[1:-1]


class ExpResultsCollector:
    def __init__(self, samples_df):
        self.samples_df = samples_df
        self.results = OrderedDict()

    def collect_results(self,
                        model_type,
                        model_description,
                        scores_test,
                        feature_weights_df,
                        model):
        rank_column, _ = apply_scores_and_rank(self.samples_df, model_type, scores_test)

        recall_at_k_df = calculate_recall_at_k(self.samples_df[self.samples_df.is_test], rank_column)
        #plot_recall_at_k(recall_at_k_df, model_description)

        mrr = calculate_mean_reciprocal_rank(self.samples_df[self.samples_df.is_test], rank_column)
        print '{0: <25}'.format(model_type), "  ".join(map(lambda x: "{:.3f}".format(x), recall_at_k_df.transpose().values[0])), " ({})".format(mrr)

        self.results[model_type] = {
            'description': model_description,
            'recall_at_k': recall_at_k_df,
            'mrr': mrr,
            'feature_weights': feature_weights_df,
            'model': model
        }

    def get_results(self):
        return pd.DataFrame([results['recall_at_k'].to_dict()['recall']
                             for results in self.results.values()],
                            index=[results['description']
                                   for results in self.results.values()])

    def get_results_mrr(self):
        return pd.DataFrame([results['mrr']
                             for results in self.results.values()],
                            index=[results['description']
                                   for results in self.results.values()],
                            columns=['mrr'])


def color_negative_red(val):
    if val < 0:
        color = 'red'
    elif val == 0:
        color = 'black'
    else:
        color = 'green'

    return 'color: %s' % color


def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def style_results_df(results_df, caption):
    return (results_df
            .style
            .set_caption(caption)
            .applymap(color_negative_red)
            .apply(highlight_max))


def pct_difference(all_results_df, baseline_feature):
    return ((all_results_df - all_results_df.loc[baseline_feature]) /
            all_results_df.loc[baseline_feature] * 100)

import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

import app_logger
import sys

sys.path.insert(0, 'src/logger/')

logger = app_logger.SimpleLogger(
    'FeatureSelector', 'feature_selection.log').get_logger()


class OutlierDetector:
    """As for univariate feature distribution only"""

    def __init__(self, n_lof_neighbors=None):
        self.methods = dict({'lof': LocalOutlierFactor(n_neighbors=n_lof_neighbors, contamination='auto'),
                             'isolation_forest': IsolationForest(contamination='auto'),
                             'angle_based_outlier_detector': None
                             })

    def build_lof(self, df: pd.DataFrame, column_name: str):
        """
        # LOF is the ratio of the average LRD of the K neighbors of A to the LRD of A.
        # if the point is not an outlier (inlier), the ratio of average LRD of neighbors is approximately equal to the
        # LRD of a point (because the density of a point and its neighbors are roughly equal)
        @param df:
        @param column_name:
        @return:
        """
        lof_detector = self.methods.get('lof')
        logger.info(f'Building lof based outlier detector with neighbors - {lof_detector.n_neighbors}')
        y_pred = lof_detector.fit_predict(df[column_name].to_frame())
        df[f'outlier_{column_name}'] = y_pred.reshape(-1, 1)
        inliner_num, outlier_num = df[f'outlier_{column_name}'].value_counts().sort_values(ascending=False)[1], \
                                   df[f'outlier_{column_name}'].value_counts().sort_values(ascending=False)[-1]
        logger.info(f'Number of inliner instances - {inliner_num}')
        logger.info(f'Number of outlier instances - {outlier_num}')
        return df

    def build_isolation_forest(self, df: pd.DataFrame, columns_list: list):
        """
        @param df:
        @param columns_list:
        @return:
        """
        isolation_forest_detector = self.methods.get('isolation_forest')
        logger.info(f'Building isolation forest based outlier detector')
        # make sure there is no Nan or Infinity
        df.fillna(0)

        for i, column in enumerate(columns_list):
            isolation_forest_detector.fit(df[column].values.reshape(-1, 1))

            min_max_list = np.linspace(df[column].min(), df[column].max(), len(df)).reshape(-1, 1)
            anomaly_score = isolation_forest_detector.decision_function(min_max_list)
            outlier = isolation_forest_detector.predict(min_max_list)
            df[f'outlier_{column}'] = outlier
            inliner_num, outlier_num = df[f'outlier_{column}'].value_counts().sort_values(ascending=False)[1], \
                                       df[f'outlier_{column}'].value_counts().sort_values(ascending=False)[-1]
            logger.info(f'Number of inliner instances - {inliner_num}')
            logger.info(f'Number of outlier instances - {outlier_num}')
            OutlierDetector.show_outlier_area(min_max_list, anomaly_score, outlier, column, i)
        return df

    def build_angle_outlier_detector(self) -> None:
        """
        """
        # TBD TODO As on usage regarding high dimensional data! Investigate
        transformer_knn = self.methods.get('angle_based_outlier_detector')
        logger.info(f'Building angle based outlier detector with neighbors')

    @staticmethod
    def show_outlier_area(list_of_vals: np.ndarray, anomaly_score_list: np.ndarray,
                          outlier_based_list: np.ndarray, column: str,
                          col_index: int):
        """
        @param list_of_vals:
        @param anomaly_score_list:
        @param outlier_based_list:
        @param column:
        @param col_index:
        """
        fig, axs = plt.subplots(1, 3, figsize=(20, 5), facecolor='w', edgecolor='k')
        axs = axs.ravel()
        axs[col_index].plot(list_of_vals, anomaly_score_list, label='anomaly score')
        axs[col_index].fill_between(list_of_vals.T[0], np.min(anomaly_score_list), np.max(anomaly_score_list),
                                    where=outlier_based_list == -1, color='r',
                                    alpha=.4, label='outlier region')
        axs[col_index].legend()
        axs[col_index].set_title(column)
        fig.show()

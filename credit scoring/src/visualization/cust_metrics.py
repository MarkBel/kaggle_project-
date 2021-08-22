from typing import Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import sys
sys.path.insert(0, 'src/logger/')
import app_logger



class CustomMetrics():
    """
    generates model quality report 
    """

    def __init__(self, labels: np.array, custom_metric: function = None,) -> None:
        """

        Args:
            labels (np.array):  classes labels 
            custom_metric (function, optional): optional custom metric. Defaults to None.
        """
        self.custom_metric = custom_metric
        self.labels = labels

    def output_classification_metrics(self, y_true: np.array, y_pred: np.array, bound: float = 0.5, average: str = "binary") -> None:
        """
        generate  model validation metrics for classification task

        Args:
            y_true (np.array): true label of the data with shape (nsamples,)
            y_pred (np.array): predicted label of the data with shape (nsamples,)
            bound (float, optional): classification threshold. Defaults to 0.5.
            average (str, optional): this parameter is required for multiclass/multilabel targets. Defaults to "binary".
        """

        print("F1 score ", f1_score(y_true=y_true, y_pred=(
            y_pred > bound).astype(int), average=average))

        print("AUC score ", roc_auc_score(
            y_true=y_true, y_pred=(y_pred > bound).astype(int), average=average))

        print("Precision score ", precision_score(
            y_true=y_true, y_pred=(y_pred > bound).astype(int), average=average))

        print("Recall score ", recall_score(
            y_true=y_true, y_pred=(y_pred > bound).astype(int), average=average))

        print(f"{self.custom_metric.__name__} score ",
              self.custom_metric(y_true, y_pred))

        self.confusion_matrix_display(y_true, y_pred, self.labels)

    def output_reg_metrics(self, y_true: np.array, y_pred: np.array) -> None:
        """
            generate  model validation metrics for regression task
        Args:
            y_true (np.array): true label of the data with shape (nsamples,)
            y_pred (np.array): predicted label of the data with shape (nsamples,)
        """

        print("mae score ", mean_absolute_error(
            y_true=y_true, y_pred=y_pred))

        print("mse score ", mean_squared_error(
            y_true=y_true, y_pred=y_pred))

        print(f"{self.custom_metric.__name__} score ",
              self.custom_metric(y_true, y_pred))

    def confusion_matrix_display(self, y_true: np.array, y_pred: np.array, ymap: Dict[Any] = None, figsize: Tuple[int] = (5, 5)) -> None:
        """
            Gemerate matrix plot of confusion matrix
        Args:
            y_true (np.array): true label of the data with shape (nsamples,)
            y_pred (np.array): predicted label of the data with shape (nsamples,)
            ymap (Dict[Any], optional): dict any -> str, length == nclass
                                        if not None, map the labels and ys to more understandable string
                                        Caution:original y_true, y_pred and labels must align. 
                                        Defaults to None.
            figsize (Tuple[int], optional): the size of the figure plotted. Defaults to (5, 5).
        """
        if ymap is not None:
            y_pred = [ymap[yi] for yi in y_pred]
            y_true = [ymap[yi] for yi in y_true]
            labels = [ymap[yi] for yi in self.labels]

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm/cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for row in range(nrows):
            for col in range(ncols):
                c = cm[row, col]
                p = cm_perc[row, col]
                if row == col:
                    s = cm_sum[row]
                    annot = f"{round(p,1)} \n {c}/{s}"
                elif c == 0:
                    annot[row, col] = ""
                else:
                    annot[row, col] = f"{round(p,1)} \n {c}"
        cm = pd.DataFrame(cm, index=labels, columns=labels)
        cm.index.name = "actual"
        cm.columns.name = "Predicted"
        _, ax = plt.subplot(figsize=figsize)
        sns.heatmap(cm, annot=annot, fmt="", ax=ax, cmap="Greens")
        plt.show()

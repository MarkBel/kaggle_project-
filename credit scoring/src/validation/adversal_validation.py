import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *

import sys
sys.path.insert(0, 'src/logger/')
import app_logger

logger = app_logger.SimpleLogger(
    'AdversalValidation', 'adversal_validation.log').get_logger()


class AdversalValidation():

    def __init__(self, file_name: str) -> None:
        """
        initialization function 
        Args:
            file_name (str): the name of the file where all the information of the train will be saved with the probability of belonging to the test 
        """
        self.file_name = file_name

    def fit_transform(self, train: pd.DataFrame, test: pd.Series, model, threshold_count=5, threshold_auc=0.55, target_col: str = None) -> None:
        """
        trains the classifier to separate the test and the train, 
        if the quality of the model was greater than the threshold_auc value and it happened more than the threshold_count, 
        then we save the train file with the probabilities of belonging to the test 

        Args:
            train (pd.DataFrame): preprocessed train with target  ;train with shape(nsample, col_test + target)
            test (pd.Series): preprocessed test ; test with shape(nsample, col_test)
            model ([type]): any sklearn classifier
            threshold_count (int, optional): the counting of auc is more than threshold_auc for saving file with probability. Defaults to 5.
            threshold_auc (float, optional): the choice of the threshold value of the quality of the model indicating that 
                                            there is a relationship between the test and the train  Defaults to 0.55.
            target_col (str, optional): target col. Defaults to None.
        """

        cols_train, cols_test = list(train.columns), list(test.columns)
        cols_test.append(target_col)

        common_cols_train = list(set(cols_train).intersection(set(cols_test)))
        
        try:
            assert len(common_cols_train) == len(cols_test) == len(cols_train)
        except AssertionError:
            logger.error("Train and test data have different features")
            logger.error(
                f"Common columns of test ans train {common_cols_train}, test columns are {cols_test}, train columns are {cols_train}")

            train = train[common_cols_train]
            cols_for_test = [
                col for col in common_cols_train if col not in target_col]
            logger.error(f"{cols_for_test}")
            test = test[cols_for_test]

        logger.info("script is running")
        origin_train = train.copy()

        train["is_test"] = 0
        test["is_test"] = 1

        all_data = pd.concat((train, test))
        all_data.reset_index(inplace=True, drop=True)

        x = all_data.drop(columns=["is_test", target_col])
        y = all_data["is_test"]
        model_pipeline = Pipeline([
            ("model", model)
        ])

        prediction = np.zeros(y.shape)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        count = 0
        for train_ind, test_ind in tqdm(cv.split(x, y), desc="Validation in progress"):
            x_train = x.iloc[train_ind]
            x_test = x.iloc[test_ind]
            y_train = y.iloc[train_ind]
            y_test = y.iloc[test_ind]

            model_pipeline.fit(x_train, y_train)

            prob = model_pipeline.predict_proba(x_test)[:, 1]

            auc = AUC(y_test, prob)
            if auc > threshold_auc:
                count += 1
                print(f" AUC: {auc}")
            else:
                print(f" AUC: {auc}")

            prediction[test_ind] = prob

        if count > threshold_count:
            logger.info(f"file will save in {self.file_name}")
            origin_train["prob"] = prediction[:origin_train.shape[0]]
            origin_train.to_csv(self.file_name, index=False)


"""


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    advalid = AdversalValidation(file_name='check.csv')
    train = train[['SalePrice','MoSold']]
    test = test[['LotArea','MoSold']]
    advalid.fit_transform(train.copy(),test.copy(),RandomForestClassifier(),threshold_count=1,threshold_auc = 0.5,target_col='SalePrice')

"""
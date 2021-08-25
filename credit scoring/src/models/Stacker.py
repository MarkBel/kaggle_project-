from typing import List, Any
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np

class Stacker():

    def __init__(self, estimators: List[str, Any], meta_model: Any) -> None:
        self.estimators = estimators
        self.meta_model = meta_model

    def fit(self, data, target) -> None:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        prediction = np.zeros(target.shape)


        for train_ind, test_ind in tqdm(cv.split(data, target), desc="Validation in progress"):
            x_train = data.iloc[train_ind]
            x_test = data.iloc[test_ind]
            y_train = target.iloc[train_ind]
            y_test = target.iloc[test_ind]


    def predict(self) -> None:
        pass

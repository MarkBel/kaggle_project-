import app_logger
from typing import List, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from numpy import hstack

sys.path.insert(0, 'src/logger/')

logger = app_logger.SimpleLogger(
    'Blender', 'Blender.log').get_logger()


class Blander():

    def __init__(self, estimators: List[str, Any], metamodel: Any, usage_data: False) -> None:
        self.estimators = estimators
        self.metamodel = metamodel
        self.usage_data = usage_data

    def fit(self, data: pd.DataFrame, target: Any) -> None:
        meta_data = []
        x_train, x_val, y_train, y_val = train_test_split(
            data, target, test_size=0.4)
        # TODO add optimization
        for name, model in self.estimators:

            model.fit(x_train, y_train)
            y_oof = model.predict(x_val)
            meta_data.append(y_oof)

        meta_data = hstack(meta_data)
        
        if self.usage_data:
            for i in len(meta_data):
                x_val[f'new_cols_{i}'] = meta_data[i]
            self.metamodel.fit(x_val,y_val)
        else:
            self.metamodel.fit(meta_data,y_val)
        return None

    def predict(self, test_data: pd.DataFrame):
        meta_prediction = []
        for name, model in self.estimators:

            y_pred = model.predict(test_data)

            meta_prediction.append(y_pred)

        meta_prediction = hstack(meta_prediction)
        
        if self.usage_data:
            for i in len(meta_prediction):
                test_data[f'new_cols_{i}'] = meta_prediction[i]
            return self.metamodel.predict(test_data)

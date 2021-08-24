import optuna
from sklearn.model_selection import train_test_split
from typing import Any
import pandas as pd
import pickle
"""
делать ямлики парсить и переводить в объекты, работать с ними
"""


class Optimizer():
    def __init__(self, model: Any, params: dict, model_path: str, log_path_params: str) -> None:

        self.model = model
        self.params = params
        self.model_path = model_path
        self.log_path_params = log_path_params

    def __save_model(self,model) -> None:
        pickle.dump(model, open(self.model_path, 'wb'))
        return None

    def __save_best_params_model(self,best_params):
        with open('logs.txt', 'a') as logs:
            logs.write(f' ?? Best parameter: ' + str(best_params)+'\n')


    def optimize(self,model,data,target,metrics):
        def objective(trial):
            # hyperparameter setting
            ridge_alpha = trial.suggest_uniform('alpha', 0.0, 10000.0)
            l1 = trial.suggest_uniform('l1_ratio', 0.0, 1)
            model = linear_model.ElasticNet(alpha=ridge_alpha,l1_ratio=l1)
            # data loading and train-test split
            X_train, X_val, y_train, y_val = train_test_split(data, target)

            # model training and evaluation
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            error = metrics(y_val, y_pred)
        
            # output: evaluation score
            return error


    def optimize_multitarget(self):
        pass


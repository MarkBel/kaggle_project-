import pandas as pd

from train_model import StackedLGXB


def process_stack_prediction(data_train: pd.DataFrame, data_train_meta, data_test_train, data_test_meta, target):
    a_A: float = 0.0
    a_B: float = 0.0
    N: int = 2

    for t in range(N):
        clf = StackedLGXB(seed=2000 + 10 * t, lgb_trees=2000, xgb_trees=2000)
        clf.fit(data_train, target)
        a_A += clf.predict_proba(data_test_train)

        # TODO In case of having train metadata
        clf = StackedLGXB(seed=(3000 + 10 * t), lgb_trees=2000, xgb_trees=2000)
        clf.fit(data_train_meta, target)
        a_B += clf.predict_proba(data_test_meta)

    a = (a_A + a_B) / 2

    a = a - min(a)
    a = a / max(a)

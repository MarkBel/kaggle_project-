from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import xgboost as xgb


class StackedLGXB(BaseEstimator, ClassifierMixin):

    # TODO COULD be extended on demand with models and params on demand
    def __init__(self, seed=0, lgb_trees=1.0, xgb_trees=1.0, cbt=0.5, ss=0.5, alpha=0.5):
        """
        @param seed: random pseudo seed
        @param lgb_trees: num of trees as of lgb
        @param xgb_trees: num of trees as of xgb
        @param cbt:percent of the columns per tree
        @param ss: percent of the samples per tree
        @param alpha: XGB param only
        """
        self.models = [lgb.LGBMClassifier(num_leaves=2, learning_rate=0.07, n_estimators=lgb_trees,
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=0 + seed),
                       lgb.LGBMClassifier(num_leaves=3, learning_rate=0.07, n_estimators=lgb_trees,
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=1 + seed),
                       lgb.LGBMClassifier(num_leaves=4, learning_rate=0.07, n_estimators=lgb_trees,
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=2 + seed),
                       lgb.LGBMClassifier(num_leaves=5, learning_rate=0.07, n_estimators=lgb_trees,
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=3 + seed, ),
                       xgb.XGBClassifier(max_depth=1,
                                         learning_rate=0.1,
                                         n_estimators=xgb_trees,
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=0 + seed),
                       xgb.XGBClassifier(max_depth=2,
                                         learning_rate=0.1,
                                         n_estimators=xgb_trees,
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=1 + seed),
                       xgb.XGBClassifier(max_depth=3,
                                         learning_rate=0.1,
                                         n_estimators=xgb_trees,
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=2 + seed),
                       xgb.XGBClassifier(max_depth=4,
                                         learning_rate=0.1,
                                         n_estimators=xgb_trees,
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=3 + seed)
                       ]
        self.weights = [(1 - alpha) * 1, (1 - alpha) * 1, (1 - alpha) * 1, (1 - alpha) * 0.5, alpha * 0.5, alpha * 1,
                        alpha * 1.5, alpha * 0.5]

    def fit(self, X, y=None):
        for t, clf in enumerate(self.models):
            clf.fit(X, y)
        return self

    def predict_proba(self, X):
        sum = 0.0
        for t, clf in enumerate(self.models):
            a = clf.predict_proba(X)[:, 1]
            sum += (self.weights[t] * a)
        return (sum / sum(self.weights))

    def predict(self, X):
        return (self.predict(X))

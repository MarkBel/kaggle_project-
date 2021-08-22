import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors


class FeatureGenerator:

    #TODO  1. Add logging 2. How much variance explained with PCA


    def __init__(self, output_dim: int = None, kernel=None, degree: int = 2, n_neighbors: int = None):
        self.methods = dict({'kernel_pca': KernelPCA(n_components=output_dim, kernel=kernel),
                             'polynomial': PolynomialFeatures(degree=degree),
                             'knn': NearestNeighbors(n_neighbors=n_neighbors)
                             })

    def build_knn(self,  X: pd.DataFrame) -> pd.DataFrame:
        transformer_knn = self.methods.get('knn')
        fitter_knn = transformer_knn.fit(X)
        _, indices = fitter_knn.kneighbors(X)
        return indices

    def build_kernel_pca(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        transformer_pca = self.methods.get('kernel_pca')
        X_transformed = transformer_pca.fit_transform(X)
        return X_transformed

    def build_polynomial_features(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        polynom = self.methods.get('polynomial')
        polynomial_features_transformed = polynom.fit_transform(X)
        return polynomial_features_transformed

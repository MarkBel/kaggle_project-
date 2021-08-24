import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors
import app_logger
import sys

sys.path.insert(0, 'src/logger/')

logger = app_logger.SimpleLogger(
    'FeatureSelector', 'feature_selection.log').get_logger()


class FeatureGenerator:

    def __init__(self, output_dim: int = None, kernel=None, degree: int = 2, n_neighbors: int = None):
        self.methods = dict({'kernel_pca': KernelPCA(n_components=output_dim, kernel=kernel),
                             'polynomial': PolynomialFeatures(degree=degree),
                             'knn': NearestNeighbors(n_neighbors=n_neighbors)
                             })

    def build_knn(self, X: pd.DataFrame) -> pd.DataFrame:
        transformer_knn = self.methods.get('knn')
        logger.info(f'Building knn based features with neighbors num- {transformer_knn.n_neighbors}')
        fitter_knn = transformer_knn.fit(X)
        _, indices = fitter_knn.kneighbors(X)
        return indices

    def build_kernel_pca(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        transformer_pca = self.methods.get('kernel_pca')
        logger.info(f'Building kernel PCA based features with num_dib - {transformer_pca.n_components}')
        X_transformed = transformer_pca.fit_transform(X)
        logger.info(
            f'Explained variance rounded to 2 digits - {round(X_transformed.explained_variance_ratio_.sum(), 2)}')
        return X_transformed

    def build_polynomial_features(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        polynom = self.methods.get('polynomial')
        logger.info(f'Building polynomial based features of degree- {polynom.degree}')
        polynomial_features_transformed = polynom.fit_transform(X)
        return polynomial_features_transformed

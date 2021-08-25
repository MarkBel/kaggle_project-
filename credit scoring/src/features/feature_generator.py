import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import NearestNeighbors

import sys
sys.path.insert(0, 'src/logger/')
import app_logger
logger = app_logger.SimpleLogger(
    'FeatureGenerator', 'feature_generation.log').get_logger()


class FeatureGenerator:
    """
    class with implementation of automatic feature generation models
    """

    def __init__(self, output_dim: int = None, kernel=None, degree: int = 2, n_neighbors: int = None):
        """
        conscructor of  class

        Args:
            output_dim (int, optional): Number of feature generation. Defaults to None.
            kernel ([type], optional): kernel of PCA method. choose nonlinear kernels . Defaults to None.
            degree (int, optional): degree of polynomial features . Defaults to 2.
            n_neighbors (int, optional): number of neighbors for knn method. Defaults to None.
        """
        self.output_dim = output_dim
        self.kernel = kernel
        self.degree = degree
        self.n_neighbors = n_neighbors

    def build_knn(self, X: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """
        generate number of classter as additional features

        Args:
            X (pd.DataFrame): train dataset for fit knn model
            test (pd.DataFrame): test dataset

        Returns:
            pd.DataFrame train dataset with additional feature ,pd.DataFrame test dataset with additional feature 
        """
        transformer_knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        logger.info(
            f'Building knn based features with neighbors num- {transformer_knn.n_neighbors}')
        fitter_knn = transformer_knn.fit(X)
        X['knn_feature'] = fitter_knn.predict(X)
        test['knn_feature'] = fitter_knn.predict(test)
        return X, test

    def build_kernel_pca(self,  X: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """
        generate nonlinear features by nonlinear kernel PCA

        Args:
            X (pd.DataFrame): train dataset
            test (pd.DataFrame): test dataset

        Returns:
             pd.DataFrame train dataset with additional features ,pd.DataFrame test dataset with additional features 

        """
        transformer_pca = KernelPCA(
            n_components=self.output_dim, kernel=self.kernel)
        logger.info(
            f'Building kernel PCA based features with num_dib - {transformer_pca.n_components}')
        X_transformed = transformer_pca.fit_transform(X)
        logger.info(
            f'Explained variance rounded to 2 digits - {round(X_transformed.explained_variance_ratio_.sum(), 2)}')
        test_pca_transformed = transformer_pca.transform(test)
        return X_transformed, test_pca_transformed

    def build_polynomial_features(self, X: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
        """
        generate polynomial features for model

        Args:
            X (pd.DataFrame): train dataset 
            test (pd.DataFrame): test dataframe

        Returns:
            pd.DataFrame train dataset with additional polynimial features ,pd.DataFrame test dataset with additional polynimial features 
        """
        polynom = PolynomialFeatures(degree=self.degree)
        logger.info(
            f'Building polynomial based features of degree- {polynom.degree}')
        polynomial_features_transformed = polynom.fit_transform(X)
        polynomial_features_transformed_test = polynom.transform(test)
        return polynomial_features_transformed, polynomial_features_transformed_test

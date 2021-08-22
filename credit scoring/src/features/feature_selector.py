from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.models import Sequential, Input, Model
from sklearn.model_selection import train_test_split
import pandas as pd
from umap import UMAP
import app_logger
import sys
sys.path.insert(0, 'src/logger/')


logger = app_logger.SimpleLogger(
    'FeatureSelector', 'feature_selection.log').get_logger()


class Feature_Selector():

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, y: pd.DataFrame, method: str, output_dim: int, file_path: str) -> None:
        self.train = train
        self.method = method
        self.output_dim = output_dim
        self.file_path = file_path

    def __create_autoencoder_model(self) -> Model:
        logger.info("initialization AE")
        encoder_layer_shapes = [512, 256, 128, 256, 128, 64, self.output_dim]

        input_ = Input(shape=(self.train.shape[0],))
        encoded = Dense(encoder_layer_shapes[0], activation='relu')(input_)
        encoded = Dense(encoder_layer_shapes[1], activation='relu')(encoded)
        encoded = Dense(encoder_layer_shapes[2], activation='relu')(encoded)
        encoded = Dense(encoder_layer_shapes[3], activation='relu')(encoded)
        encoded = Dense(encoder_layer_shapes[4], activation='relu')(encoded)
        encoded = Dense(encoder_layer_shapes[5], activation='relu')(encoded)
        encoded = Dense(encoder_layer_shapes[6], activation='sigmoid')(encoded)
        model = Model(input_, encoded)
        model.compile(optimizer='Adam', loss='mean_squared_error')
        return model

    def __build_autoencoder_fit(self) -> Model:
        logger.info("Training AE")
        autoencoder = self.__create_autoencoder_model()
        train, test, y_train, y_test = train_test_split(
            self.train.values(), self.y, test_size=0.2)

        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=35,
                                   verbose=0,
                                   min_delta=1e-4)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=5,
                                      cooldown=2,
                                      verbose=0)

        autoencoder.fit(train, train,
                        epochs=200,
                        batch_size=32,
                        validation_data=(test, test), callbacks=[early_stop, reduce_lr], verbose=0)
        return autoencoder

    def AE_selector(self,) -> None:
        logger.info("Predicting AE and Save to csv")
        autoencoder = self.__build_autoencoder_fit()
        model_bn = Model(autoencoder.input, autoencoder.layers[6].output)
        np.totxt("train_"+self.file_path,
                 model_bn.predict(self.train.values()), delimiter=',')
        np.totxt("test_"+self.file_path,
                 model_bn.predict(self.train.values()), delimiter=',')
        return None

    def PCA_selector(self,) -> None:
        logger.info("Training AE and Save result to csv")
        pca = PCA(n_components=self.output_dim)
        pca.fit(self.train)
        logger.info("Components = ", pca.n_components_, ";\nTotal explained variance = ",
                    round(pca.explained_variance_ratio_.sum(), 2))
        np.totxt("train_"+self.file_path,
                 pca.transform(self.train), delimiter=',')
        np.totxt("test_"+self.file_path,
                 pca.transform(self.test), delimiter=',')
        return None

    def Umap_selector(self,) -> None:
        umap = UMAP(n_components=self.output_dim)
        umap.fit(self.train)
        np.totxt("train_"+self.file_path,
                 umap.transform(self.train), delimiter=',')

        np.totxt("test_"+self.file_path,
                 umap.transform(self.test), delimiter=',')
        return None

    def SVD_selector(self,) -> None:
        svd = TruncatedSVD(n_components=self.output_dim, n_iter=10)
        svd.fit(self.train)

        logger.info("Components = ", svd.n_components_, ";\nTotal explained variance = ",
                    round((svd.explained_variance_ratio_.sum()), 2))
        np.totxt("train_"+self.file_path,
                 svd.transform(self.train), delimiter=',')
        np.totxt("test_"+self.file_path,
                 svd.fit_transform(self.test), delimiter=',')
        return None

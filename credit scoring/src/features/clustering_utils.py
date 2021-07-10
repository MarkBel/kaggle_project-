import pandas as pd
from sklearn.preprocessing import  MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def preprocess_data_with_scaler(df:pd.DataFrame) -> None:
    """
    get metadata
    return preprocessed metadata
    """
    scaler = MinMaxScaler()
    for col in df.columns:
        df[col] = scaler.fit_transform(df[col])


def get_cluster(df:pd.DataFrame, num_features:int = None) -> None:
    if isinstance(num_features,int):
        pca = PCA(n_components = num_features,random_state = 42)
    df = pca.fit_transform(df)
    max_num_clust = 21
    kMeans_inertia = pd.DataFrame(data=[],index=range(2,max_num_clust), \
                                columns=['inertia'])


    for n_clusters in range(2,max_num_clust):
        kmeans = KMeans(n_clusters = n_clusters, random_state = 42, \
                n_jobs = -1)

        kmeans.fit(df)
        kMeans_inertia.loc[n_clusters] = kmeans.inertia_
    kMeans_inertia.plot()
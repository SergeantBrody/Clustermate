import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class ClusteringAssistant:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.df = pd.DataFrame(self.data)
        self.vectorizer = TfidfVectorizer()
        self.documents = self.df['product_title'].values.astype("U")
        self.features = self.vectorizer.fit_transform(self.documents)

    def BestofK(self):
        k_range = range(2, 31)
        silhouette_scores = []
        calinski_harabasz_scores = []
        davies_bouldin_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.features)
            silhouette_score_k = silhouette_score(self.features, cluster_labels)
            calinski_harabasz_score_k = calinski_harabasz_score(self.features.toarray(), cluster_labels)
            davies_bouldin_score_k = davies_bouldin_score(self.features.toarray(), cluster_labels)
            silhouette_scores.append(silhouette_score_k)
            calinski_harabasz_scores.append(calinski_harabasz_score_k)
            davies_bouldin_scores.append(davies_bouldin_score_k)

        best_k = max(range(2, 31), key=lambda k: (silhouette_scores[k-2], calinski_harabasz_scores[k-2], -davies_bouldin_scores[k-2]))
        
        print(f'Suggested numer of clusters: {best_k}')
        
    
    def kmeans_clustering(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
        kmeans.fit(self.features)
        self.df['cluster'] = kmeans.labels_
        self.df.to_csv('clustered_data.csv', index=False)
        return kmeans
    
    def save_clusters(self, kmeans):
        clusters = self.df.groupby('cluster')

        for cluster in clusters.groups:
            cluster_data = clusters.get_group(cluster)['asin']
            cluster_data.to_csv(f'Cluster{cluster}.csv', index=False, header=False)

    def create_training_data(self, clustering_model):
        sample = []
        if isinstance(clustering_model, KMeans):
            n_clusters = clustering_model.n_clusters
            for i in range (n_clusters):
                cluster_df = self.df[self.df['cluster'] == i]
                if cluster_df.shape[0] < 10:
                    sample.extend(np.random.choice(cluster_df['asin'], 1, replace=False))
                else:
                    sample.extend(np.random.choice(cluster_df['asin'], int(cluster_df.shape[0] * 0.1), replace=False))
        elif isinstance(clustering_model, DBSCAN):
            unique_clusters = set(self.df[self.df['cluster'] != -1]['cluster'])

            for cluster in unique_clusters:
                cluster_df = self.df[self.df['cluster'] == cluster]
                if cluster_df.shape[0] < 10:
                    sample.extend(np.random.choice(cluster_df['asin'], 1, replace=False))
                else:
                    sample.extend(np.random.choice(cluster_df['asin'], int(cluster_df.shape[0] * 0.1), replace=False))
        
        sample_df = pd.DataFrame({
            'asin': sample,
            'Label': '' #Empty label column
        }) 

        output_file = f'output_sample.csv'
        sample_df.to_csv(output_file, index=False)

        print(f"Sample data saved to {output_file}")
        return sample_df

    def display_cluster_centroids(self, kmeans):
        print("Cluster centroids:")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names_out()
        for i in range(kmeans.n_clusters):
            print(f"Cluster {i}: ")
            for j in order_centroids[i, :10]:
                print(f' {terms[j]}')
            print('-------------------')

    def display_cluster_sizes(self):
        cluster_sizes = self.df['cluster'].value_counts()
        print("Cluster sizes")
        print(cluster_sizes)
        cluster_sizes.to_csv('cluster_sizes.csv')

    def dbscan_clustering(self):
        dbscan = DBSCAN(eps=0.7, min_samples=2)
        dbscan.fit(self.features)
        self.df['cluster'] = dbscan.labels_
        self.df.to_csv('clustered_data.csv', index=False)
        num_clusters = len(self.df['cluster'].unique())
        print(f"Number of clusters: {num_clusters}")
        return dbscan
   
    
import numpy as np
import random


class KMeans():

    def __init__(self, n_clusters: int, init: str = 'random', max_iter=300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None  # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            dist = self.euclidean_distance(X, self.centroids)
            for i in range(X.shape[0]):
                clustering[i] = np.argmin(dist[i])

            self.update_centroids(clustering, X)
            iteration += 1
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        """
        Update centroids based on the current clustering.
        :param clustering:
        :param X:
        :return:
        """
        for feature in range(X.shape[1]):
            for cluster in range(self.n_clusters):
                self.centroids[cluster][feature] = np.mean(X[clustering == cluster, feature])

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        self.centroids = np.zeros((self.n_clusters, X.shape[1]))
        if self.init == 'random':
            for i in range(self.n_clusters):
                self.centroids[i] = X[random.randrange(X.shape[0])]
        elif self.init == 'kmeans++':
            self.centroids[0] = X[random.randrange(X.shape[0])]
            for i in range(1, self.n_clusters):
                sum = 0
                dist = self.euclidean_distance(X, self.centroids)
                for j in range(X.shape[0]):
                    sum += np.min(dist[j])
                sum *= random.random()
                for j in range(X.shape[0]):
                    sum -= np.min(dist[j])
                    if sum <= 0:
                        self.centroids[i] = X[j]
                        break
        else:
            raise ValueError(
                'Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1: np.ndarray, X2: np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        eu_distance = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                sum = 0
                for k in range(len(X1[i])):
                    sum += pow(X1[i][k] - X2[j][k], 2)
                eu_distance[i][j] = np.sqrt(sum)
        return eu_distance

    def calculate_eu_distance(self, X1: np.ndarray, X2: np.ndarray):
        eu_distance = np.zeros(X2.shape[0])
        for i in range(X2.shape[0]):
            sum = 0
            for j in range(X2.shape[1]):
                sum += pow(X1[j] - X2[i][j], 2)
            eu_distance[i] = np.sqrt(sum)
        return eu_distance

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # calculate the silhouette coefficient
        silhouette_score = np.zeros(self.n_clusters)
        for i in range(self.n_clusters):
            this_cluster = X[clustering == i]
            for point in this_cluster:
                dist_in_cluster = np.mean(self.calculate_eu_distance(point, this_cluster))

                min_dist_between_clusters = 100
                for j in range(self.n_clusters):
                    if j == i:
                        continue
                    dist_between_clusters = np.mean(self.calculate_eu_distance(point, X[clustering == j]))
                    if dist_between_clusters < min_dist_between_clusters:
                        min_dist_between_clusters = dist_between_clusters
            
                silhouette_score[i] += 1 - dist_in_cluster / min_dist_between_clusters

            silhouette_score[i] /= this_cluster.shape[0]
        
        return silhouette_score


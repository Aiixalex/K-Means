import numpy as np


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
            update_centroids(clustering, X)
            iteration += 1
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        # your code

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            # your code
        elif self.init == 'kmeans++':
            # your code
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
        eu_distance = np.zeros[X1.shape[0], X2.shape[0]]
        for i in range(X1.shape[0]):
            for j in range(X2.shape[0]):
                sum = 0
                for k in range(len(X1[i])):
                    sum += pow(X1[i][k] - X2[j][k], 2)
                eu_distance[i][j] = np.sqrt(sum)
        return eu_distance


    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code

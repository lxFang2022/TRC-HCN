import random
import numpy as np
from sklearn.preprocessing import StandardScaler

class KMeansPriorPenalty():

    def __init__(self, n_cluster):
        self.n_cluster = n_cluster
        self.prior = np.load("priorKnowledge.npy")
        for j in range(1026):
            for i in range(len(self.prior[0])):
                if(self.prior[j][i] != 0):
                    self.prior[j][i] = i + 1
        for i in range(len(self.prior[1])):
            self.prior[i,0] = np.max(self.prior[i,:])

    def set_centroids(self, X):
        """Randomly assign centroids"""
        return X[random.choices(range(X.shape[0]), k=self.n_cluster), :]

    def assign(self, centroids, X):
        """Assign the datapoints to the centroids"""
        return ((X[:, None] - centroids) ** 2).mean(2).argmin(1)

    def get_loss(self, X, classes, centroids, loss_penalty):
        loss = 0
        for class_ in range(self.n_cluster):
            x_class = X[classes == class_, :]
            centroid_class = centroids[class_]

            loss += np.sum(np.sqrt(np.sum((x_class - centroid_class) ** 2, 1)))

        loss += np.sum(np.sign(np.abs(classes-self.prior[:,0])) * loss_penalty)

        return loss

    def update_centroids(self, X, classes, centroids, eta, loss):
        """Update the centroids"""

        for class_ in range(self.n_cluster):
            x_class = X[classes == class_, :]

            if x_class.shape[0] > 0:
                for feature in range(X.shape[1]):
                    centroids[class_, feature] += -eta * 1 / x_class.shape[0] * np.sum(
                        (centroids[class_, feature] - x_class[:, feature])) / loss  #pwa 妙啊

        return centroids

    def fit(self, X, steps, n_iter, eta, loss_penalty):
        """Run the algorithm

        :param X: [m n] time series dataset of m points and n features
        :param steps: (int) maximum number of step per iteration
        :param n_iter: (int) number of iteration to be ran
        :param eta: (float) learning rate
        :param loss_penalty: (float) prior penalty to be added in time series
        """

        assert loss_penalty >= 0, 'The regularization term should be positive'

        # Scale data
        sc = StandardScaler()
        X_std = sc.fit_transform(X)

        output = {}

        for iter_ in range(n_iter):

            # Pick first step centroids
            centroids = self.set_centroids(X_std)
            previous_loss = np.inf

            for i in range(steps):
                # print(i)
                # Assign each point to a class
                classes = self.assign(centroids, X_std)

                # Computes the global loss
                loss = self.get_loss(X_std, classes, centroids, loss_penalty)

                # If improvement keep going, otherwise track metrics
                if (loss < previous_loss) and (i != steps - 1):
                    previous_loss = loss
                    centroids = self.update_centroids(X_std, classes, centroids, eta, loss)

                else:
                    output[iter_] = [loss, centroids, classes]
                    break

        # Get best result
        losses = [output[x][0] for x in range(n_iter)]
        best = np.argmin(losses)

        # Returns the centroid and the class belonging
        return output[best][1], output[best][2]
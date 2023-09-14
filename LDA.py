import copy

import numpy as np
import pandas as pd
from tqdm import tqdm


class LDA:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        self.x = x
        self.y = y
        self.labels = self.y["9"].unique()

        print("number of samples : ", self.x.shape[0])
        print("number of features : ", self.x.shape[1])

    def class_scatter_matrix(self, category):
        category_data = self.x.iloc[self.y[self.y == category].dropna().index]
        mu = np.mean(category_data, axis=0)
        scatter = np.dot((category_data - mu).T, (category_data - mu))

        return scatter

    def calc_within_scatter_matrix(self):
        self.within_scatter = np.zeros(
            (self.x.shape[1], self.x.shape[1]), dtype=np.float64
        )

        for c in self.labels:
            self.within_scatter += self.class_scatter_matrix(c)

    def calc_between_scatter_matrix(self):
        self.between_scatter = np.zeros(
            (self.x.shape[1], self.x.shape[1]), dtype=np.float64
        )

        mu = np.mean(self.x, axis=0)

        for c in self.labels:
            c_data = self.x.iloc[self.y[self.y == c].dropna().index]
            c_mu = np.mean(c_data, axis=0)
            Nj = c_data.shape[0]

            self.between_scatter += Nj * np.dot((c_mu - mu).T, c_mu - mu)

    def fit(self, n_components=None):
        if n_components == None:
            n_components = self.x.shape[1]

        self.calc_between_scatter_matrix()
        self.calc_within_scatter_matrix()

        self.Separability_matrix = np.dot(
            np.linalg.inv(self.within_scatter), self.between_scatter
        )

        eigen_values, eigen_vectors = np.linalg.eig(self.Separability_matrix)

        eigen_pairs = [
            (np.abs(eigen_values[i]), eigen_vectors[:, i])
            for i in range(len(eigen_values))
        ]
        eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

        self.w = np.hstack(
            [eigen_pairs[i][1].reshape(self.x.shape[1], 1) for i in range(n_components)]
        )

        return np.dot(self.x, self.w)

    def predict(self, x):
        return np.dot(x, self.w)

    def separability_eigen_values(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.Separability_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]

        return eigenvalues

    def separability_trace(self):
        separability_trace = list()
        data = copy.deepcopy(self.x)
        features = []
        for feature in tqdm(data.columns):
            features.append(feature)
            self.x = data[features]
            self.calc_between_scatter_matrix()
            self.calc_within_scatter_matrix()
            separability_trace.append(
                np.trace(np.dot(self.within_scatter.T, self.between_scatter))
            )

        return separability_trace


if __name__ == "__main__":
    x_train = pd.read_csv("Fashion-MNIST/trainData.csv")
    x_test = pd.read_csv("Fashion-MNIST/testData.csv")

    y_train = pd.read_csv("Fashion-MNIST/trainLabels.csv")
    y_test = pd.read_csv("Fashion-MNIST/testLabels.csv")

    lda = LDA(x=x_train, y=y_train)

    separability_trace = LDA.separability_trace()

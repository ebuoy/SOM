import numpy as np
import random


def dist_quad(x, y):
    return np.sum((x - y) ** 2)


def gauss(d, sig):
    return np.exp(-((d / sig) ** 2) / 2) / sig


def normalized_gaussian(d, sig):
    return (np.exp(-((d / sig) ** 2) / 2) / sig) * sig


class Neurone:
    def __init__(self, i, j, row, col, data):
        self.i = i
        self.j = j
        self.x = i / row
        self.y = j / col
        self.n = row * col
        self.maxData = np.max(data)

        # initializing weights randomly
        if len(data.shape) == 2:
            self.weight = np.max(data) * np.random.random(data.shape[1])
        else:
            self.weight = [[] for i in range(data.shape[0])]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self.weight[i].append(np.random.random(data.shape[2]))
            self.weight = np.array(self.weight)

    def update_weights(self, new):
        self.weight += new


class SOM:
    def __init__(self, row, column, data, nbEpoch, distance=dist_quad):
        # Définition des paramètres nécessaires à l'entraînement
        self.eps0 = 0.9
        self.epsEnd = 0.01
        self.epsilon = self.eps0
        self.epsilon_stepping = (self.epsEnd - self.eps0) / nbEpoch

        self.sig0 = 0.5
        self.sigEnd = 0.025
        self.sigma = self.sig0
        self.sigma_stepping = (self.sigEnd - self.sig0) / nbEpoch

        self.row = int(row)  # nombre de neurones choisis pour modéliser les données
        self.column = int(column)
        self.maxdata = np.max(data)
        self.data = np.array(data) / self.maxdata

        # Initialisation de la grille
        self.nodes = [[] for i in range(self.row)]
        for i in range(self.row):
            for j in range(self.column):
                self.nodes[i].append(Neurone(i, j, self.row, self.column, self.data))
        self.nodes = np.array(self.nodes)

        # We store here the distances between nodes
        # TODO : make it work with a non-square arrays and multiple dimensions
        self.MDist = np.empty_like(self.nodes)
        for i in range(self.row):
            for j in range(self.column):
                self.MDist[i][j] = (self.nodes[i][0].x - self.nodes[j][0].x) ** 2

    def winner(self, vector, distance=dist_quad):
        dist = np.empty_like(self.nodes)
        for i in range(self.row):  # Computes the distances between the tested vector and all nodes
            for j in range(self.column):
                dist[i][j] = distance(self.nodes[i, j].weight, vector)
        return np.unravel_index(np.argmin(dist, axis=None), dist.shape)  # Returning the Best Matching Unit's index.

    def train(self, k, epochTime, f=normalized_gaussian, distance=dist_quad):
        if k % epochTime == 0:
            self.epsilon += self.epsilon_stepping
            self.sigma += self.sigma_stepping
            self.generate_random_list()

        # The training vector is chosen randomly
        vector_coordinates = self.unique_random_vector()
        vector = self.data[vector_coordinates]

        # Getting the Best matching unit
        bmu = self.winner(vector, distance)
        self.updating_weights(bmu, vector, f)

        return vector_coordinates, bmu[0], bmu[1]

    def updating_weights(self, bmu, vector, f=normalized_gaussian):
        # Updating weights of all nodes
        for i in range(self.row):
            for j in range(self.column):
                dist = np.sqrt(self.MDist[i, bmu[0]] + self.MDist[j, bmu[1]])/np.sqrt(2)  # Normalizing the distances
                self.nodes[i, j].weight += f(dist, self.sigma)*self.epsilon*(vector-self.nodes[i, j].weight)

    def fully_random_vector(self):
        return np.random.randint(np.shape(self.data)[0])

    def unique_random_vector(self):
        return self.vector_list.pop(0)

    def generate_random_list(self):
        self.vector_list = list(range(len(self.data)))
        random.shuffle(self.vector_list)

    def getmap(self):
        map = [[] for i in range(self.row)]
        for i in range(self.row):
            for j in range(self.column):
                map[i].append(self.nodes[i, j].weight)

        return np.array(map) * self.maxdata

    def getmaplist(self):
        map = []
        for i in range(self.row):
            for j in range(self.column):
                map.append(self.nodes[i, j].weight)

        return np.array(map) * self.maxdata
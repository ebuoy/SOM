import numpy as np


def dist_quad(x, y):
    return np.sum((x - y) ** 2)


def gauss(d, sig):
    return np.exp(-((d / sig) ** 2) / 2) / sig


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
    def __init__(self, row, column, data, nbiter, distance=dist_quad):

        # Définition des paramètres nécessaires à l'entraînement
        self.eps0 = 0.01
        self.epsmax = 0.1
        self.epsilon = self.eps0
        self.epsilon_stepping = (self.epsmax - self.eps0) / nbiter

        self.sig0 = 1 / 4 * 10 ** (-1)
        self.sigmax = np.sqrt(10 ** (-1))
        self.sigma = self.sig0
        self.sigma_stepping = (self.sigmax - self.sig0) / nbiter

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

        # La grille est initialisée de manière aléatoire

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

    def train(self, f=gauss, distance=dist_quad):
        self.epsilon += self.epsilon_stepping
        self.sigma += self.sigma_stepping

        # The training vector is chosen randomly
        vector_coordinates = self.random_vector_coordinates()
        vector = self.data[self.random_vector_coordinates()]

        # Getting the Best matching unit
        bmu = self.winner(vector, distance)

        self.updating_weights(bmu, vector, f)

        return vector_coordinates, bmu[0], bmu[1]

    def updating_weights(self, bmu, vector, f=gauss):
        # Updating weights of all nodes
        for i in range(self.row):
            for j in range(self.column):
                dist = np.sqrt(self.MDist[i][bmu[0]] + self.MDist[j][bmu[1]])
                self.nodes[i, j].weight += f(dist, self.sigma)*self.epsilon*(vector-self.nodes[i, j].weight)

    def random_vector_coordinates(self):
        return np.random.randint(np.shape(self.data)[0])

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
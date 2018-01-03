import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection

def distquad(x,y):
    n=np.shape(x)[0]
    dist=0
    for i in range (n):
        dist+=(x[i]-y[i])**2
    return np.sqrt(dist)

def gauss(d,sig):
    return (np.exp(-((d/sig)**2)/2))/(sig)

class Neurone:
    def __init__(self, i, j, row,col,data):
        self._i=i
        self._j=j
        self._x=i/row
        self._y=j/col
        self._n=row*col
        if len(data.shape)==2:
            self._weight=np.max(data)*np.random.random(data.shape[1])
        else:
            self._weight=[[] for i in range (data.shape[0])]
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    self._weight[i].append(np.max(data)*np.random.random(data.shape[2]))
            self._weight=np.array(self._weight)
        
class SOM:
    def __init__(self,row,column,data):
        
        #Définition des paramètres nécessaires à l'entraînement
        self._eps0=0.01
        self._epsmax=0.1
        
        self._sig0=1/4*10**(-1)
        self._sigmax=np.sqrt(10**(-1))
        
        self._row=int(row)#nombre de neurones choisis pour mod�liser les données
        self._column=int(column)
        self._data=np.array(data)
        
        #Initialisation de la grille
        self._nodes=[[] for i in range(self._row)]
        for i in range (self._row):
            for j in range (self._column):
                self._nodes[i].append(Neurone(i,j,self._row,self._column,self._data))
        self._nodes=np.array(self._nodes)
                #La grille est initialisée de manière aléatoire
        
    def winner(self,vector,distance=distquad):#Par défaut, la distance utilisée est la distance quadratique
        row=self._row
        column=self._column
        dist=[[]for i in range (row)]
        """dist=np.zeros((row,column))
        
        for i in range (row):
            for j in range (column):
                dist[i,j]=distance(self._nodes[i,j]._weight,vector)"""
        for i in range(row):
            for j in range(column):
                dist[i].append(distance(self._nodes[i,j]._weight,vector))
        dist=np.array(dist)
        min=np.argmin(dist)
        iwin=0
        jwin=min
        while jwin>=0:
            jwin-=column
            iwin+=1
        iwin-=1
        jwin+=column
        return(iwin,jwin)
    
    def train(self,k,nbiter,f=gauss,distance=distquad):
        
        eps=self._eps0+(self._epsmax-self._eps0)*(nbiter - k)/nbiter
        sig=self._sig0+(self._sigmax-self._sig0)*(nbiter - k)/nbiter
        
        #eps=self._eps0*(self._epsmax/self._eps0)**((nbiter-k)/nbiter)
        #sig=self._sig0*(self._sigmax/self._sig0)**((nbiter-k)/nbiter)
        
        #Pour l'apprentissage, le vecteur est choisi au hasard
        coordvect=np.random.randint(np.shape(self._data)[0])
        vector=self._data[coordvect]
        iwin,jwin=self.winner(vector)
        self._nodes[iwin,jwin]._weight+=eps*gauss(distance(self._nodes[iwin,jwin]._weight,vector),sig)*(vector-self._nodes[iwin,jwin]._weight)

        #Les voisins du gagnant subissent aussi les effets du changement
        
        for i in range(self._row):
            for j in range(self._column):
                if i!=iwin or j!=jwin:
                    coeff_dist_win_ij=gauss(distance(np.array([self._nodes[iwin,jwin]._x,self._nodes[iwin,jwin]._y]),np.array([self._nodes[i,j]._x,self._nodes[i,j]._y])),sig)
                    #Coefficient permettant de déterminer le taux d'apprentissage de tous les voisins
                    self._nodes[i,j]._weight+=coeff_dist_win_ij*eps*(vector-self._nodes[i,j]._weight)
        return coordvect,(iwin,jwin)
    
    def getmap(self):
        
        map=[[] for i in range(self._row)]
        for i in range(self._row):
            for j in range(self._column):
                map[i].append(self._nodes[i,j]._weight)
        
        return np.array(map)
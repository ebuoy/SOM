import numpy as np

def distquad(x,y):
    n=np.shape(x)[0]
    dist=0
    for i in range n:
        dist+=(x-y)**2
    return np.squrt(dist)/n

class Neurone:
    def __init__(self, i, j, n,data):
        self._i=i
        self._j=j
        self._x=i/n
        self._y=j/n
        self._weight=np.maximum(data)*np.random.rand(np.shape(data[2]))
        
class SOM:

    def __init__(self,row,column,data):
        
        #Définition des paramètres nécessaires à l'entraînement
        self._eps0=0,01
        self._espmax=0,1
        self._sig0=
        
        self._row=row #nombre de neurones choisis pour mod�liser les données
        self._column=column
        self._data=np.array(data)
        
        #Initialisation de la grille
        self._nodes=np.zeros(row,column) 
        for i in range row:
            for j in range column:
                self._node[i,j]=Neurone(i,j,row*column,data)
                #La grille est initialisée de manière aléatoire
        
        
    def winner(vector,distance=distquad):#Par défaut, la distance utilisée est la distance quadratique
        row=self._row
        column=self._column
        dist=np.zeros(row,column)
        
        for i in range row:
            for j in range column:
                dist[i,j]=distance(self._nodes[i,j]._weight,vector)
        min=np.argmin(dist)
        iwin=0
        jwin=min
        while jwin>=0:
            jwin-=column
            iwin+=1
        iwin-=1
        jwin+=column
        return(iwin+column,jwin+1)
    
    def train(distance=distquad):
        t=0
        #Début de l'apprentissage, le premier vecteur stimulus est choisi au hasard
        vector_init=self._data(np.rint(np.shape(self._data)[0]*np.random.rand()),np.rint(np.shape(self._data)[1]*np.random.rand()))
        
        iwin,jwin=self.winner(vector_init)
        
        self._nodes[iwin,jwin]+=mu*np.exp(-distance(self._nodes[iwin,jwin],vector_init)**2/(2*sigma**2))*(sel.nodes[iwin,jwin]-vector_init)
        
        
import numpy as np

def distquad(x,y):
    n=np.shape(x)[0]
    dist=0
    for i in range (n):
        dist+=(x-y)**2
    return np.squrt(dist)/n

def gauss(d,sig):
    return (np.exp(-(d/sig)**2)/sig)

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
        self._eps0=0.01
        self._espmax=0.1
        self._sig0=0.01
        self._sigmax=0.1
        
        self._row=row #nombre de neurones choisis pour mod�liser les données
        self._column=column
        self._data=np.array(data)
        
        #Initialisation de la grille
        self._nodes=np.zeros(row,column) 
        for i in range (row):
            for j in range (column):
                self._node[i,j]=Neurone(i,j,row*column,data)
                #La grille est initialisée de manière aléatoire
        
        
    def winner(vector,distance=distquad):#Par défaut, la distance utilisée est la distance quadratique
        row=self._row
        column=self._column
        dist=np.zeros(row,column)
        
        for i in range (row):
            for j in range (column):
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
    
    def train(i,nbiter,f=gauss,distance=distquad):
        #Pour l'apprentissage, le vecteur est choisi au hasard
        eps=self._eps0+(self._epsmax-self.eps0)*(nbiter - i)/nbiter
        sig=self._sig0+(self._sigmax-self.sig0)*(nbiter - i)/nbiter
        
        vector=self._data(np.rint(np.shape(self._data)[0]*np.random.rand()),np.rint(np.shape(self._data)[1]*np.random.rand()))
        iwin,jwin=self.winner(vector)
        self._nodes[iwin,jwin]+=eps*np.gauss(distance(self._nodes[iwin,jwin],vector))*(self.nodes[iwin,jwin]-vector)
            

    
    # ---------------------
    # Exemple 1 test dans [0,1]
    
    data = np.random.rand(10,10)
    
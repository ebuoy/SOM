import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection

nbtest=11
path=r'C:\Users\Emeline\Documents\Cours\ENSMN\2A\Parcours Recherche\Carte de Kohonen\Test %d'%(nbtest)
def distquad(x,y):
    n=np.shape(x)[0]
    dist=0
    for i in range (n):
        dist+=(x[i]-y[i])**2
    return np.sqrt(dist)

def gauss(d,sig):
    return (np.exp(-((d/sig)**2)/2))/(sig*np.sqrt(2*np.pi)))

class Neurone:
    def __init__(self, i, j, n,data):
        self._i=i
        self._j=j
        self._x=i/n
        self._y=j/n
        self._n=n
        self._weight=np.max(data)*np.random.random(np.shape(data)[1])
        
class SOM:
    def __init__(self,*args):
        
        #Définition des paramètres nécessaires à l'entraînement
        self._eps0=0.01
        self._epsmax=0.1
        self._sig0=10**(-1)
        self._sigmax=np.sqrt(10**(-1))
        #self._sig0=10**(-2)
        #self._sigmax=10**(-1)
        
        self._row=int(args[0])#nombre de neurones choisis pour mod�liser les données
        self._column=int(args[1])
        self._data=np.array(args[2])
        
        #Initialisation de la grille
        self._nodes=[[] for i in range(self._row)]
        for i in range (self._row):
            for j in range (self._column):
                self._nodes[i].append(Neurone(i,j,self._row*self._column,self._data))
        self._nodes=np.array(self._nodes)
                #La grille est initialisée de manière aléatoire
        
        
    def winner(self,vector,distance=distquad):#Par défaut, la distance utilisée est la distance quadratique
        row=self._row
        column=self._column
        dist=np.zeros((row,column))

        
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
        return(iwin,jwin)
    
    def train(self,i,nbiter,f=gauss,distance=distquad):
        
        eps=self._eps0+(self._epsmax-self._eps0)*(nbiter - i)/nbiter
        sig=self._sig0+(self._sigmax-self._sig0)*(nbiter - i)/nbiter
        
        #eps=self._eps0*(self._epsmax/self._eps0)**((nbiter-i)/nbiter)
        #sig=self._sig0*(self._sigmax/self._sig0)**((nbiter-i)/nbiter)
        
        #Pour l'apprentissage, le vecteur est choisi au hasard
        vector=self._data[np.random.randint(np.shape(self._data)[0])]
        iwin,jwin=self.winner(vector)
        #self._nodes[iwin,jwin]._weight+=eps*gauss(distance(self._nodes[iwin,jwin]._weight,vector),sig)*(vector-self._nodes[iwin,jwin]._weight)
        self._nodes[iwin,jwin]._weight+=eps*(vector-self._nodes[iwin,jwin]._weight)

        #Les voisins du gagnant subissent aussi les effets du changement
        
        for i in range(self._row):
            for j in range(self._column):
                if i!=iwin and j!=jwin:
                    coeff_dist_win_ij=gauss(distance(np.array([self._nodes[iwin,jwin]._x,self._nodes[iwin,jwin]._y]),np.array([self._nodes[i,j]._x,self._nodes[i,j]._y])),sig)
                    #Coefficient permettant de déterminer le taux d'apprentissage de tous les voisins
                    #self._nodes[i,j]._weight+=coeff_dist_win_ij*eps*gauss(distance(self._nodes[i,j]._weight,vector),sig)*(vector-self._nodes[i,j]._weight)
                    self._nodes[i,j]._weight+=coeff_dist_win_ij*eps*(vector-self._nodes[i,j]._weight)

    # -------------------------
    # Exemple 1 test dans [0,1]
    
n=1000
data = np.array([np.random.random(2) for i in range(n)])
#data=np.array([np.array([i,j]) for i in range (n) for j in range(n)])/n #test avec une equirépartition des stimuli
winners=[]
carte = SOM(7,7,data)
nbiter= 60000

def figure(k,carte,data):

    if k!=0:
        carte.train(k,nbiter)

    # Dessin de la carte

    x=[data[i][0] for i in range(data.shape[0])]
    y=[data[i][1] for i in range(data.shape[0])]
    xc=[]
    yc=[]
    for i in range(5):
        for j in range(5):
                xc.append(carte._nodes[i][j]._weight[0])
                yc.append(carte._nodes[i][j]._weight[1])
    
    lines=[]
    
    for i in range(carte._row-1):
        for j in range(carte._column-1):
            lines.append([carte._nodes[i,j]._weight,carte._nodes[i-1,j]._weight])
            lines.append([carte._nodes[i,j]._weight,carte._nodes[i,j-1]._weight])
    
    for i in range(carte._row):
        lines.append([carte._nodes[i,carte._column-1]._weight,carte._nodes[i-1,carte._column-1]._weight])
        lines.append([carte._nodes[carte._row-1,i]._weight,carte._nodes[carte._row-1,i-1]._weight])
    
    plt.clf()    
    fig=plt.axes()
    plt.scatter(x,y)
    plt.scatter(xc,yc,color='r')
    lc=LineCollection(lines,colors='r',linewidths=1)
    fig.add_collection(lc)
    name=('%s/itération %d.png'%(path,k))
    plt.savefig(name)

os.makedirs(path)
for i in range (nbiter+1):
    if i<500:
       figure(i,carte,data)
    if 500<=i<nbiter-10000:
        if i%1000==0:
            figure(i,carte,data)
    if nbiter-10000<=i<=nbiter:
        if i%100==0:
            figure(i,carte,data)
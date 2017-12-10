from classe.SOM import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
nbtest= "sigma eps linéaire"
path=r'C:\Users\Emeline\Documents\Cours\ENSMN\2A\Parcours Recherche\Carte de Kohonen\Test %s'%(nbtest)
    
n=1000
data = np.array([np.random.random(2) for i in range(n)])
#data=np.array([np.array([i,j]) for i in range (n) for j in range(n)])/n #test avec une equirépartition des stimuli
winners=[]
carte = SOM(7,7,data)
nbiter=10000

def figure(k,carte,data):

    # Dessin de la carte

    x=[data[i][0] for i in range(data.shape[0])]
    y=[data[i][1] for i in range(data.shape[0])]
    xc=[]
    yc=[]
    for i in range(carte._row):
        for j in range(carte._column):
                xc.append(carte._nodes[i][j]._weight[0])
                yc.append(carte._nodes[i][j]._weight[1])
    
    lines=[]
    
    for i in range(carte._row-1):
        for j in range(carte._column-1):
            lines.append([carte._nodes[i,j]._weight,carte._nodes[i+1,j]._weight])
            lines.append([carte._nodes[i,j]._weight,carte._nodes[i,j+1]._weight])
    
    for i in range(1,carte._row):
        lines.append([carte._nodes[carte._row-1,i]._weight,carte._nodes[carte._row-1,i-1]._weight])
        lines.append([carte._nodes[i,carte._column-1]._weight,carte._nodes[i-1,carte._column-1]._weight])
        
    plt.clf()    
    fig=plt.axes()
    plt.scatter(x,y,color='b',marker='x')
    plt.scatter(xc,yc,color='r')
    lc=LineCollection(lines,colors='r',linewidths=1)
    fig.add_collection(lc)

    name=('%s/itération %d.png'%(path,k))
    plt.savefig(name)
    
os.makedirs(path)
    
for i in range (nbiter+1):
    if i==0:
        figure(0,carte,data)
    
    if 0<i<500:
        carte.train(i,nbiter)
        figure(i,carte,data)
    if 500<=i<nbiter-10000:
        carte.train(i,nbiter)
        if i%1000==0:
            figure(i,carte,data)
    if nbiter-10000<=i<=nbiter:
        carte.train(i,nbiter)
        if i%100==0:
            figure(i,carte,data)
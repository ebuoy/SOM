from classe.SOM import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
os.chdir('Test 2D-3D')
nbtest= "\Test cercles reliés 3"
path=os.getcwd()+nbtest

#------ Données random dans [0,1]
n=500
#data = np.array([np.random.random(2) for i in range(n)])
#data=np.array([np.array([i,j]) for i in range (n) for j in range(n)])/n #test avec une equirépartition des stimuli

#------- On prend des données en paquets distincts
"""o1=np.array([3,4])
r1=2
o2=np.array([15,10])
r2=5

data1=[]
data2=[]

for i in range(500):
    p=np.random.random()
    if p>0.5:
        a1=o1[0]+r1*np.random.random()
        a2=o2[0]+r2*np.random.random()

    else:
        a1=o1[0]-r1*np.random.random()
        a2=o2[0]-r2*np.random.random()
        
    pprime=np.random.random()
    if pprime>0.5:
        b1=o1[1]+np.sqrt(r1**2-(a1-o1[0])**2)*np.random.random()
        b2=o2[1]+np.sqrt(r2**2-(a2-o2[0])**2)*np.random.random()
        
    else:
        b1=o1[1]-np.sqrt(r1**2-(a1-o1[0])**2)*np.random.random()
        b2=o2[1]-np.sqrt(r2**2-(a2-o2[0])**2)*np.random.random()
    data1.append([a1,b1])
    data2.append([a2,b2])
    
    
data=np.array(data1+data2)"""

#----- On prend 2 paquets de données reliées par une barres

o1=np.array([3,4])
o2=np.array([21,4])
r=4


data1=[]
data2=[]
data3=[]

"""for i in range(n):
    p=np.random.random()
    if p>0.5:
        a1=o1[0]+r*np.random.random()
        a2=o2[0]+r*np.random.random()

    else:
        a1=o1[0]-r*np.random.random()
        a2=o2[0]-r*np.random.random()
        
    pprime=np.random.random()
    if pprime>0.5:
        b1=o1[1]+np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
        b2=o2[1]+np.sqrt(r**2-(a2-o2[0])**2)*np.random.random()
        
    else:
        b1=o1[1]-np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
        b2=o2[1]-np.sqrt(r**2-(a2-o2[0])**2)*np.random.random()
    psec=np.random.random()
    
    a3=(o1[0]+(r/2)*np.sqrt(3))+(o2[0]-o1[0]-r*np.sqrt(3))*np.random.random()
    
    if psec>0.5:
        b3=o1[1]+(r/2)*np.random.random()
    else:
        b3=o1[1]-(r/2)*np.random.random()
    data1.append([a1,b1])
    data2.append([a2,b2])
    data3.append([a3,b3])"""
data=np.array(data1+data2+data3)

#--- Données en cercle
data=[]
for i in range(n):
    p=np.random.random()
    if p>0.5:
        a1=o1[0]+r*np.random.random()
        a2=o2[0]+r*np.random.random()
    data.append([a1,a2])

data=np.array(data)

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
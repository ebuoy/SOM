from classe.SOM import *
import scipy.misc as msc
from PIL import Image
import numpy as np

file="\\Audrey.jpg"
fileC="\\AudreyC"
path=r".\Compression\image"

pathorigin=path+file

im=Image.open(pathorigin)

L,H=im.size
L,H=int(L),int(H)

px=im.getdata() #Cette librairie permet de faire le travail sur une image en niveau de gris
pxarray=np.array(px)

pxmatrix=[[] for i in range(H)]

for i in range(H):
    for j in range(L):
        if type(pxarray[0])==type(pxarray):
            pxmatrix[i].append(pxarray[i*L+j][0])
        elif type(pxarray[0])==int:
            pxmatrix[i].append(pxarray[i*L+j])
        

### Apprentissage

h,l=10,10 #largeur et hauteur des imagettes
nl,nh=L//l,H//h #nombre d'imagette par ligne et par colonne

if nl!= L/l:
    L=l*nl
    
if nh!= H/h:
    H=h*nh

datamat = []

for m in range(0,H,h):
    for i in range(0,L,l):
        list = []
        for j in range(0,h):
            d = []
            for k in range(0,l):
                d.append(pxmatrix[j+m][k+i])
            list += d
        datamat.append(list)
        
datamat=np.array(datamat)

#datamat est la liste des imagettes de tailles h*l
datacomp=[0 for i in range(nh*nl)] #datacomp est la liste du numéro du neurone vainqueur pour l'imagette correspondante
n=10 #Il y a 100 neurones dans le réseau

carte=SOM(n,n,datamat)

nbiter=nh*nl*10000

for i in range(nbiter):
    vect,iwin,jwin=carte.train(i,nbiter)
    datacomp[vect]=iwin*n+jwin
map2=[]

map=np.round(carte.getmaplist())
map=np.round(map)
map=map.astype(int)

for i in range(len(map)):
    for j in range(len(map[0])):
        map2.append(map[i,j])

### Compression
pathdest=r"C:.\Compression\image"

Comp=open(pathdest+fileC,'wb')

Comp.write(str(len(datacomp)).encode())
Comp.write('\n'.encode())
Comp.write(str(H).encode())
Comp.write('\n'.encode())
Comp.write(str(L).encode())
Comp.write('\n'.encode())
Comp.write(str(h).encode())
Comp.write('\n'.encode())
Comp.write(str(l).encode())
Comp.write('\n'.encode())
Comp.write(bytes(datacomp+map2))#len(datacomp)=1024
Comp.close()
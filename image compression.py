from classe.SOM import *
import scipy.misc as msc
from PIL import Image
import numpy as np

file="\orcwhale.jpg"
fileC="\orcwhaleC"
path=r"C:\Users\Emeline\Documents\Cours\ENSMN\2A\Parcours Recherche\Carte de Kohonen\Compression\image"

pathorigin=path+file

im=Image.open(pathorigin)

L,H=im.size
L,H=int(L),int(H)

px=im.getdata() #Cette librairie permet de faire le travail sur une image en niveau de gris
pxarray=np.array(px)

pxmatrix=[[] for i in range(L)]

for i in range(H):
    for j in range(L):
        pxmatrix[i].append(pxarray[i*L+j][0])

### Apprentissage

l,h=20,20 #largeur et hauteur des imagettes
nl,nh=L//l,H//h #nombre d'imagette par ligne et colonne
datamat=[]
for k in range (nh):
    for p in range(nl):
        datamat.append(pxmatrix[k*h][p*l:(p+1)*l]+pxmatrix[k*h+1][p*l:(p+1)*l])

datamat=np.array(datamat)
#data est la liste des imagettes de tailles h*l
datacomp=[0 for i in range(nh*nl)] #datacomp est la liste du numéro du neurone vainqueur pour l'imagette correspondante
n=5 #Nombre de neurone du réseau

carte=SOM(n,n,datamat)

nbiter=60000

for i in range(nbiter):
    vect,winner=carte.train(i,nbiter)
    datacomp[vect]=winner
datacomp=np.array(datacomp)
datacomp=datacomp.astype(bytes)

map=np.round(carte.getmap())
map=np.round(map)
map=map.astype(str)
map=map.tolist()
completesize=[H,L]
imagettesize=[h,l]
#completesize=np.array((H,L))
#imagettesize=np.array((h,l))
map=np.array(map)
map=map.astype(bytes)
#completesize=np.array((H,L)).astype(str)
#imagettesize=np.array((h,l)).astype(str)
#completesize=np.array((H,L))
#imagettesize=np.array((h,l))
### Compression
pathdest=r"C:\Users\Emeline\Documents\Cours\ENSMN\2A\Parcours Recherche\Carte de Kohonen\Compression\imageC"

Comp=open(pathdest+fileC,'wb')

Comp.write(bytes(map))
Comp.write('\n'.encode())
Comp.write(bytes(completesize))
Comp.write('\n'.encode())
Comp.write(bytes(imagettesize))
Comp.write('\n'.encode())
Comp.write(datacomp)
Comp.close()

from classe.SOM import *
from PIL import Image
#from SOM import *  # Use this for the Cython speedboost
import numpy as np
np.set_printoptions(threshold=np.inf)  # used to display numpy arrays in full

file="Audrey.png"
fileC="AudreypC"
path=r"./Compression/image/"

pathorigin=path+file

im=Image.open(pathorigin)

L,H=im.size
L,H=int(L),int(H)

px = im.getdata()  # Cette librairie permet de faire le travail sur une image en niveau de gris
pxarray = np.array(px)

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

# datamat est la liste des imagettes de tailles h*l
datacomp = np.zeros(nh*nl, int)  # datacomp est la liste du numéro du neurone vainqueur pour l'imagette correspondante
old = datacomp

n = 10  # Il y a 100 neurones dans le réseau

nbEpoch = 1000
epochTime = nh*nl
nbiter = epochTime*nbEpoch


carte = SOM(n, n, datamat, nbEpoch)


def display_som(som_list):
    px2 = []
    lst2 = ()
    for i in range(n):
        lst = ()
        for j in range(n):
            som_list[i * n + j] = som_list[i * n + j].reshape((h, l))
            lst = lst + (som_list[i * n + j],)
        px2.append(np.concatenate(lst, axis=1))
        lst2 += (px2[i],)
    px = np.concatenate(lst2, axis=0)
    som_image = Image.fromarray(px)
    som_image.show()
    return som_image


def compute_mean_error(datacomp, datamat, SOMList):
    error = np.zeros(len(datacomp))
    for i in range(len(datacomp)):
        error[i] = np.mean(np.abs(datamat[i] - SOMList[datacomp[i]]))
    return np.mean(error)


for i in range(nbiter):
    vect, iwin, jwin = carte.train(i, epochTime)
    datacomp[vect] = iwin*n+jwin
    if i%epochTime == 0:
        print("Epoch : ", i//epochTime+1, "/", nbEpoch)
        diff = np.count_nonzero(datacomp - old)
        print("Changed values :", diff)
        print("Mean error : ", compute_mean_error(datacomp, datamat, carte.getmaplist()))
        old = np.array(datacomp)

display_som(carte.getmaplist())
# file.save("./Compression/SOM.bmp")

datacomp = datacomp.tolist()
map2 = []


map = np.round(carte.getmaplist())
map = np.round(map)
map = map.astype(int)

for i in range(len(map)):
    for j in range(len(map[0])):
        map2.append(map[i,j])

### Compression
pathdest=r"./Compression/"

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
Comp.write(str(datacomp+map2).encode())  # len(datacomp) = 1024
Comp.close()
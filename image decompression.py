from tkinter import *
import scipy.misc as msc
from PIL import Image
import numpy as np
from classe.SOM import *


## Décompression

path=r"./Compression/"
fileC="AudreypC"

pathdest=r"./Compression/"
fileD="AudreypD.bmp"

pathC=path+fileC
imageC=open(pathC,'rb')
dataC=imageC.read()
imageC.close()

values=dataC.split('\n'.encode(),5)
len_datacomp=int(values[0])
H,L=int(values[1]),int(values[2])
h,l=int(values[3]),int(values[4])
data = values[5].decode().replace('[',"").replace(']',"").split(',')
for i in range(len(data)):
    data[i] = int(data[i])
datacomp=data[0:len_datacomp]
map2=data[len_datacomp:]
nl,nh=L//l,H//h #nombre d'imagette par ligne et colonne

## Reconstitution de l'image
data=[] #On recréée les données avec les imagettes
map=[] #On recréée la carte avec les imagettes

for k in range(0,len(map2),h*l):
    map.append(map2[k:k+h*l])
    
for i in range(len(datacomp)):
    data.append(map[datacomp[i]])

NH = (h-1)*nh
NL = (l-1)*nl
im = []

for m in range(0,len(data),nl):
    for i in range(0,len(data[0]),h):
        for j in range(nl):
            for k in range(h):
                im.append(int(data[j+m][k+i]))
  
            
px = []
for i in range(0,len(im),L):
    px.append(np.array(im[i:i+L]))

px1 = np.array(px, 'uint8')
file = Image.fromarray(px1) # régler le problème d'affichage du fichier
file.show()
pxb = bytes(im)
file2 = Image.frombytes(mode='L',size=(L,H),data=pxb)
file2.save(pathdest+fileD)


        
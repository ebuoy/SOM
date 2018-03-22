from classe.SOM import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from tkinter import *
os.chdir('Test 2D-3D')


#------ Données random dans [0,1]
def square(n):
    data = np.array([np.random.random(2) for i in range(n)])
    return data
def eq_square(n):
        data=np.array([np.array([i,j]) for i in range (n) for j in range(n)])/n #test avec une equirépartition des stimuli
        return data

#------- On prend des données en paquets distincts
def sep_circle(n):
    n = n//4
    o1=np.array([3,4])
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
        
        
    data=np.array(data1+data2)
    
    return data

#----- On prend 2 paquets de données reliées par une barres
def weights(n):
    n = n//5
    o1=np.array([3,4])
    o2=np.array([21,4])
    r=4
    
    
    data1=[]
    data2=[]
    data3=[]
    
    for i in range(n):
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
        data3.append([a3,b3])
    data=np.array(data1+data2+data3)
    return data

#--- Données en cercle
def circle(n):
    n = n//2
    data=[]
    for i in range(n):
        p=np.random.random()
        if p>0.5:
            a1=o1[0]+r*np.random.random()
        elif p<0.5:
            a1=o1[0]-r*np.random.random()
        pprime=np.random.random()
        if pprime>0.5:
            b1=o1[1]+np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
        elif pprime<0.5:
            b1=o1[1]-np.sqrt(r**2-(a1-o1[0])**2)*np.random.random()
    
        data.append([a1,b1])
    
    data=np.array(data)
    return data

#nbiter=30000

def figure(k,carte,data):

    # Dessin de la carte
    map=carte.getmap()
    x=[data[i][0] for i in range(data.shape[0])]
    y=[data[i][1] for i in range(data.shape[0])]
    xc=[]
    yc=[]
    for i in range(carte.row):
        for j in range(carte.column):
                xc.append(map[i][j][0])
                yc.append(map[i][j][1])
    
    lines=[]
    
    for i in range(carte.row-1):
        for j in range(carte.column-1):
            lines.append([map[i][j],map[i+1][j]])
            lines.append([map[i][j],map[i][j+1]])
    
    for i in range(1,carte.row):
        lines.append([map[carte.row-1][i],map[carte.row-1][i-1]])
        lines.append([map[i][carte.column-1],map[i-1][carte.column-1]])
        
    plt.clf()    
    fig=plt.axes()
    plt.scatter(x,y,color='b',marker='x')
    plt.scatter(xc,yc,color='r')
    lc=LineCollection(lines,colors='r',linewidths=1)
    fig.add_collection(lc)

    name=('%s/itération %d.png'%(path,k))
    plt.savefig(name)
    


"""for i in range (nbiter+1):
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
            figure(i,carte,data)"""
            
def train(carte,data,nbEpoch):
    for i in range (nbEpoch+1):
        carte.train(i,len(data))
        if i%100 == 0:
            figure(i,carte,data)
    
#Programme principal

    
def launch_SOM():
    
    global photo
    
    if n.get() == "":
        n_data = 400
    else:
        n_data = int(n.get())
        
    if var.get() == "Square":
        data = square(n_data)
    elif var.get() == "Equireparted square":
        data = eq_square(n_data)
    elif var.get() == "Two Seperated circles":
        data = sep_circle(n_data)
    elif var.get() == "Circle":
        data = circle(n_data)
    elif var.get() == "Kind of Weights":
        data = weights(n_data)
        
    if neur.get() == "":
        nb_neur = 5
    else:
        nb_neur = int(neur.get())
    
    if nbiter.get() == "":
        nb_iter = 25*n_data
        
    else:
        nb_iter = int(nbiter.get())
        
    fen.destroy()
    
    os.makedirs(path)
    
    carte = SOM(nb_neur,nb_neur,data,nb_iter)
    train(carte,data,nb_iter)
    
    fen1=Tk()
    can1 = Canvas(fen1, bg = 'dark grey', height=600, width=800)
    can1.pack(side=LEFT)
    bou1 = Button(fen1, text='Quit', command=fen1.destroy)
    bou1.pack()
    
    def refresh(ite):
        global photo
        nom1 = '%s\itération %s.png'%(path,ite)
        photo1 = PhotoImage(file=nom1)
        item = can1.create_image(400, 300, image=photo1)
        photo = photo1
        
    sca=Scale(fen1, orient='vertical', from_=0, to=nb_iter, resolution=200, tickinterval = 100, length = 500,label = 'Number of iteration', command = refresh)
    sca.pack()
    nom = '%s/itération %s.png'%(path,sca.get())
    photo = PhotoImage(file=nom)
    item = can1.create_image(400, 300, image=photo)
    fen1.mainloop()

    
def launch():
    
    """if var.get() == "Square":
        data = square(int(n.get()))
    elif var.get() == "Equireparted square":
        data = eq_square(int(n.get()))
    elif var.get() == "Two Seperated circles":
        data = sep_circle(int(n.get()))
    elif var.get() == "Circle":
        data = circle(int(n.get()))
    elif var.get() == "Kind of Weights":
        data = weights(int(n.get()))"""
        
    print(nbiter.get() == "")
    print(n.get() == "")
    print(neur.get() == "")
    print(var.get())
    #print(data)
    fen.destroy()
    
    
fen=Tk()
txt0 = Label(fen, text = "Name")
name = Entry(fen)
nbtest = "\%s"%name.get()
path = os.getcwd()+nbtest

txt1 = Label(fen, text = "Number of iteration")
nbiter = Entry(fen)

#On choisit quelle distribution de données on souhaite



txt2 = Label(fen, text = "Number of data :")
n = Entry(fen)
txt3 = Label(fen, text = "Number of neurons :")
neur = Entry(fen)
bou = Button(fen, text="Launch the SOM", command=launch_SOM)

txt0.grid(row=0)
txt1.grid(row=1)
txt2.grid(row=2)
txt3.grid(row=3)

name.grid(row=0,column=1)
nbiter.grid(row=1,column=1)
n.grid(row=2,column=1)
neur.grid(row=3,column=1)

txt4 = Label(fen, text = "What kind of data?")
txt4.grid(row =4)
var=StringVar()

rb1 = Radiobutton(fen,text="Square",value="Square",variable=var)
rb2 = Radiobutton(fen,text="Equireparted square",value="Equireparted square",variable=var)
rb3 = Radiobutton(fen,text="Two Seperated circles",value="Two Seperated circles",variable=var)
rb4 = Radiobutton(fen,text="Circle",value="Circle",variable=var)
rb5 = Radiobutton(fen,text="Kind of Weights",value="Kind of Weights",variable=var)


rb1.grid(row=5)
rb2.grid(row=6)
rb3.grid(row=7)
rb4.grid(row=8)
rb5.grid(row=9)

bou.grid(row=10)

fen.mainloop()



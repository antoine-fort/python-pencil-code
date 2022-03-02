# pvar_tra.py
#
# Read the pvarfiles from a mix of two species represented as tracers
#
# Authors :
# Antoine Fort (antoine.fort@obspm.fr)
#
#

import pencil_old as pc_old
import pencil as pc
import numpy as np
import matplotlib.pyplot as plt
import scipy as scp
from mpl_toolkits.mplot3d import Axes3D

def pvarTra(*args,**kwargs):

    """
    This code is usefull in a specific context. It concerns a mixture of two species
    represented as tracers delimited at an initial time by a spatial border.

    Signature:
    ----------

    pvarTra(varfile="pvar.dat",datadir="./data", proc=-1)

    Parameters
    ----------
     *var_file*: str
         Name of the VAR file.
         If not specified, use var.dat (which is the latest snapshot of the fields)

     *datadir*: str
         Directory where the data is stored.

     *proc*: int
         Processor to be read. If -1 read all and assemble to one array.

     *border_axis*: int
         Number of the axis to create the separation between the two species.

     *border*: float
         Value of the limit to create the separation between the two species.

    Returns
    ---------
    Instance of the pvarTra.dataTracers class.
    The data are stored as class members :
        *id* list of int :
            List of the id for all the tracers
        *pos* array 3* number of tracers :
            List of all the tracers positions along each axis
        *true_clr* : list of bool
            List of the nature of each tracer with a bool.
            The position in the list corresponds to the id.
        *nb* : int
            Number of the PVAR used
        *t* : float
            Time of the pvar used
        *scale* : list of int
            List of the tracers size to plot with scatter
        *grid_size* : list of float
            List of the borders of the box in this way [x0,x1,y0 ...

    Functions
    ---------
    This is a summarize of each function inside the class dataTracers
    (to see precisely each function read the documentation in each function)

    __init__
        Initialize the tracer populations with a initial PVAR file.

    keys
        Give all the members of the class.

    update
        Update which tracer is which species. Useless if you used updateBoundaries just before.

    slicePlot
        Select a slice to plot the tracers with scatter.
        Can change also the size for plotting.

    plot2d
        Plot a 2d slice (definied by slicePlot) along two axis

    plot3d
        Same as plot2d but in 3d

    updateBoundaries
        It updates the boundaries. Because we are using periodic boundaries
        some tracers can be at the other part of the simulation.

    nearestNeighbors
        It finds the nearest neighbors of each tracer.
        It gives a value in function of the two species in the nearest neighbors.

    create3dGrid
        It creates a 3d grid to interpolate nearestNeighbors.

    plot2dTraNeighField
        plot a slice of the create3dGrid

    Example
    ----------

    pvar=pvarTra("PVAR200")
    pvar.keys()
    pvar.update(200)
    pvar.slicePlot(-0.1,0.1,2)
    pvar.plot2d(0,1,2)
    pvar.updateBoundaries(0.05,200,250,5)
    pvar.update(250)
    pvar.slicePlot(-0.1,0.1,2)
    pvar.plot2d(0,1,2)
    pvar.plot3d()
    pvar.nearestNeighborsClr()
    pvar.create3dGrid([64,64,64])
    pvar.plot2dTraNeighField(32)

    """
    return(dataTracers(*args,**kwargs))

def sort_position(id,pos):
    pos_sort=np.zeros((3,len(id)))
    for k in range(0,len(id)):
        pos_sort[:,id[k]-1]=pos[:,k]
    return(pos_sort)


class dataTracers(object):

    def __init__(self,varfile="pvar.dat",datadir="./data",proc=-1,border_axis=0,border=0.0):

        pvar=pc_old.read_pvar(varfile,datadir="./data",proc=-1)
        var=pc.read.var(varfile[1:])
        id=pvar.ipars
        pos=np.array([pvar.xp,pvar.yp,pvar.zp])

        pos=sort_position(id,pos)

        true_color=np.array(["blue"]*len(id))
        for k in range(0,len(id)):
            if pos[border_axis][k]>border:
                true_color[k]="red"

        self.true_clr=true_color
        self.id=id
        self.pos=pos
        self.t=var.t
        self.nb=int(varfile[4:])
        self.scale=np.array([50]*len(self.id))

        box=pc.read.grid()
        self.grid_size=[-box.Lx/2,box.Lx/2,-box.Ly/2,box.Ly/2,-box.Lz/2,box.Lz/2]

    def keys(self):
        for i in self.__dict__.keys():
            print(i)


    def update(self,n):

        pvar=pc_old.read_pvar(varfile="PVAR"+str(n))

        pos=np.array([pvar.xp,pvar.yp,pvar.zp])
        id=pvar.ipars

        pos=sort_position(id,pos)

        self.nb=n
        self.pos=pos
        self.id=id

    def slicePlot(self,bslice,fslice,axis=3,size=50):

        self.scale=np.array([size]*len(self.id))

        for k in range(0,len(self.id)) :
            if self.pos[axis][k]<bslice or self.pos[axis][k]>fslice :
                self.scale[k]=0

    def plot2d(self,x,y,z,alpha=0.3):

        fig, ax = plt.subplots()
        ax.scatter(self.pos[x], self.pos[y], c=self.true_clr, s=self.scale, alpha=alpha, edgecolors='none')

        ax.legend()
        ax.grid(True)
        plt.title("map of the tracers with color red=ice blue=vapour PVAR")
        plt.savefig("./data/figures/plot2d"+str(self.nb)+".jpg")

    def plot3d(self,x=0,y=1,z=2,alpha=0.02):

        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.scatter(self.pos[x], self.pos[y], self.pos[z], c=self.true_clr, s=self.scale, alpha=alpha, edgecolors='none')
        ax.legend()
        ax.grid(True)

        plt.savefig("./data/figures/plot3d"+str(self.nb)+".jpg")

    def updateBoundaries(self,tbeg,tend,tstep):
        bsize=self.grid_size[1]
        for i in range(tbeg+tstep,tend+tstep,tstep):
            print("updateBoundaries reading of PVAR"+str(i))
            pvar=pc_old.read_pvar(varfile="PVAR"+str(i))
            pos=np.array([pvar.xp,pvar.yp,pvar.zp])
            id=pvar.ipars

            pos=sort_position(id,pos)

            for l in range(0,len(self.true_clr)):
                if (pos[0][l]*self.pos[0][l])<(-1)*((bsize/2)**2):
                    if self.true_clr[l]=="blue":
                        self.true_clr[l]="red"
                    else:
                        self.true_clr[l]="blue"

            self.pos=pos
            self.id=id
        self.nb=tend

    def nearestNeighborsClr(self,nb_neighbors=4):
        pos=self.pos.T
        tree=scp.spatial.cKDTree(pos)
        clrdiff=[]
        counter=0
        for l in range(len(pos)):
            if l > len(pos)/100*counter:
                print("nearestNeighborsClr creating the cKDTree :"+str(l/len(pos)*100))
                counter+=1
            d,i=tree.query(pos[l],nb_neighbors+1)
            clrcount=0
            for k in range(1,len(i)):
                if self.true_clr[i[k]]=="blue" :
                    clrcount-=1
                else:
                    clrcount+=1
            clrdiff.append(clrcount)
        self.clrdiff=np.array(clrdiff)

    def create3dGrid(self,part_size):
        grid_size=self.grid_size
        x=np.linspace(grid_size[0]+(grid_size[1]-grid_size[0])/(2*part_size[0]),grid_size[1]-(grid_size[1]-grid_size[0])/(2*part_size[0]),part_size[0])
        y=np.linspace(grid_size[2]+(grid_size[3]-grid_size[2])/(2*part_size[1]),grid_size[3]-(grid_size[3]-grid_size[2])/(2*part_size[1]),part_size[1])
        z=np.linspace(grid_size[4]+(grid_size[5]-grid_size[4])/(2*part_size[2]),grid_size[5]-(grid_size[5]-grid_size[4])/(2*part_size[2]),part_size[2])
        xx=[]
        yy=[]
        zz=[]
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    xx.append(x[i])
                    yy.append(y[j])
                    zz.append(z[k])
        inter_grid=scp.interpolate.griddata((self.pos[0],self.pos[1],self.pos[2]),self.clrdiff,(xx,yy,zz))

        values_ongrid=np.zeros((part_size[0],part_size[1],part_size[2]))
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    values_ongrid[i,j,k]=inter_grid[0]
                    inter_grid=inter_grid[1:]

        self.tra_neigh_field=np.array(values_ongrid)

    def plot2dTraNeighField(self,layer,x=0,y=1,z=2):
        fig, ax = plt.subplots()
        if z==0:
            plt.imshow(self.tra_neigh_field[layer,:,:].T, origin='lower', extent=[self.grid_size[x*2+0], self.grid_size[x*2+1],self.grid_size[y*2+0],self.grid_size[y*2+1]], interpolation='nearest', cmap='jet')
        if z==1:
            plt.imshow(self.tra_neigh_field[:,layer,:].T, origin='lower', extent=[self.grid_size[x*2+0], self.grid_size[x*2+1],self.grid_size[y*2+0],self.grid_size[y*2+1]], interpolation='nearest', cmap='jet')
        if z==2:
            plt.imshow(self.tra_neigh_field[:,:,layer].T, origin='lower', extent=[self.grid_size[x*2+0], self.grid_size[x*2+1],self.grid_size[y*2+0],self.grid_size[y*2+1]], interpolation='nearest', cmap='jet')
        plt.xlabel("x")
        plt.ylabel("y")
        cbar=plt.colorbar()
        cbar.ax.set_ylabel("ux")
        plt.title("2D graph of neighbors map (x,y) PVAR")
        plt.savefig("./data/figures/plot2dTraNeighField"+str(self.nb)+".jpg")



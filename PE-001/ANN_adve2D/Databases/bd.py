import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sqlite3


# Paramètres de la simulation
#
# Physiques
#
#  a,b = vitesses d'onde
#  xmin,ymin, lx_dom, ly_dom : définissent le domaine de calcul
#
# Numériques
#
# imax, jmax : Nombre de cellules de calcul
# cfl : nombre CFL
# itermax : Nombre d'itérations a effectuer
# freq_ech : Fréquence des sorties graphiques
#
# Vitesse d'onde
a = 1
b = 1

# Abscisse minimale
xmin=0
# Ordonnée minimale
ymin=0

# Longueur du domaine suivant x
l_dom_x=1.0
# Longueur du domaine suivant y
l_dom_y=1.0



# Nombre CFL
cfl=0.5

# Choix de l'initialisation
# iinlet=0 (créneau)
iinlet=0
# Choix du schéma
# 1 (schéma CIR ou de Roe)  2 (LW)
ischema=1

# Fréquence des sorties
freq_ech=1


db = sqlite3.connect("Table.db")

cur = db.cursor()

def add_to_db(P,F,V):

	cur.execute("INSERT INTO A VALUES ({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})".format(P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],F[0],F[1],F[2],F[3],V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7]))

	P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8] = P[6],P[3],P[0],P[7],P[4],P[2],P[8],P[5],P[2]
	F[0],F[1],F[2],F[3] = F[1],F[3],F[0],F[2]
	V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7] = V[2],V[3],V[6],V[7],V[0],V[1],V[4],V[5]
	V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7] = V[3],-V[2],V[7],-V[6],V[1],-V[0],V[5],-V[4]

	cur.execute("INSERT INTO A VALUES ({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})".format(P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],F[0],F[1],F[2],F[3],V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7]))

	P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8] = P[6],P[3],P[0],P[7],P[4],P[2],P[8],P[5],P[2]
	F[0],F[1],F[2],F[3] = F[1],F[3],F[0],F[2]
	V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7] = V[2],V[3],V[6],V[7],V[0],V[1],V[4],V[5]
	V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7] = V[3],-V[2],V[7],-V[6],V[1],-V[0],V[5],-V[4]

	cur.execute("INSERT INTO A VALUES ({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})".format(P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],F[0],F[1],F[2],F[3],V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7]))

	P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8] = P[6],P[3],P[0],P[7],P[4],P[2],P[8],P[5],P[2]
	F[0],F[1],F[2],F[3] = F[1],F[3],F[0],F[2]
	V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7] = V[2],V[3],V[6],V[7],V[0],V[1],V[4],V[5]
	V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7] = V[3],-V[2],V[7],-V[6],V[1],-V[0],V[5],-V[4]

	cur.execute("INSERT INTO A VALUES ({},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{})".format(P[0],P[1],P[2],P[3],P[4],P[5],P[6],P[7],P[8],F[0],F[1],F[2],F[3],V[0],V[1],V[2],V[3],V[4],V[5],V[6],V[7]))	

	



def rho_inlet1(x):
	return np.exp(-50*((x-0.5)**2))

def rho_inlet2(x):
    w = 2*np.pi*5
    if x>=np.pi/w and x<=9*np.pi/w:
        return 0.5*(1+np.cos(w*x))
    else:
        return 0

def rho_inlet(x):
    w = 2*np.pi*6
    return 0.5*(1+np.cos(w*x))

def rho_inlet4(x):
    w = 2*np.pi*5
    return 0.5*(1+np.cos(w*x))


def rho_ex1(i,j):
    R=np.sqrt((x[i]-1)**2+y[j]**2)   #ATTENTION (x-1) car on sait que l'origine du cercle est décalée à gauche
    return rho_inlet(1-R)

def rho_ex(i,j):
    R=np.sqrt((x[i]-1)**2+y[j]**2)   #ATTENTION (x-1) car on sait que l'origine du cercle est décalée à gauche
    t = (x[i]-2.5)**2+(y[j]-1.25)**2
    if t<4:
        return rho_inlet(1-R)
    elif t>4.1:
        return 0
    else:
        return (41-10*t)*rho_inlet(1-R)
    

def rho_ex3(i,j):
    R=np.sqrt((x[i]-1)**2+y[j]**2)   #ATTENTION (x-1) car on sait que l'origine du cercle est décalée à gauche
    t = (x[i]-2.5)**2+(y[j]-1.25)**2
    if t<3.5:
        return rho_inlet(1-R)
    elif t>3.6:
        return 0
    else:
        return (36-10*t)*rho_inlet(1-R)

def rho_ex4(i,j):
    R=np.sqrt((x[i]-1)**2+y[j]**2)   #ATTENTION (x-1) car on sait que l'origine du cercle est décalée à gauche
    w = 2*np.pi*5
    return np.cos(w*((x-0.25)**2+y**2))*rho_inlet(1-R)

def trapeze(xp,yp):
    N = len(xp)
    h = xp[2]-xp[1]
    s = 0
    for i in range(1,N-1):
        s += yp[i]
    return h*(yp[0]+yp[N-1] + 2*s)/2.0


def flux(k,l):
	I = []
	#horizontal haut
	ypp = rhoex[k-fac//2:k+fac//2]
	yp = [b[k+i]*ypp[i][l+fac//2] for i in range(len(ypp))]
	xp = x[k-fac//2:k+fac//2]
	I.append(trapeze(xp,yp))
	
	#vertical gauche
	ypp = rhoex[k-fac//2][l-fac//2:l+fac//2]
	yp = [a[l+j]*ypp[j] for j in range(len(ypp))]
	xp = y[l-fac//2:l+fac//2]
	I.append(trapeze(xp,yp))
	#vertical droite
	
	ypp = rhoex[k+fac//2][l-fac//2:l+fac//2]
	yp = [a[l+j]*ypp[j] for j in range(len(ypp))]
	xp = y[l-fac//2:l+fac//2]
	I.append(trapeze(xp,yp))

	#horizontal bas
	ypp = rhoex[k - fac//2 : k + fac//2]
	yp = [b[k+i]*ypp[i][l - fac//2] for i in range(len(ypp))]
	xp = x[k-fac//2:k+fac//2]
	I.append(trapeze(xp,yp))

	return I


imax = 50
jmax = 50

fac = 10

dx = l_dom_x/(fac*imax)
dy = l_dom_y/(fac*jmax)


x = [xmin + (i-1)*l_dom_x/(fac*imax) for i in range(1,fac*imax+1)]
y = [ymin + (j-1)*l_dom_y/(fac*jmax) for j in range(1,fac*jmax+1)]

a = np.array([ y_i for y_i in y])
b = np.array([ 1-x_i for x_i in x])

rhoex = [[0 for j in range(1,fac*jmax+1)] for i in range(1,fac*imax+1)]
for i in range(fac*imax):
    for j in range(fac*jmax):
        rhoex[i][j]=rho_ex(i,j)


for i in range(1,imax-1):
	i = fac*i
	for j in range(1,jmax-1):
		j = fac*j

		W = [rhoex[i-fac][j+fac],rhoex[i][j+fac],rhoex[i+fac][j+fac],rhoex[i-fac][j],rhoex[i][j],rhoex[i+fac][j],rhoex[i-fac][j-fac],rhoex[i][j-fac],rhoex[i+fac][j-fac]]
		V = [a[i+fac//2],b[i+fac//2],a[j-fac//2],b[j-fac//2],a[j+fac//2],b[j+fac//2],a[i-fac//2],b[i-fac//2]]
		add_to_db(W,flux(i,j),V)



X , Y = np.meshgrid(y,x)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.contour3D(X,Y,rhoex, 50)
plt.show()


db.commit()
db.close()
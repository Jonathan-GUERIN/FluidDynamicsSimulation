import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from joblib import load

import time

import json

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
xmin=0.1
# Ordonnée minimale
ymin=0.1

# Longueur du domaine suivant x
l_dom_x=0.99
# Longueur du domaine suivant y
l_dom_y=0.99

# Nombre de cellules
imax=1001
jmax=1001

# Nombre CFL
cfl=0.5

# Choix de l'initialisation
# iinlet=0 (créneau) ou 1 (gaussienne)
iinlet=1

# Choix du schéma
# 1 (schéma CIR ou de Roe)  2 (LW) 3 (RBC) 4 (Réseau de neurones)
ischema=4

# Fréquence des sorties
freq_ech=1

#dx et dy

dx = l_dom_x / (imax-1)
dy = l_dom_y / (jmax-1)

x = np.array([xmin + (i-1)*dx for i in range(1,imax+1)])
y = np.array([ymin + (j-1)*dy for j in range(1,jmax+1)])

a = np.array([ y_i for y_i in y])
b = np.array([ 1-x_i for x_i in x])

Phi = [[min(1,(dy * np.abs(A))/(dx * np.abs(B))) for A in a] for B in b]
Psi = [[min(1,(dx * np.abs(B))/(dy * np.abs(A))) for A in a] for B in b]

# Chargement du réseau
def f_loss(y_true, y_pred):
    
    dx_tensor=tf.constant(1/1001)
    dy_tensor=tf.constant(1/1001)
    e = 0.000001
    n = 2
    
    wobj_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in range(0,tf.shape(y_pred)[0]):        
        wobj_list = wobj_list.write(i,tf.add(tf.divide(tf.subtract(y_pred[i][0],y_pred[i][3]),dx_tensor),tf.divide(tf.subtract(y_pred[i][2],y_pred[i][1]),dy_tensor)))
        
    wobj = wobj_list.stack()
        
    mse = tf.reduce_mean(tf.square(tf.subtract(y_true,y_pred)))
    rest = e*pow(tf.maximum(0.0,-wobj),n)
    
    loss = mse + tf.reduce_mean(rest)
    
    return loss

reseau = tf.keras.models.load_model("Advection2D_contrainteLoss-1.h5",custom_objects={'f_loss': f_loss})
inputscaler = load('input_scaler.bin')
outputscaler = load('output_scaler.bin')



#Fonctions

def c1(xi):
    return xi

def c2(xi):
    return 1-xi

def f1_phys(rho,i,j):
    return a[j]*rho

def f2_phys(rho,i,j):
    return b[i]*rho

def a1_phys(rho,i,j):
    
    return a[j]

def a2_phys(rho,i,j):
    return b[i]

def q1_num(rhol,rhor,i,j):
    if ischema == 1:
        return abs(a[j])
    elif ischema == 2:
        return a[j]*dt

def q2_num(rhol,rhor,i,j):
    if ischema == 1:
        return abs(b[i])
    elif ischema == 2:
        return b[i]*dt

def rho_inlet(xi):
    if iinlet == 0:
        if xi > 0.2 and xi < 0.5:
            return 1
        else:
            return 0
    elif iinlet == 1:
        return np.exp(-50*((xi-0.5)**2))

def rho_ex(x,y):
    R=np.sqrt((x-1)**2+y**2)   #ATTENTION (x-1) car on sait que l'origine du cercle est décalée à gauche
    return rho_inlet(l_dom_x-R)

def predire_flux(i,j):
    # Cette fonction prédit les flux à la position (x[i],y[j])
    # pour les bords on donne la valeur 0 au parametres non définis 

    try:
        p1=rho[i-1][j+1]
    except:
        p1=0
    try:
        p2=rho[i][j+1]
    except:
        p2=0
    try:
        p3=rho[i+1][j+1]
    except:
        p3=0
    try:
        p4=rho[i-1][j]
    except:
        p4=0
    try:
        p5=rho[i][j]
    except:
        p5=0
    try:
        p6=rho[i+1][j]
    except:
        p6=0
    try:
        p7=rho[i-1][j-1]
    except:
        p7=0
    try:
        p8=rho[i][j-1]
    except:
        p8=0
    try:
        p9=rho[i+1][j-1]
    except:
        p9=0
    try:
        A1=a[j+1]
    except:
        A1=0
    try:
        B1=b[i]
    except:
        B1=0
    try:
        A2=a[j-1]
    except:
        A2=0
    try:
        B2=b[i]
    except:
        B2=0
    try:
        A3=a[j]
    except:
        A3=0
    try:
        B3=b[i-1] 
    except:
        B3=0
    try:
        A4=a[j]
    except:
        A4=0
    try:
        B4=b[i+1]
    except:
        B4=0

    
    data={'P1':[p1],  
              'P2':[p2],    
              'P3':[p3],  
              'P4':[p4],    
              'P5':[p5],      
              'P6':[p6],    
              'P7':[p7],  
              'P8':[p8],    
              'P9':[p9],
              'A1':[A1],         
              'B1':[B1],           
              'A2':[A2],       
              'B2':[B2],        
              'A3':[A3],          
              'B3':[B3],              
              'A4':[A4],      
              'B4':[B4]}    
    data_input=pd.DataFrame(data)
    data_input = inputscaler.transform(data_input)
    output=reseau.predict(data_input)
    Y_pred_unscaled = outputscaler.inverse_transform(output)
    # if p7 >= 0.5: 
    #     print(Y_pred_unscaled[0])
    return Y_pred_unscaled[0] # [F1,F2,F3,F4] une liste
    
            
    
    
#Affichage
print('Nb cellules suivant x : ' +str(imax))
print('Nb cellules suivant y : ' + str(jmax))
print('Abscisse minimale : ' + str(xmin))
print('Ordonnée minimale : ' +str(ymin))
print('Abscisse maximale : ' + str(xmin+l_dom_x,))
print('Ordonnée maximale : ' +str(ymin+l_dom_y))
print("Pas d'espace suivant x : "+str(dx))
print("Pas d'espace suivant y : "+str(dy))
print('Nombre CFL : '+str(cfl))
print('Fréquence des sorties : '+str(freq_ech))

#Création de x et y et de rho et rhoex

rho = [[0 for j in range(1,jmax+1)] for i in range(1,imax+1)]
for i in range(1,imax+1):
    for j in range(1,jmax+1):
        rho[i-1][j-1]=0
rhoex = [[0 for j in range(1,jmax+1)] for i in range(1,imax+1)]
for i in range(1,imax):
    for j in range(1,jmax):
        rhoex[i-1][j-1]=rho_ex(x[i-1],y[j-1])

print('La simulation démarre...')


#Conditions aux limites

#    Conditions d'entrée

for i in range(1,imax+1):
    rho[i-1][0]=rho_inlet(x[i-1])
for j in range(1,jmax+1):
    rho[0][j-1]=rho_inlet(x[0])

#    Conditions de sortie

for i in range(1,imax+1):
    rho[i-1][jmax-1]=rho[i-1][jmax-1-1]
for j in range(1,jmax+1):
    rho[imax-1][j-1]=rho[imax-1-1][j-1]

#Boucle Principale

#init

norm_res = 1
norm_err = 1
iteration = 9

#Fonctions de calcul de a1max et a2max

def c_a1max(crho):
    eps = 1E-6
    L = [[abs(a1_phys(k,1,j)) for k in crho] for j in range(len(y))]
    return max(max(max(L)), eps)

def c_a2max(crho):
    eps = 1E-6
    L = [[abs(a2_phys(k,i,1)) for k in crho] for i in range(len(x))]
    return max(max(max(L)), eps)

#On décremente i et j de 1
#Fonction calcul flux num

def c_flux_num(crho):
    if ischema == 1:
        q1 = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        cf1num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        for i in range(1,imax-1+1):
            i = i-1
            for j in range(2,jmax-1+1):
                j = j-1
                q1[i][j] = q1_num(crho[i][j], crho[i+1][j],i,j)
                cf1num[i][j] = 0.5*(f1_phys(crho[i][j],i,j)+f1_phys(crho[i+1][j],i,j))-0.5*q1[i][j]*(crho[i+1][j]-crho[i][j])

        q2 = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        cf2num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        for i in range(2,imax-1+1):
            i = i-1
            for j in range(1,jmax-1+1):
                j = j-1
                q2[i][j] = q2_num(crho[i][j],crho[i][j+1],i,j)
                cf2num[i][j] = 0.5*(f2_phys(crho[i][j],i,j)+f2_phys(crho[i][j+1],i,j))-0.5*q2[i][j]*(crho[i][j+1]-crho[i][j])

        return cf1num,cf2num

    elif ischema == 2:
        q1 = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        cf1num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        for i in range(1,imax-1+1):
            i = i-1
            for j in range(2,jmax-1+1):
                j = j-1
                q1[i][j] = q1_num(crho[i][j], crho[i+1][j],i,j)
                res1 = (f1_phys(crho[i+1][j],i,j)-f1_phys(crho[i][j],i,j))/dx + 0.25*(f2_phys(crho[i][j+1],i,j)+f2_phys(crho[i+1][j+1],i,j)-f2_phys(crho[i][j-1],i,j)-f2_phys(crho[i+1][j-1],i,j))/dy
                cf1num[i][j] = 0.5*(f1_phys(crho[i][j],i,j)+f1_phys(crho[i+1][j],i,j))-0.5*q1[i][j]*res1
        q2 = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        cf2num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        for i in range(2,imax-1+1):
            i = i-1
            for j in range(1,jmax-1+1):
                j = j-1
                q2[i][j] = q2_num(crho[i][j],crho[i][j+1],i,j)
                res2 = (f2_phys(crho[i][j+1],i,j)-f2_phys(crho[i][j],i,j))/dy + 0.25*(f1_phys(crho[i+1][j],i,j)+f1_phys(crho[i+1][j+1],i,j)-f1_phys(crho[i-1][j],i,j)-f1_phys(crho[i-1][j+1],i,j))/dx
                cf2num[i][j] = 0.5*(f2_phys(crho[i][j],i,j)+f2_phys(crho[i][j+1],i,j))-0.5*q2[i][j]*res2

        return cf1num,cf2num
    
    elif ischema == 3:
        cf1num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        for i in range(1,imax-1+1):
            i = i-1
            for j in range(2,jmax-1+1):
                j = j-1
                res1 = Phi[i][j] * np.sign(a[j]) * dx * ((f1_phys(crho[i+1][j],i,j)-f1_phys(crho[i][j],i,j))/dx + 0.25*(f2_phys(crho[i][j+1],i,j)+f2_phys(crho[i+1][j+1],i,j)-f2_phys(crho[i][j-1],i,j)-f2_phys(crho[i+1][j-1],i,j))/dy)
                cf1num[i][j] = 0.25*(f1_phys(crho[i][j],i,j)+f1_phys(crho[i+1][j],i,j)) - 0.25 * res1
        
        cf2num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        for i in range(2,imax-1+1):
            i = i-1
            for j in range(1,jmax-1+1):
                j = j-1
                res2 = Psi[i][j] * np.sign(b[i]) * dy * ((f2_phys(crho[i][j+1],i,j) - f2_phys(crho[i][j],i,j))/dy + 0.25*(f1_phys(crho[i+1][j],i,j)+f1_phys(crho[i+1][j+1],i,j)-f1_phys(crho[i-1][j],i,j)-f1_phys(crho[i-1][j+1],i,j))/dx)
                cf2num[i][j] = 0.25*(f2_phys(crho[i][j],i,j)+f2_phys(crho[i][j+1],i,j)) - 0.25 * res2 

        return cf1num,cf2num
    
    elif ischema == 4:
        cf1num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        cf2num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
        for i in range(1,imax-1):
            i = i - 1
            for j in range(1,jmax-1):
                j = j-1
                flux = predire_flux(i, j)
                cf1num[i][j]= flux[0] + flux[3]
                cf2num[i][j] = flux[1] + flux[2]     
                # print(cf1num[i][j])
                # print(cf2num[i][j])
        #utiliser F1-F4
        return cf1num, cf2num

# Mesure temps moyen prediction

# temps = []
# for i in range (10):
#     start = time.time()
#     predire_flux(3,3)
#     temps.append(time.time() - start)

# print(1/10 * sum(temps))

#boucle

for i in range(1,imax+1):
    rho[i-1][0]=rho_inlet(x[i-1])
for j in range(1,jmax+1):
    rho[0][j-1]=rho_inlet(x[0])

for i in range(1,imax+1):
    rho[i-1][jmax-1]=rho[i-1][jmax-1-1]
for j in range(1,jmax+1):
    rho[imax-1][j-1]=rho[imax-1-1][j-1]

while iteration < 5000:

    #pas de temps
    a1max = c_a1max(rho)
    a2max = c_a2max(rho)

    dt = cfl*min(dx/a1max,dy/a2max)

    #flux num

    f1num,f2num = c_flux_num(rho)

    rhoold = [[0 for j in range(jmax)] for i in range(imax)]

    for i in range(imax):
        for j in range(jmax):
            rhoold[i][j] = rho[i][j]

    for i in range(2,imax-1+1):
        i = i-1
        for j in range(2,jmax-1+1):
            j = j-1
            rho[i][j] = rho[i][j] - (dt/dx)*(f1num[i][j]-f1num[i-1][j])-(dt/dy)*(f2num[i][j]-f2num[i][j-1])

    #Conditions aux limites

    #    conditions d'entrée

    for i in range(1,imax+1):
        rho[i-1][0]=rho_inlet(x[i-1])
    for j in range(1,jmax+1):
        rho[0][j-1]=rho_inlet(x[0])

    #    conditions de sortie

    for i in range(1,imax+1):
        rho[i-1][jmax-1]=rho[i-1][jmax-1-1]
    for j in range(1,jmax+1):
        rho[imax-1][j-1]=rho[imax-1-1][j-1]

    iteration += 1

    #calcul norm

    norm_res = 0
    for i in range(1,imax+1):
        i = i-1
        for j in range(1,jmax+1):
            j = j-1
            norm_res += (rho[i][j]-rhoold[i][j])**2
    norm_res = np.sqrt(norm_res/(imax*jmax))

    norm_err = 0
    for i in range(1,imax+1):
        i = i-1
        for j in range(1,jmax+1):
            j = j-1
            norm_err += (rho[i][j]-rhoex[i][j])**2
    norm_err = np.sqrt(norm_err/(imax*jmax))

    print(iteration)
    Y, X = np.meshgrid(y,x)
    
        # fig = plt.figure()
        # ax = fig.add_subplot(111,projection='3d')
        # ax.plot_surface(X,Y,np.array(rho), cmap=cm.coolwarm)
        #ax.plot_surface(X,Y,np.array(rhoex))
        # plt.show()
        
    with open(f"rho_n={iteration}.txt", 'w') as f:
        for line in rho:
            f.write(str(line) + '\n')
    

Y, X = np.meshgrid(y,x)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,np.array(rho), cmap=cm.coolwarm)
#ax.plot_surface(X,Y,np.array(rhoex))
plt.show()
#fig.savefig('demo.png', bbox_inches='tight')



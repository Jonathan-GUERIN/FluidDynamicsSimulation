
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


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
a = 0
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



#Fonctions

def f1_phys(rho):
    return a*rho

def f2_phys(rho):
    return b*rho

def a1_phys(rho):
    return a

def a2_phys(rho):
    return b

def q1_num(rhol,rhor):
    if ischema == 1:
        return abs(a)
    elif ischema ==2:
        return a*dt

def q2_num(rhol,rhor):
    if ischema == 1:
        return abs(b)
    elif ischema == 2:
        return b*dt

def rho_inlet(x):
    #if iinlet == 0:
    #   if x > 0.2 and x < 0.5:
    #        return 1
    #    else:
    #        return 0
    return np.exp(-50*((x-0.5)**2))




def rho_ex(x,y):
    return rho_inlet(x-(a/b)*y)

list_norm_err = []

for imax in range(10,60,10) :
    # Nombre de cellules
    jmax = imax
    
    #dx et dy
    
    dx = l_dom_x / (imax-1)
    dy = l_dom_y / (jmax-1)
    
    #Affichage
    #print('Nb cellules suivant x : ' +str(imax))
    #print('Nb cellules suivant y : ' + str(jmax))
    #print('Abscisse minimale : ' + str(xmin))
    #print('Ordonnée minimale : ' +str(ymin))
    #print('Abscisse maximale : ' + str(xmin+l_dom_x,))
    #print('Ordonnée maximale : ' +str(ymin+l_dom_y))
    #print("Pas d'espace suivant x : "+str(dx))
    #print("Pas d'espace suivant y : "+str(dy))
    #print('Nombre CFL : '+str(cfl))
    #print('Fréquence des sorties : '+str(freq_ech))
    
    #Création de x et y et de rho et rhoex
    
    x = [xmin + (i-1)*dx for i in range(1,imax+1)]
    y = [ymin + (j-1)*dy for j in range(1,jmax+1)]
    
    rho = [[0 for j in range(1,jmax+1)] for i in range(1,imax+1)]
    for i in range(1,imax+1):
        for j in range(1,jmax+1):
            rho[i-1][j-1]=0
    rhoex = [[0 for j in range(1,jmax+1)] for i in range(1,imax+1)]
    for i in range(1,imax):
        for j in range(1,jmax):
            rhoex[i-1][j-1]=rho_ex(x[i-1],y[j-1])
            
    
    rhoold = [[0 for j in range(jmax)] for i in range(imax)]
    
    print('La simulation démarre. imax = ' + str(imax))
    
    
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
    iteration = 0
    
    
    #Fonctions de calcul de a1max et a2max
    
    def c_a1max(crho):
        eps = 1E-6
        L = [abs(a1_phys(k)) for k in crho]
        return max(max(L), eps)
    
    def c_a2max(crho):
        eps = 1E-6
        L = [abs(a2_phys(k)) for k in crho]
        return max(max(L), eps)
    
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
                    q1[i][j] = q1_num(crho[i][j], crho[i+1][j])
                    cf1num[i][j] = 0.5*(f1_phys(crho[i][j])+f1_phys(crho[i+1][j]))-0.5*q1[i][j]*(crho[i+1][j]-crho[i][j])
    
            q2 = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
            cf2num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
            for i in range(2,imax-1+1):
                i = i-1
                for j in range(1,jmax-1+1):
                    j = j-1
                    q2[i][j] = q2_num(crho[i][j],crho[i][j+1])
                    cf2num[i][j] = 0.5*(f2_phys(crho[i][j])+f2_phys(crho[i][j+1]))-0.5*q2[i][j]*(crho[i][j+1]-crho[i][j])
    
            return cf1num,cf2num
    
        elif ischema == 2:
            q1 = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
            cf1num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
            for i in range(1,imax-1+1):
                i = i-1
                for j in range(2,jmax-1+1):
                    j = j-1
                    q1[i][j] = q1_num(crho[i][j], crho[i+1][j])
                    res1 = (f1_phys(crho[i+1][j])-f1_phys(crho[i][j]))/dx + 0.25*(f2_phys(crho[i][j+1]+f2_phys(crho[i+1][j+1])-f2_phys(crho[i][j-1])-f2_phys(crho[i+1][j-1])))/dy
                    cf1num[i][j] = 0.5*(f1_phys(crho[i][j])+f1_phys(crho[i+1][j]))-0.5*q1[i][j]*res1
            q2 = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
            cf2num = [[0 for j in range(1,jmax-1+1)] for i in range(1,imax-1+1)]
            for i in range(2,imax-1+1):
                i = i-1
                for j in range(1,jmax-1+1):
                    j = j-1
                    q2[i][j] = q2_num(crho[i][j],crho[i][j+1])
                    res2 = (f2_phys(crho[i][j+1])-f2_phys(crho[i][j]))/dy + 0.25*(f1_phys(crho[i+1][j])+f1_phys(crho[i+1][j+1])-f1_phys(crho[i-1][j])-f1_phys(crho[i-1][j+1]))/dx
                    cf2num[i][j] = 0.5*(f2_phys(crho[i][j])+f2_phys(crho[i][j+1]))-0.5*q2[i][j]*res2
    
            return cf1num,cf2num
    
    
    #boucle
    
    while norm_res > 1E-16:
    
        #pas de temps
        a1max = c_a1max(rho)
        a2max = c_a2max(rho)
    
        dt = cfl*min(dx/a1max,dy/a2max)
    
        #flux num
    
        f1num,f2num = c_flux_num(rho)
    
    
    
        for i in range(imax):
            for j in range(jmax):
                rhoold[i][j] = rho[i][j]
    
        for i in range(2,imax-1+1):
            i = i-1
            for j in range(2,jmax-1+1):
                j = j-1
                rho[i][j] = rho[i][j] - (dt/dx)*(f1num[i][j]-f1num[i-1][j])-(dt/dy)*(f2num[i][j]-f2num[i][j-1])
            #print("aa " + str(i) + "kkk " + str(imax))
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
        for i in range(2,imax):
            for j in range(2,jmax):
                norm_err += (rho[i][j]-rhoex[i][j])**2
        norm_err = np.sqrt(norm_err/(imax*jmax))
        
    list_norm_err.append(np.log10(norm_err))
    
plt.plot(list_norm_err)
plt.title('erreur Roe')

#X , Y = np.meshgrid(y,x)
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.contour3D(X,Y,rho, 50, cmap=cm.coolwarm)
#ax.contour3D(X,Y,rhoex, 50)
#plt.show()
#fig.savefig('demo.png', bbox_inches='tight')


#plt.plot(list(range(len(rhoex))), rhoex)




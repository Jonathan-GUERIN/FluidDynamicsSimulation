import numpy as np
from param import *
import matplotlib.pyplot as plt


def phi(x):
	if k_NL == 0:
		return 1
	else:
		return ((Ti-Tl)*x/T0+Tl/T0)**r

def solex(tau,eta,nmax):
	sumtheta = 0
	for n in range(nmax):
		nn = 2*n+1
		sumtheta = sumtheta + (1/nn)*np.exp(-tau*(nn*np.pi/2)**2)*np.sin(nn*np.pi*eta/2)
	return 4/np.pi*sumtheta

# maillage

dxi = 2. / (imax-1)
xi = []
for i in range(1,imax+1):
	xi.append((i-1)*dxi)

Fo = 0.0

# trace distribution initiale
theta = list(range(imax))
for i in range(1,imax):
	theta[i-1]=1
theta[0] = 0
theta[imax-1] = 0

plt.plot(xi,theta)
plt.show()

# boucle principale

for j in range(1,itermax+1):

	phimax = 0
	for i in range(2,imax-1):
		phimax = max(phimax,phi(theta[i-1]))

	dFo = acc*0.5*dxi*dxi/phimax
	if j == 1:
		dFoinit = dFo

	Fo += dFo

	flux_num = list(range(imax))
	for i in range(2,imax+1):
		flux_num[i-1] = phi(0.5*(theta[i-2]+theta[i-1]))*((theta[i-1] - theta[i-2]))/dxi

	for i in range(2,imax):
		theta[i-1] = theta[i-1] + (dFo/dxi) * (flux_num[i] - flux_num[i-1])

	thetaex = list(range(imax))
	for i in range(1,imax+1):
		thetaex[i-1] = solex(Fo,xi[i-1],nmode)

	if j%freq_dist == 0:
		plt.subplot(1,3,1)
		plt.plot(xi,theta,"ro",)
		plt.plot(xi,thetaex,"#3366ff")
		plt.subplot(1,3,2)
		plt.plot([j],[dFo/dFoinit],"bo")
		plt.subplot(1,3,3)
		plt.plot([L*L*Fo/alpha],[Tl+(Ti-Tl)*theta[int((imax+1)/2)]],"ro")
		plt.plot([L*L*Fo/alpha],[Tl+(Ti-Tl)*thetaex[int((imax+1)/2)]],"go")


plt.show()


Tex = [Tl+(Ti-Tl)*th for th in thetaex]
T = [Tl+(Ti-Tl)*th for th in theta]
x = [L*(k-1) for k in xi]

plt.plot(x,T,"ro")
plt.plot(x,Tex,"#3366ff")
plt.show()
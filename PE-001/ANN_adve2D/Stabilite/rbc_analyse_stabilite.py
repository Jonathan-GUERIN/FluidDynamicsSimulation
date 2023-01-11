# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:42:05 2021

@author: PE-01
"""

import numpy as np
import matplotlib.pyplot as plt

dx = 0.1
dy = 0.1

def relation_stabilite(chi1 , chi2 , a , b , dx , dy,  phi, psi):
    Reg = 1 + (phi * a)/dx * (np.cos(chi1) - 1) + (psi * b)/dy * (np.cos(chi2) - 1) - 1/2 * ((phi*b)/dx + (psi*a)/dy) * np.sin(chi1) * np.sin (chi2)
    Img = (a * np.sin(chi1) + b * np.sin(chi2))
    g = Reg + 1j * Img 
    return g

stable_pairs = []

for a_point in np.linspace(-2,2,200):
    for b_point in np.linspace(-2,2,200):
        if b_point != 0:
            Phi = dx * min(1, np.abs(a_point/b_point)) * np.sign(a_point)
        else:
            Phi = 1
        if a_point != 0:
            Psi = dy * min(1, np.abs(b_point/a_point)) * np.sign(b_point)
        else:
            Psi = 1
        valeur = True
        for chi1 in np.linspace(0,2*np.pi, 100):
            for chi2 in np.linspace(0,2*np.pi, 100):
                g = relation_stabilite(chi1,chi2,a_point,b_point,dx,dy,Phi,Psi)
                valeur = valeur and (np.abs(g) <= 1)
                if not valeur:
                    break
            if not valeur:
                break
        if valeur:
            stable_pairs.append((a_point,b_point))

a_stable = [paire[0] for paire in stable_pairs]
b_stable = [paire[1] for paire in stable_pairs]

plt.plot(a_stable,b_stable,'*')
                
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:24:26 2019

@author: jean-baptiste
"""

# On va estimer a priori mu selon Q
#c est de dimension 24, il représente l'espérance du cout, la coordonnée i est le cout au mois i.
#On rentre à la main des valeurs de c pour coller au modèle.
#On a comme paramètre une loi normale N(c,S)
#Au début, on va considérer S comme tridiagonale (une dépendance d'un mois à l'autre)
# S est R^24x24


#On a de l'autre coté des observations sur k mois, des échantillons x_1, ... x_k.
#On veut construire mu qui équilibre entre "coller aux échantillons" et "coller à la prédiction".

#On veut avoir une prédiction selon N(mu,R)

#Mathématiquement, cela revient à jouer sur les paramètres R et S.
# On doit trouver mu qui minimise ps(Xk-Uk,Rk^-1 (Xk-Uk)) + ps(mu-c,S^-1 (mu - c))

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der



def tridiag(n,s1,s2):
    return np.array([[s1*(abs(i-j)==0)+s2*(abs(i-j)==1) for j in range(n)] for i in range(n)])

def ps(n,a,b):
    assert(len(a)>=n and len(b)>=n)
    res=0
    for k in range(n):
        res+=a[k]*b[k]
    return res

def to_minimize_vector(mu_v):
    res=[]
    for mu in mu_v:
        res.append( ps(k,X-mu[:k],np.dot(np.linalg.inv(R),X-mu[:k])) + ps(dim,mu-C,np.dot(np.linalg.inv(S),mu - C)))
    return np.array(res)


def predict(tendancy, sigma_tendancy_1, sigma_tendancy_2, raw_data, sigma_data_1, sigma_data_2):
    args= {0 : tendancy, 1 : sigma_tendancy_1, 2 : sigma_tendancy_2, 
            3 : raw_data, 4 : sigma_data_1, 5 : sigma_data_2}
    C=args[0]
    S=tridiag(dim,args[1], args[2])
    X=args[3]
    k=len(X)
    R=tridiag(k,args[4],args[5])
    
    def to_minimize(mu):
        return ps(k,X-mu[:k],np.dot(np.linalg.inv(R),X-mu[:k])) + ps(dim,mu-C,np.dot(np.linalg.inv(S),mu - C))

    res=minimize(to_minimize,np.array([50 for i in range(dim)]))
    print(res.x)
    plt.plot(range(dim),C)
    plt.plot(range(k),X)
    plt.plot(range(24),res.x)


dim=24

C=[-(x-(dim//2))**2+(dim//2)**2 for x in range(dim)]

s1=4
s2=2

X=np.array([50,10,90,80,10,0])


#predict(C,s1,s2,X,s1,s2)
predict(C,10,5,X,3,1)


#Alors ici on va essayer de représenter minimize, donc estimer une dimension 24 en une dimension 1.

# a chaque i dans abscisses on va représenter un vecteur de dimension 24 qui est a peu près représentatif.
# on commence par le vecteur [i,...,i]

#abscisses_1=np.array([i for i in range(1000)])

#abscisses_dim_1=np.array([[i-200 for j in range(dim)] for i in abscisses_1])
#abscisses_dim_2=np.array([[j-dim//2 + i - 500 for j in range(dim)] for i in abscisses_1])

#plt.plot(abscisses_1,to_minimize_vector(abscisses_dim_1))
#plt.plot(abscisses_1,to_minimize_vector(abscisses_dim_2))


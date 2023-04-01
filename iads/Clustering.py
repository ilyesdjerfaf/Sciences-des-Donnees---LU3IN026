# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import math
import matplotlib.pyplot as plt
# ------------------------ 

def normalisation(data):
    noms = np.array(data.columns)
    desc = np.array(data[noms])
    mini = np.min(desc, axis=0)
    maxi = np.max(desc, axis=0)
    desc = (desc - mini) / (maxi - mini)
    return pd.DataFrame(desc , columns = noms)

def dist_euclidienne(X , Y):
    return sum((X - Y )**2)**0.5

def centroide(data):
    return data.mean(axis=0)

def dist_centroides(data_1 , data_2):
    c_1 = centroide(data_1)
    c_2 = centroide(data_2)
    return dist_euclidienne(c_1 , c_2)

def initialise_CHA(data):
    return {i: [i] for i in range(len(data))}

    

def complete_linkage(data_1 , data_2):
    max = -math.inf
    for i in range(len(data_1)):
        for j in range(i, len(data_2)):
            dis = dist_euclidienne(data_1.iloc[i] , data_2.iloc[j])
            if max < dis :
                max =dis
    return max

def simple_linkage(data_1 , data_2):
    min = math.inf
    for i in range(len(data_1)):
        for j in range(i, len(data_2)):
            dis = dist_euclidienne(data_1.iloc[i] , data_2.iloc[j])
            if min > dis :
                min =dis
    return min

def average_linkage(data_1 , data_2):
    moy = 0
    for i in range(len(data_1)):
        for j in range( len(data_2)):
            moy += dist_euclidienne(data_1.iloc[i] , data_2.iloc[j])
    return moy //(len(data_1) + len(data_2))

def fusionne(data , P0 , verbose = False , linkage='centroid'):
    P1 = P0.copy()
    distance = math.inf
    keys_P0 = list(P0.keys())
    for i in range(len(P0)):
        j = i+1 
        while(j<len(P0)):
            if linkage== 'centroid':
                d = dist_centroides(data.iloc[P0[keys_P0[i]]] , data.iloc[P0[keys_P0[j]]])
            elif linkage=='complete' :
                d = complete_linkage(data.iloc[P0[keys_P0[i]]] , data.iloc[P0[keys_P0[j]]])
            elif linkage == "simple":
                d = simple_linkage(data.iloc[P0[keys_P0[i]]] , data.iloc[P0[keys_P0[j]]])
            elif linkage == "average" :
                d =  average_linkage(data.iloc[P0[keys_P0[i]]] , data.iloc[P0[keys_P0[j]]])
            else :
                print(f"{linkage} n'est pas n'existe pas")
            if d < distance :
                distance = d
                cle_1 = keys_P0[i]
                cle_2 = keys_P0[j]
            j= j+1
    P1[keys_P0[i]+1] = P0[cle_1] + P0[cle_2]
    del P1[cle_1]
    del P1[cle_2]
    if verbose :
        print(f"Distance mininimale trouvée entre  [{cle_1}, {cle_2}]  =  {distance}")
    return P1 , cle_1, cle_2, distance

def CHA(DF,linkage='centroid', verbose=False,dendrogramme=False) :
    l = []
    P = initialise_CHA(DF)
    while (len(P)!=1):
        result = fusionne(DF , P , verbose , linkage)
        l.append([result[1],result[2] , result[3] ,len(P[result[2]]) + len(P[result[1]])  ])
        P = result[0]
    if dendrogramme : 
        plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)    
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)

        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(
            l, 
            leaf_font_size=24.,  # taille des caractères de l'axe des X
        )

        # Affichage du résultat obtenu:
        plt.show()
    return l

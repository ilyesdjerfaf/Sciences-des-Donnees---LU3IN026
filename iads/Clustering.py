# -*- coding: utf-8 -*-

"""
Package: iads
File: Clustering.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions de Clustering

# import externe
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import matplotlib.cm as cm

import math
# ------------------------ 

def normalisation(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())

def dist_euclidienne(X , Y):
    return np.linalg.norm(X - Y )
    

def centroide(dataframe):
    return np.mean(dataframe,axis=0)

def dist_centroides(X,Y):
    return dist_euclidienne(centroide(X),centroide(Y))

def initialise_CHA(data):
    return {i: [i] for i in range(len(data))}

def fusionne(dataframe, P0, verbose = False):
    distances = {}
    for i in P0:
        j = i+1
        for j in P0:
            if i < j:
                distances[(i,j)] = dist_centroides(dataframe.iloc[P0[i]],dataframe.iloc[P0[j]])
    # on cherche le couple (i,j) qui correspond à la plus petite distance:
    min_dist = min(distances.values())
    for couple in distances:
        if distances[couple] == min_dist:
            cle_1,cle_2 = couple
            break
    # on fusionne les clusters i et j de P0:
    P1 = P0.copy()
    P1[i+1] = P0[cle_1] + P0[cle_2]
    del P1[cle_1]
    del P1[cle_2]
    # on affiche le résultat si verbose = True
    if verbose:
        print(f"Distance mininimale trouvée entre  [{cle_1}, {cle_2}] : {min_dist}")
    return P1, cle_1, cle_2, min_dist


def CHA_centroid(data):
    l = []
    P = initialise_CHA(data)
    while (len(P) != 1):
        result = fusionne(data, P)
        l.append([result[1], result[2], result[3],
                 len(P[result[2]]) + len(P[result[1]])])
        P = result[0]
    return l


def CHA_centroid(data, verbose=False, dendrogramme=False):
    l = []
    P = initialise_CHA(data)
    while (len(P) != 1):
        result = fusionne(data, P, verbose)
        l.append([result[1], result[2], result[3],
                 len(P[result[2]]) + len(P[result[1]])])
        P = result[0]

    if dendrogramme:
        # Paramètre de la fenêtre d'affichage:
        plt.figure(figsize=(30, 15))  # taille : largeur x hauteur
        plt.title('Dendrogramme', fontsize=25)
        plt.xlabel("Indice d'exemple", fontsize=25)
        plt.ylabel('Distance', fontsize=25)
        # Construction du dendrogramme pour notre clustering :
        scipy.cluster.hierarchy.dendrogram(l, leaf_font_size=24.,)

        # Affichage du résultat obtenu:
        plt.show()
    return l


def clustering_hierarchique_complete(X, Y):
    return np.max(cdist(X, Y, 'euclidean'))


def clustering_hierarchique_simple(X, Y):
    return np.min(cdist(X, Y, 'euclidean'))


def clustering_hierarchique_average(X, Y):
    return np.mean(cdist(X, Y, 'euclidean'))


def fusionnes(dataframe, P0, verbose=False, linkage='complete'):
    distances = {}
    for i in P0:
        j = i+1
        for j in P0:
            if i < j:
                if linkage == 'complete':
                    distances[(i, j)] = clustering_hierarchique_complete(
                        dataframe.iloc[P0[i]], dataframe.iloc[P0[j]])
                elif linkage == 'simple':
                    distances[(i, j)] = clustering_hierarchique_simple(
                        dataframe.iloc[P0[i]], dataframe.iloc[P0[j]])
                elif linkage == 'average':
                    distances[(i, j)] = clustering_hierarchique_average(
                        dataframe.iloc[P0[i]], dataframe.iloc[P0[j]])
                else:
                    print('Mauvais paramètre pour linkage')
                    return None
    # on cherche le couple (i,j) qui correspond à la plus petite distance:
    min_dist = min(distances.values())
    for couple in distances:
        if distances[couple] == min_dist:
            cle_1, cle_2 = couple
            break
    # on fusionne les clusters i et j de P0:
    P1 = P0.copy()
    P1[i+1] = P0[cle_1] + P0[cle_2]
    del P1[cle_1]
    del P1[cle_2]
    # on affiche le résultat si verbose = True
    if verbose:
        print(
            f"Distance mininimale trouvée entre  [{cle_1}, {cle_2}] : {min_dist}")
    return P1, cle_1, cle_2, min_dist


def CHA(DF, linkage='centroid', verbose=False, dendrogramme=False):
    """  ##### donner une documentation à cette fonction
    """
    ############################ A COMPLETER
    if linkage == 'centroid':
        return CHA_centroid(DF, verbose, dendrogramme)

    else:
        l = []
        P = initialise_CHA(DF)
        while (len(P) != 1):
            result = fusionnes(DF, P, verbose, linkage)
            l.append([result[1], result[2], result[3],
                     len(P[result[2]]) + len(P[result[1]])])
            P = result[0]

        if dendrogramme:
            # Paramètre de la fenêtre d'affichage:
            plt.figure(figsize=(30, 15))  # taille : largeur x hauteur
            plt.title('Dendrogramme', fontsize=25)
            plt.xlabel("Indice d'exemple", fontsize=25)
            plt.ylabel('Distance', fontsize=25)
            # Construction du dendrogramme pour notre clustering :
            scipy.cluster.hierarchy.dendrogram(l, leaf_font_size=24.,)

            # Affichage du résultat obtenu:
            plt.show()
        return l
    
def inertie_cluster(Ens):
    return np.sum(dist_euclidienne(Ens , centroide(Ens))**2)

def init_kmeans(K,Ens):
    return np.array(Ens.loc[np.random.choice(Ens.index , K , replace = False)])

def dist_eucl(X , Y):
    return np.sum((X-Y)**2 , axis=1)**0.5

def plus_proche(Exe,Centres):
    return np.argmin(dist_eucl(Centres , np.array(Exe)))

def affecte_cluster(Base,Centres):
    mat_affect = {i:[] for i in range(len(Centres))}
    for i in range(len(Base)):
        mat_affect[plus_proche(Base.iloc[i] , Centres)].append(i)
    return mat_affect

def nouveaux_centroides(Base,U):
    new_centroid = []
    for i in U.keys() :
        new_centroid.append(centroide(Base.iloc[U[i]]))
    return np.array(new_centroid)

def inertie_globale(Base, U):
    inertie_glob = 0
    for i in U.keys():
        inertie_glob += inertie_cluster(Base.iloc[U[i]])
    return inertie_glob

def kmoyennes(K, Base, epsilon, iter_max):
    # initialisation des centroides
    Centroides = init_kmeans(K, Base)

    # initialisation du dictionnaire d'affectation
    DictAffect = affecte_cluster(Base, Centroides)

    # initialisation de l'inertie globale
    inertie1 = inertie_globale(Base, DictAffect)

    # affichage des informations
    #print(f'iteration {1} Inertie : {inertie1:1.4f} Difference: {(inertie1 - epsilon - 1):1.4f}')

    # initialisation du compteur d'itérations
    compteur = 2

    # boucle principale
    while compteur <= iter_max:

        # recalcul des nouveaux centroides
        Centroides = nouveaux_centroides(Base, DictAffect)

        # recalcul du dictionnaire d'affectation
        DictAffect = affecte_cluster(Base, Centroides)

        # recalcul de l'inertie globale
        inertie2 = inertie_globale(Base, DictAffect)

        # calcul de la différence d'inertie
        diff = inertie1 - inertie2

        # mise à jour de l'inertie globale
        inertie1 = inertie2

        # affichage des informations
        #print(f'iteration {compteur} Inertie : {inertie1:1.4f} Difference: {diff:1.4f}')

        # test de la condition d'arrêt
        if diff < epsilon:
            break
        
        # incrémentation du compteur
        compteur += 1
  
    return Centroides, DictAffect

def compacite_cluster(Ens):
    max_dist = -math.inf
    for i in range(0,len(Ens)):
        dist = np.max(dist_eucl(np.array(Ens.iloc[i]),np.array(Ens)))
        if (dist > max_dist):
            max_dist =dist
    return max_dist

def compacite_globale(Base, U):
    compacite_glob = 0
    for i in U.keys():
        compacite_glob += compacite_cluster(Base.iloc[U[i]])
    return compacite_glob

def semin(Base, U):
    centroid = [centroide(np.array(Base.iloc[U[i]])) for i in U.keys()]
    sep_min = math.inf
    for i in range(len(centroid)-1):
        sep = np.min(dist_eucl(centroid[i] , centroid[i+1 :] ))
        if sep < sep_min:
            sep_min = sep
    return sep_min

def index_Dunn(Base, U):
    return compacite_globale(Base, U) / semin(Base, U)

def index_Xie_Beni(Base, U):
    return inertie_globale(Base, U) / semin(Base, U)

def affiche_resultat(Base,Centres,Affect):
    couleurs = cm.tab20(np.linspace(0, 1, len(Centres)))    

    for i in range(len(Centres)):
        plt.scatter(Base.iloc[Affect[i]]['X1'], Base.iloc[Affect[i]]['X2'], color=couleurs[i])
    plt.scatter(Centres[:,0],Centres[:,1],color='r',marker='x')
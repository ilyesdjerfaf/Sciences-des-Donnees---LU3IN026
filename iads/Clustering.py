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

    raise NotImplementedError("Please Implement this method")
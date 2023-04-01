# -*- coding: utf-8 -*-

"""
Package: iads
File: utils.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""


# Fonctions utiles pour les TDTME de LU3IN026
# Version de départ : Février 2023

# import externe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random




# ------------------------ 


# genere_dataset_uniform:


def genere_dataset_uniform(p, n, binf=-1, bsup=1):
    """ int * int * float^2 -> tuple[ndarray, ndarray]
        Hyp: n est pair
        p: nombre de dimensions de la description
        n: nombre d'exemples de chaque classe
        les valeurs générées uniformément sont dans [binf,bsup]
    """
    t1 = np.random.uniform(binf, bsup, (2*n, p))
    t2 = np.asarray([-1 for i in range(0, n)] + [+1 for i in range(0, n)])
    return t1, t2


# genere_dataset_gaussian:
def genere_dataset_gaussian(positive_center, positive_sigma, negative_center, negative_sigma, nb_points):
    """ les valeurs générées suivent une loi normale
        rend un tuple (data_desc, data_labels)
    """
    exemple_classe_negative = np.random.multivariate_normal(
        negative_center, negative_sigma, nb_points)

    exemple_classe_positive = np.random.multivariate_normal(
        positive_center, positive_sigma, nb_points)

    t1 = np.vstack((exemple_classe_negative, exemple_classe_positive))
    t2 = np.asarray([-1 for i in range(0, nb_points)] +
                    [+1 for i in range(0, nb_points)])
    return t1, t2


# plot2DSet:
def plot2DSet(desc, labels):
    """ ndarray * ndarray -> affichage
        la fonction doit utiliser la couleur 'red' pour la classe -1 et 'blue' pour la +1
    """
    # Extraction des exemples de classe -1:
    data_negatifs = desc[labels == -1]
    # Extraction des exemples de classe +1:
    data_positifs = desc[labels == +1]

    # Affichage de l'ensemble des exemples :
    # 'o' rouge pour la classe -1
    plt.scatter(data_negatifs[:, 0],
                data_negatifs[:, 1], marker='o', color="red")
    plt.scatter(data_positifs[:, 0], data_positifs[:, 1],
                marker='x', color="blue")  # 'x' bleu pour la classe +1


# plot_frontiere:
def plot_frontiere(desc_set, label_set, classifier, step=30):
    """ desc_set * label_set * Classifier * int -> NoneType
        Remarque: le 4e argument est optionnel et donne la "résolution" du tracé: plus il est important
        et plus le tracé de la frontière sera précis.        
        Cette fonction affiche la frontière de décision associée au classifieur
    """
    mmax = desc_set.max(0)
    mmin = desc_set.min(0)
    x1grid, x2grid = np.meshgrid(np.linspace(
        mmin[0], mmax[0], step), np.linspace(mmin[1], mmax[1], step))
    grid = np.hstack((x1grid.reshape(x1grid.size, 1),
                     x2grid.reshape(x2grid.size, 1)))

    # calcul de la prediction pour chaque point de la grille
    res = np.array([classifier.predict(grid[i, :]) for i in range(len(grid))])
    res = res.reshape(x1grid.shape)
    # tracer des frontieres
    # colors[0] est la couleur des -1 et colors[1] est la couleur des +1
    plt.contourf(x1grid, x2grid, res, colors=[
                 "darksalmon", "skyblue"], levels=[-1000, 0, 1000])


def create_XOR(n, var):
    """ int * float -> tuple[ndarray, ndarray]
        Hyp: n et var sont positifs
        n: nombre de points voulus
        var: variance sur chaque dimension
    """
    cov = [[var, 0], [0, var]]
    mean = [[0, 0], [1, 0], [1, 1], [0, 1]]
    #on
    classe_negative_1 = np.random.multivariate_normal(mean[0], cov, n)
    classe_positive_1 = np.random.multivariate_normal(mean[1], cov, n)
    classe_negative_2 = np.random.multivariate_normal(mean[2], cov, n)
    classe_positive_2 = np.random.multivariate_normal(mean[3], cov, n)
    t1 = np.vstack((classe_negative_1, classe_negative_2,
                    classe_positive_1, classe_positive_2))
    t2 = np.asarray([-1 for i in range(0, 2 * n)] +
                    [+1 for i in range(0, 2 * n)])
    return t1, t2

# ------------------------ A COMPLETER


def genere_train_test(desc_set, label_set, n_pos, n_neg):
    """ permet de générer une base d'apprentissage et une base de test
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        n_pos: nombre d'exemples de label +1 à mettre dans la base d'apprentissage
        n_neg: nombre d'exemples de label -1 à mettre dans la base d'apprentissage
        Hypothèses: 
           - desc_set et label_set ont le même nombre de lignes)
           - n_pos et n_neg, ainsi que leur somme, sont inférieurs à n (le nombre d'exemples dans desc_set)
    """
    n = desc_set.shape[0]
    index_neg = [i for i in range(n) if label_set[i] == -1]
    index_pos = [i for i in range(n) if label_set[i] == 1]
    index_pos_train = random.sample(index_pos, n_pos)
    index_neg_train = random.sample(index_neg, n_neg)

    train_desc = np.concatenate(
        (desc_set[index_pos_train], desc_set[index_neg_train]))
    train_label = np.concatenate(
        (label_set[index_pos_train], label_set[index_neg_train]))

    index_pos_test = [i for i in index_pos if i not in index_pos_train]
    index_neg_test = [i for i in index_neg if i not in index_neg_train]

    test_desc = np.concatenate(
        (desc_set[index_pos_test], desc_set[index_neg_test]))
    test_label = np.concatenate(
        (label_set[index_pos_test], label_set[index_neg_test]))

    return (train_desc, train_label), (test_desc, test_label)

def crossval(X, Y, n_iterations, iteration):
    n = len(Y) // n_iterations
    Xtest = X[iteration*n : (iteration+1)*n ]
    Ytest = Y[iteration*n : (iteration+1)*n ]
    Xapp = np.concatenate( (X[:iteration*n],X[(iteration+1)*n:]))
    Yapp = np.concatenate( (Y[:iteration*n],Y[(iteration+1)*n:]))
    return Xapp, Yapp, Xtest, Ytest

def crossval_strat(X, Y, n_iterations, iteration):
    labels = np.unique(Y)
    index_pos = np.where(Y == labels[1])[0]
    index_neg = np.where(Y == labels[0])[0]

    index_pos_test = index_pos[iteration*(len(index_pos) // n_iterations): (
        iteration+1)*(len(index_pos) // n_iterations)]
    index_neg_test = index_neg[iteration*(len(index_neg) // n_iterations): (
        iteration+1)*(len(index_neg) // n_iterations)]

    Xtest = np.concatenate((X[index_neg_test], X[index_pos_test]))
    Ytest = np.concatenate((Y[index_neg_test], Y[index_pos_test]))

    index_app = np.setdiff1d(np.arange(len(Y)), np.concatenate([index_pos_test, index_neg_test]))   

    Xapp = X[index_app]
    Yapp = Y[index_app]
    return Xapp, Yapp, Xtest, Ytest



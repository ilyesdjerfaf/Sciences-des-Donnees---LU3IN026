# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 
#TODO: à compléter  plus tard
# ------------------------ 
# def crossval_strat(X, Y, n_iterations, iteration):
#     labels = np.unique(Y)
#     index_pos = np.where(Y == labels[1])[0]
#     index_neg = np.where(Y == labels[0])[0]

#     index_pos_test = index_pos[iteration*(len(index_pos) // n_iterations): (
#         iteration+1)*(len(index_pos) // n_iterations)]
#     index_neg_test = index_neg[iteration*(len(index_neg) // n_iterations): (
#         iteration+1)*(len(index_neg) // n_iterations)]

#     Xtest = np.concatenate((X[index_neg_test], X[index_pos_test]))
#     Ytest = np.concatenate((Y[index_neg_test], Y[index_pos_test]))

#     index_app = np.setdiff1d(np.arange(len(Y)), np.concatenate([index_pos_test, index_neg_test]))   

#     Xapp = X[index_app]
#     Yapp = Y[index_app]
#     return Xapp, Yapp, Xtest, Ytest


def crossval_strat(X, Y, n_iterations, iteration):
    '''ancienne fonction'''
    unique_Y = np.unique(Y)
    index_list = []
    for i in range(len(unique_Y)):
        indexs, = np.where(Y == unique_Y[i])
        index_list.append(indexs)

    index_test_list = []
    for i in range(len(index_list)):
        index_i = index_list[i]
        indexs_test = index_i[iteration*(len(index_i) // n_iterations): (iteration+1)*len(index_i)//n_iterations]
        index_test_list.append(indexs_test)

    X_Test = X[index_test_list[0]]
    for i in range(1, len(index_test_list)):
        X_Test = np.concatenate((X_Test, X[index_test_list[i]]))

    Y_Test = Y[index_test_list[0]]
    for i in range(1, len(index_test_list)):
        Y_Test = np.concatenate((Y_Test, Y[index_test_list[i]]))

    index_train = []
    for i in range(len(X)):
        is_in = False
        for index in index_test_list:
            if i in index:
                is_in = True

        if is_in == False:
            index_train.append(i)

    X_Train = X[index_train]
    Y_Train = Y[index_train]

    return X_Train, Y_Train, X_Test, Y_Test


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L) , np.std(L)
    raise NotImplementedError("Vous devez implémenter cette fonction !")

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    for i in range(nb_iter):
        Xapp, Yapp, Xtest, Ytest = crossval_strat(X, Y, nb_iter, i)
        newC = copy.deepcopy(C)
        newC.train(Xapp , Yapp)
        perf.append(newC.accuracy(Xtest, Ytest))
        print(i," : taille app.= ",Xapp.shape[0],",  taille test = ",Xtest.shape[0]," ,Accuracy:  ",perf[i])
    
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)

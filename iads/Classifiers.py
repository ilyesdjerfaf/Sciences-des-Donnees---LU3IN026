# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2023

# Import de packages externes
import numpy as np
import pandas as pd
import collections

# ---------------------------

# ------------------------ A COMPLETER :


class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """

    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        if input_dimension <= 0:
            raise ValueError(
                "La dimension de la description des exemples doit etre strictement positive")
        else:
            self.dimension = input_dimension
        # raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI :
        resultat = [self.predict(desc_set[i])
                    for i in range(0, len(label_set))]
        equal_count = sum([1 for i in range(0, len(label_set))
                          if label_set[i] == resultat[i]])
        return equal_count / len(label_set)
        # ............

        # ------------------------------


# ------------------------ A COMPLETER :

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !

    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.k = k
        # raise NotImplementedError("Please Implement this method")

    def score(self, x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        distance = np.argsort([np.linalg.norm(x - y) for y in self.desc])
        p = sum([1 for a in range(self.k) if self.label[distance[a]] == +1])/self.k
        return 2*(p - 0.5)

        # raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return (+1 if self.score(x) >= 0 else -1)
        raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.desc = desc_set
        self.label = label_set

        # raise NotImplementedError("Please Implement this method")


# ------------------------ A COMPLETER :
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """        
        self.v = np.random.uniform(-1,1,(1, input_dimension))
        self.w = self.v/np.linalg.norm(self.v)
        # raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        print("Pas d'apprentissage pour ce classifieur !")
        # raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(self.w, x)
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return (+1 if self.score(x) >= 0 else -1)
        raise NotImplementedError("Please Implement this method")
    

class ClassifierKNN_MC(Classifier):

    def __init__(self, input_dimension, k, c):
        self.k = k
        self.c = c
        # raise NotImplementedError("Please Implement this method")

    def score(self, x):
        distance = np.argsort([np.linalg.norm(x - y) for y in self.desc])
        knn = [self.label[a] for a in distance[:self.k]]
        return collections.Counter(knn).most_common(1)[0][0]
        raise NotImplementedError("Please Implement this method")

    def predict(self, x):
        return self.score(x)
        raise NotImplementedError("Please Implement this method")

    def train(self, desc_set, label_set):
        self.desc = desc_set
        self.label = label_set
        self.labels_unique = list(set(label_set))
        # raise NotImplementedError("Please Implement this method")

# ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON


class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """

    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        self.learning_rate = learning_rate
        if init:
            self.w = np.zeros(input_dimension)
        else:
            self.w = np.array([0.001*(2*np.random.rand()-1)
                              for i in range(input_dimension)])
        self.allw = [self.w.copy()]

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        n = len(desc_set)
        indexes = [i for i in range(n)]
        np.random.shuffle(indexes)
        for i in indexes:
            yi_predicted = self.predict(desc_set[i])
            if yi_predicted != label_set[i]:
                self.w = self.w + self.learning_rate*label_set[i] * desc_set[i]
                self.allw.append(self.w.copy())

    def train(self, desc_set, label_set, nb_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - nb_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        converge = False
        cpt = 0
        liste_difference = []
        while (not converge and cpt < nb_max):
            ancien_w = self.w.copy()
            self.train_step(desc_set, label_set)
            diff = abs(ancien_w - self.w)
            norm = np.linalg.norm(diff)
            liste_difference.append(norm)
            if norm <= seuil:
                converge = True
            cpt += 1

        return liste_difference

    def score(self, x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)

    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return +1 if self.score(x) >= 0 else -1
    
    def get_allw(self):
        return self.allw


class ClassifierPerceptronBiais(ClassifierPerceptron):
    """ Perceptron de Rosenblatt avec biais
        Variante du perceptron de base
    """

    def __init__(self, input_dimension, learning_rate=0.01, init=True):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate (par défaut 0.01): epsilon
                - init est le mode d'initialisation de w: 
                    - si True (par défaut): initialisation à 0 de w,
                    - si False : initialisation par tirage aléatoire de valeurs petites
        """
        # Appel du constructeur de la classe mère
        super().__init__(input_dimension, learning_rate, init)

    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        ### A COMPLETER !
        n = len(desc_set)
        indexes = [i for i in range(n)]
        np.random.shuffle(indexes)
        for i in indexes:
            f_xi = super().score(desc_set[i])
            if f_xi * label_set[i] < 1:
                self.w = self.w + self.learning_rate * \
                    (label_set[i] - f_xi)*desc_set[i]
                self.allw.append(self.w.copy())

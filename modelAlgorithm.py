# model-algorithm.py
# ------------------
# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to 20020001-UET (Github).
#
# Attribution Information: This experiment was developed at UET Artificial 
# Intelligence Laboratory (AI Lab). 

"""
model-algorithm.py include Polynomial Hypothesis model using Mini-batch 
Gradient Descent (based) algorithm to train. This file is divided into 
three sections:

    (1) Optional functions

    (2) Feature Scaling functions

    (3) Polynomial Hypothesis class model

To using this module, simply import model-algorithm in your python code.
Thank you for reading!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##########################################################################
#                           OPTIONAL FUNCTIONS                           #
##########################################################################

def extractDataFromCSV(file_name):
    """
    extractDataFromCSV returns a list contains three things: list of data 
    t, list of data y and the dataset include both of them. 

    The argument needed for this function is file_name (.csv).
    """
    data = pd.read_csv(file_name) 

    data_t = data['t'].tolist()
    data_y = data['y'].tolist()

    dataset = [data_t, data_y]

    return [data_t, data_y, dataset]

def plotDataset(ax, dataset, model):
    """
    plotDataSet plots the original data and the predicted ones.
    """
    t, y = dataset
    t, y, predict = model.predicting(dataset)
    ax.plot(t, y, "ro")
    ax.plot(t, predict, "bx")

def polynomialCalc(theta, x, degree):
    """
    This function is a polynomial calculator.
    """

    ans = np.float128(0)
    pow = np.float128(1)
    for i in range(degree):
       ans = ans + theta[i]*pow
       pow *= x
    return ans

def costFunction(t, y, theta, degree):
    cost = np.float128(0)
    for i in range(len(t)):
        cost += np.power(polynomialCalc(theta, t[i], degree) - y[i], 2, dtype=np.float128)
    return cost / len(t)

##########################################################################
#                        FEATURE SCALING FUNCTIONS                       #
##########################################################################

class FeatureScaling:
    def __init__(self, data):
        self.data = data.copy()
        self.value = 0
        self.s = 1

    def scaling(self, method):
        if (method != 0):
            if (method == 1):
                self.value = min(self.data)
            elif (method == 2):
                self.value = sum(self.data) / len(self.data)
            self.s = max(self.data) - min(self.data)

            self.data = preProcessData(self.data, self.value, self.s)

def min_maxNormalization(data):
    tmp = data.copy()

    s = max(tmp) - min(tmp)
    minValue = min(tmp)

    for i in range(len(tmp)):
        tmp[i] = (tmp[i] - minValue) / s

    return [tmp, minValue, s]

def meanNormalization(data):
    tmp = data.copy()

    s = max(tmp) - min(tmp)
    mu = sum(tmp) / len(tmp)

    for i in range(len(tmp)):
        tmp[i] = (tmp[i] - mu) / s

    return [tmp, mu, s]

def preProcessData(data, value, s):
    tmp = data.copy()

    for i in range(len(tmp)):
        tmp[i] = (tmp[i] - value) / s

    return tmp

##########################################################################
#                   POLYNOMIAL HYPOTHESIS CLASS MODEL                    #
##########################################################################

class PolynomialHypothesis:
    """
    A PolynomialHypothesis model is a model use polynomial regression 
    algorithm to predict value from training set. 

    This model is using Mini-batch Gradient Descent to train. (It become 
    SGD when batch_size parameter is equal to 1.

    """

    def __init__(self, training_set, alpha, batch_size, iteration, degree, feature_scaling_method):
        """
        Constructor of Polynomial Hypothesis, there are 6 arguments:

            (1) training_set

            (2) alpha 

            (3) batch_size

            (4) iteration

            (5) degree
            
            (7) feature_scaling_method

        """
        # Training set initialize
        self.training_set = training_set.copy()

        # Super parameters initialize
        self.alpha = alpha
        self.batch_size = batch_size
        self.iteration = iteration

        # Online parameters initialize
        self.degree = degree

        # Theta initialize
        self.theta = np.zeros((degree,), dtype=np.float128)

        # Feature Scaling initialize
        self.feature_scaling_method = feature_scaling_method
        t, y = self.training_set
        self.t = FeatureScaling(t)
        self.y = FeatureScaling(y)

        self.t.scaling(self.feature_scaling_method)
        self.y.scaling(self.feature_scaling_method)

        self.training_set = [self.t.data, self.y.data]

    def training(self):
        """
        This function calculates theta parameters and find the best
        solution in order to minimize the cost function.
        """

        # Randomly shuffle training set
        np.random.shuffle(self.training_set)

        # Initialize data
        t, y = self.training_set
        m = len(t)

        cost = costFunction(t, y, self.theta, self.degree)

        # Mini-batch GD based algorithm:
        for iterator in range(self.iteration):

            for i in range(m//self.batch_size):
                # Backup a new theta
                tmp_theta = self.theta.copy()
                grad_theta = np.zeros((self.degree,), dtype=np.float128)

                for j in range(self.degree):
                    for k in range(self.batch_size):
                        grad_theta[j] += self.alpha*(polynomialCalc(self.theta, t[i+k], self.degree) - y[i+k])*np.power(t[i+k], j, dtype=np.float128)
                    grad_theta[j] /= self.batch_size

                for j in range(self.degree):
                    tmp_theta[j] -= grad_theta[j]

                # Save the new theta
                self.theta = tmp_theta.copy()

            # Debug Gradient Descent
            current_cost = costFunction(t, y, self.theta, self.degree)
            if (current_cost >= cost):
                return
            cost = current_cost

    def predicting(self, testing):
        """
        This function return predict result form the testing data.
        """
        # Initialize data
        t, y = testing.copy()
        m = len(t)
        predict = np.zeros((m,))

        # Feature Scaling
        t = preProcessData(t, self.t.value, self.t.s)
        y = preProcessData(y, self.y.value, self.y.s)

        # Predicting the result
        for i in range(m):
            predict[i] = polynomialCalc(self.theta, t[i], self.degree)

        return [t, y, predict]

    def rSquared(self, testing):
        """
        This function calculate the r squared (or r2 score).
        """

        # Initialize data
        t, y = testing.copy()
        m = len(t)
        predict = np.zeros((m,))

        # Feature Scaling
        t = preProcessData(t, self.t.value, self.t.s)
        y = preProcessData(y, self.y.value, self.y.s)

        # Predicting the result
        for i in range(m):
            predict[i] = polynomialCalc(self.theta, t[i], self.degree)

        y_mean = sum(y) / len(y)
        ssres = np.float128(0)
        sstot = np.float128(0)
        for i in range(m):
            ssres += np.power(y[i] - predict[i], 2, dtype=np.float128)
            sstot += np.power(y[i] - y_mean, 2, dtype=np.float128)

        return 1 - ssres / sstot

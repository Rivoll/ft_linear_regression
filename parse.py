#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:43:53 2019

@author: pkeita
"""
import numpy as np
import matplotlib.pyplot as plt


def Parse(file):
    data = np.genfromtxt (file, delimiter=",")
    X = data[:, 0]
    y = data[:, 1]
    X = np.matrix(X[1:]).transpose()
    y = np.matrix(y[1:]).transpose()
    return(X, y)
        

def normalequation(X, y):
    XX = np.dot(X.transpose(), X)
    XX = XX.astype(int)
    XXi = np.linalg.inv(XX)
    theta = np.dot(np.dot(XXi, X.transpose()), y)
    return (theta)


def costfunction(X, y, theta):
    H = np.dot(X, theta)
    H = H - y
    H = np.square(H)
    cost = H.sum() / (2 * H.shape[0])    
    return (cost)
    

def gradient(X, y, theta):
    H = np.dot(X, theta)
    HmY = H - y
    grad = np.dot(X.transpose(), HmY) / X.shape[0]   
    return (grad)


def update_theta(theta, grad, alpha):
    grad = grad * alpha
    theta = theta - grad
    return (theta)

    
def standardizer(Vector):
    return (Vector - np.mean(Vector)) / (np.std(Vector))

def destandardizer(Vector, mean, std):
    return Vector * std + mean

    
def gradient_descent(X, y, theta, alpha):
    cost = costfunction(X, y, theta)
    while True:
        grad = gradient(X, y, theta)
        theta = update_theta(theta, grad, alpha)
        new_cost = costfunction(X, y, theta)
        diff = cost - new_cost
        cost = new_cost
        if diff < 0.0001:
            break
    return theta


def find_theta(X, y):
    a = (y[1] - y[0]) / (X[1] - X[0])
    b = y[0] - X[0] * a
    theta = [float(b), float(a)]
    return(theta)
    

def write_np_array(file_name, to_write):
    np.savetxt(file_name, to_write);


def main():
    alpha = 1.01
    X, y = Parse("data.csv")
    Xnorm = standardizer(X)
    Xnorm = np.insert(Xnorm, 0, 1, axis=1)
    ynorm = standardizer(y)
    theta = np.matrix("0;0") # a changer, ca doit prendre la valeur du fichier theta
    theta = gradient_descent(Xnorm, ynorm, theta, alpha)
    ypred_norm = np.dot(Xnorm, theta)
    ypred = destandardizer(ypred_norm, np.mean(y), np.std(y))
    theta = find_theta(X, ypred)
    plt.scatter(X.tolist(), y.tolist(), c="red")
    plt.plot(X.tolist(), ypred)
    plt.ylabel("Price")
    plt.xlabel("Km")
    print(theta)
    print(type(theta[0]))
    write_np_array("theta.csv", theta)
    return (theta)


if __name__ == "__main__":
    main()

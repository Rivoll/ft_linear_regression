#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 05:54:23 2019

@author: pkeita
"""
import sys
import numpy as np


if __name__ == "__main__":
    mileage = sys.argv[1]
    theta = np.genfromtxt ('theta.csv', delimiter=",")
    try:
        print(theta[0] + theta[1] * float(mileage))
    except:
        print("Usage : python3.6 prediction Kilometers")
#!/usr/bin/env python3

import numpy as np
import csv
import matplotlib.pyplot as plt


def f(w, x):
    return 1.0/(1.0 + np.exp(-(np.matmul(x, np.transpose(w)))))


def error(we, X, Y):
    err = 0.0
    i = 0
    for x, y in zip(X, Y):
        fx = f(we, x)
        err += 0.5 * (fx - y) ** 2
        i += 1
    return err/i


X, Y = [], []
with open('./training_data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)  # Skip header line
    for row in readCSV:
        a = list(map(float, row))
        X.append(a[:-1])
        Y.append(a[-1])

Xtrain = np.asarray(X)
Ytrain = np.asarray(Y)

X, Y = [], []
with open('./validation_data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    next(readCSV)  # Skip header line
    for row in readCSV:
        a = list(map(float, row))
        X.append(a[:-1])
        Y.append(a[-1])

Xvalid = np.asarray(X)
Yvalid = np.asarray(Y)


i = 0

etrain = []
evalid = []
for i in range(10):
    w = np.load(f'./weights/weights_after_epoch_{i}.npy')
    etrain.append(error(w, Xtrain, Ytrain))
    evalid.append(error(w, Xvalid, Yvalid))
plt.plot(range(10), etrain, 'bo-', range(10), evalid, 'go-')
plt.grid(True)
plt.show()

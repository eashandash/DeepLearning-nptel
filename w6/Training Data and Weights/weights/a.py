import numpy as np
import csv

X = []

with open('../data.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	next(readCSV)  # Skip header line
	for row in readCSV:
		a = list(map(float, row))
		X.append(a)

X = np.asarray(X)

par1 = np.load('./autoencoder1.npy')
[W_e_1, b_1, W_d_1, c_1] = par1

par2 = np.load('./autoencoder2.npy')
[W_e_2, b_2, W_d_2, c_2] = par2


def f(w,b,x):
	return 1.0/(1.0 + np.exp(-(np.matmul(x, np.transpose(w)) + b)))


def error(we,b,wd,c):
	err = 0.0
	for x in X:
		h1 = f(we,b,x)
		o1 = f(wd,c,h1)
		err += np.sum((o1-x)**2)
	return err


e1 = error(W_e_1,b_1,W_d_1,c_1)
e2 = error(W_e_2, b_2, W_d_2, c_2)


print(e1<e2)





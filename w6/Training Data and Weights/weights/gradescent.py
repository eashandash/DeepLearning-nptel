import numpy as np
import csv

X = []
Y = []



with open('data.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
		X.append(float(row[0]))
		Y.append(float(row[1]))

def f(w,b,x):
	return 1.0/(1.0 + np.exp(-(w*x + b)))


def error(w,b):
	err = 0.0
	for x,y in zip(X,Y):
		fx = f(w,b,x)
		err+=0.5*(fx-y)**2
	return err


def grad_b(w,b,x,y):
	fx = f(w,b,x)
	return (fx-y)*fx*(1-fx)

def grad_w(w,b,x,y):
	fx = f(w,b,x)
	return (fx-y)*fx*(1-fx)*x

def do_gradient_descent():
	w,b,eta,max_pochs = 1.0,1.0,0.01,100
	for i in range(max_pochs):
		dw,db = 0.0,0.0
		for x,y in zip(X,Y):
			dw += grad_w(w,b,x,y)
			db += grad_b(w,b,x,y)

		w = w - eta*dw
		b = b - eta*db

		print(error(w,b))


do_gradient_descent()
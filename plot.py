import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv

X = []
Y = []
O = []

with open('foo.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	# next(readCSV)  # Skip header line
	for row in readCSV:
		X.append(float(row[0]))
		Y.append(float(row[1]))
		O.append(float(row[2]))


plt.scatter(X, Y, c=O)
plt.show()


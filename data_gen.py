import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

size = 50

X = np.random.rand(size)
Y = np.random.rand(size)
O = np.random.randint(low=0,high=2,size=size) # [low,high)

X = (X*2)-1
Y = (Y*2)-1

data = []

for x,y,o in zip(X,Y,O):

	data.append([x,y,int(o)])

plt.scatter(X, Y, c=O)
plt.show()

np.savetxt("foo.csv", data, delimiter=",")

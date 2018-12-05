import numpy as np

p = np.asarray([
	[-1,1,1],
	[1,1,1],
	[1,1,-1]
	])

d = np.asarray([
	[4,0,0],
	[0,-2,0],
	[0,0,5]
	])

pin = np.asarray([
	[-1,1,0],
	[1,0,1],
	[0,1,-1]
	])


k = np.matmul(p,d)

l = np.matmul(k,pin/2)

print(l)
# Script to convert .npy file to csv

import numpy as np

arr = np.load('autoencoder2.npy')

W_e_2 = arr[0]
b_2 = arr[1]
W_d_2 = arr[2]
c_2 = arr[3]

numpy.savetxt("W_e_2.csv", W_e_2, delimiter=",")
numpy.savetxt("b_2.csv",   b_2, delimiter=",")
numpy.savetxt("W_d_2.csv", W_d_2, delimiter=",")
numpy.savetxt("c_2.csv",   c_2, delimiter=",")

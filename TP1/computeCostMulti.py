import numpy as np


def computeCostMulti(X, y, theta):  
   m = y.size
   J = 0.
   for i in range(m):
      J += (X[i, :]@theta - y[i, 0])**2
   J = 1/(2*m)*J[0]

   return J

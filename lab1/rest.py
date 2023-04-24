import main
import numpy as np
import scipy
import math

l = np.linspace(0, 123.5, 12, dtype = float)
print(l)
a = np.arange(0, 123.5, 12, dtype=float)
print(a)

##################################################################

M = np.array([[3,1,-2,4],[0,1,1,5],[-2,1,1,6],[4,3,0,1]])
print(M[0][0])
print(M[2][2])
print(M[2][1])
w1 = M[:,2]
w2 = M[1,:]
print(w1)
print(w2)

##################################################################

v1 = np.array([[1],[3],[13]])
v2 = np.array([[8],[5],[-2]])

print(4*v1)
print(-v2+np.array([[2],[2],[2]]))

##################################################################

M1 = np.array([[1,-7,3],[-12,3,4],[5,13,-3]])

print(np.multiply(M1, 3))
print(np.multiply(M1, 3) + np.ones((3, 3)))
print(np.transpose(M1))
print(np.matmul(M1, v1))
print(np.matmul(np.transpose(v2), M1))

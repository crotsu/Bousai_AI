import numpy as np

A = np.array([[2,0,4],[1,1,-3]])
B = np.array([[5,7,3],[2,0,-1]])
C = 2*A+3*B
print("2A+3B=")
print(C)

print()

C = np.dot(A.T,B)
print("A^T x B=")
print(C)

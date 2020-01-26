import numpy as np

A = np.array([[5,4],[3,2]])
Ainv = np.linalg.inv(A)
print("Ainv=")
print(Ainv)

print()

print("Ainv x A=")
print(np.dot(Ainv,A))

print()

np.set_printoptions(precision=3, suppress=True) #有効桁数3桁で丸めて，指数表示を禁止する
print("Ainv x A=")
print(np.dot(Ainv,A))

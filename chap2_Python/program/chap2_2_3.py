import numpy as np

# 指数が読みにくいので...
np.set_printoptions(precision=3, suppress=True) #有効桁数3桁で丸めて，指数表示を禁止する

# 行列A
A = np.array([[2,3,-5],[1,-1,1],[3,-6,2]])

# ベクトルb
b = np.array([[3],[0],[-7]])

x = np.linalg.solve(A, b)
print(x)

print()

# 確認
print(np.dot(A,x))

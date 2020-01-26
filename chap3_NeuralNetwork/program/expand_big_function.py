#
# ニューラルネットワークのパラメータを全部展開して大きな関数であることを確認する
#
import numpy as np

# パラメータ
EPSILON = 4.0

# シグモイド関数
def sigmoid(x):
    return 1/(1+np.exp(-1*EPSILON*x))

# 入力（XORの入力部分）
dataX = np.array(
    [[0,0],
     [0,1],
     [1,0],
     [1,1]]
)

# 教師信号（XORの出力部分）
dataY = np.array(
    [[0],
     [1],
     [1],
     [0]]
)

# 重みと閾値(back_propagation.pyでnp.random.seed(3)で計算した値)
wab =  1.94
wac = -1.88
wbd = -1.54
wbe =  1.60
wcd = -1.21
wce =  1.29
tha =  0.88
thb = -0.92
thc =  0.58

for p in range(len(dataX)):

    ##########
    # 前向き計算
    ##########
    outa = sigmoid(wab * sigmoid(wbd * dataX[p][0] + wbe * dataX[p][1] + thb) + wac * sigmoid(wcd * dataX[p][0] + wce * dataX[p][1] + thc) + tha)

    error = (outa-dataY[p])**2
    print(dataY[p], outa, error)

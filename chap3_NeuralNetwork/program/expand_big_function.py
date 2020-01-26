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

# 展開した関数
def expand(x1, x2):
    y = sigmoid(1.94 * sigmoid(-1.54 * x1 + 1.60 * x2 + -0.92) + -1.88 * sigmoid(-1.21 * x1 + 1.29 * x2 + 0.58) + 0.88)
    return y

for p in range(len(dataX)):

    ##########
    # 前向き計算
    ##########
    #outa = sigmoid(wab * sigmoid(wbd * dataX[p][0] + wbe * dataX[p][1] + thb) + wac * sigmoid(wcd * dataX[p][0] + wce * dataX[p][1] + thc) + tha)

    x1 = dataX[p][0]
    x2 = dataX[p][1]
    outa = expand(x1, x2)

    error = (outa-dataY[p])**2
    print(dataY[p], outa, error)

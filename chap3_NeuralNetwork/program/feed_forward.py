# ニューラルネットワークの前向き計算
# 練習用のため拡張性ゼロのアホアホプログラミング

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

# 乱数で与える場合
'''
wab = (np.random.rand()-0.5)*2 * 0.3 # -0.3から0.3の一様乱数
wac = (np.random.rand()-0.5)*2 * 0.3
wbd = (np.random.rand()-0.5)*2 * 0.3
wbe = (np.random.rand()-0.5)*2 * 0.3
wcd = (np.random.rand()-0.5)*2 * 0.3
wce = (np.random.rand()-0.5)*2 * 0.3
tha = (np.random.rand()-0.5)*2 * 0.3
thb = (np.random.rand()-0.5)*2 * 0.3
thc = (np.random.rand()-0.5)*2 * 0.3
'''

for p in range(len(dataX)):

    ##########
    # 前向き計算
    ##########
    
    # 入力層
    outd = dataX[p][0]
    oute = dataX[p][1]
    
    # 中間層
    xb = wbd * outd + wbe * oute + thb
    outb = sigmoid(xb)
    
    xc = wcd * outd + wce * oute + thc
    outc = sigmoid(xc)
    
    # 出力層
    xa = wab * outb + wac * outc + tha
    outa = sigmoid(xa)
    
    error = (outa-dataY[p])**2
    print(dataY[p], outa, error)

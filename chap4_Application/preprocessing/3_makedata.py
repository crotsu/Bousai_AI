#
# 災害画像と非災害画像の画像から学習用の数値データに変換する
#

import glob
import os
from PIL import Image
import numpy as np

# 画像サイズ
photo_size = 200

# ディレクトリがdisasterとnon_disasterなので，これをループするようにする
prefix_name = ["", "non_"]

# 変換後の数値データを格納する
X = []
y = []

for pre in prefix_name:
    path = "../disaster_decision/" + pre + "disaster/*"

    # ファイル名を取得(lsと同じ)
    files = glob.glob(path)

    for filename in files:
        print(filename)

        img = Image.open(filename)

        data = np.asarray(img)
        data = data/256 # 0.0から1.0に正規化
        data = data.reshape(photo_size, photo_size, 3)

        X.append(data)
        if pre=="":
            y.append(0) # 非災害
        else:
            y.append(1) # 災害

X = np.array(X, dtype=np.float32)
np.savez("disaster.npz", X=X, y=y)
print("finished!")

#
# test_dataにあるファイルからテストデータを作成する.
#

import glob
import os
from PIL import Image
import numpy as np

# 余白は埋めてリサイズする．
# 以下を参考にした
# https://note.nkmk.me/python-pillow-add-margin-expand-canvas/
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

# 各種パラメータ
# 画像サイズ
photo_size = 200
# パス
path = "./test_data/*"
# 変換後のファイル名
basename = "testIMG_"

#
# 1. rename
#
files = glob.glob(path)

ct = 0
for filename in files:
    #print(filename)

    # ファイル名と拡張子に分ける
    root, ext = os.path.splitext(filename)

    # 拡張子をすべて小文字にする（例：JPG -> jpg）
    ext = ext.lower()

    # リネームする
    new_filename = path[:-1] + basename + str(ct) + ext
    #print(new_filename)
    os.rename(filename, new_filename)

    ct += 1

print("")
print("1. [rename] finished!")


#
# 2. resize
#
files = glob.glob(path)

for filename in files:
    img = Image.open(filename)
    img = img.convert("RGB")
    img_new = expand2square(img, (0, 0, 0)).resize((photo_size, photo_size))
    img_new.save(filename)

print("2. [resize] finished!")


#
# 3. makedata
#
files = glob.glob(path)

# 変換後の数値データを格納する
X = []
y = []

for filename in files:
    img = Image.open(filename)
    data = np.asarray(img)
    data = data/256 # 0.0から1.0に正規化
    data = data.reshape(photo_size, photo_size, 3)
    X.append(data)
    y.append(1) # 災害か非災害かわからないので，とりあえず0を入れておく．

X = np.array(X, dtype=np.float32)
np.savez("./test.npz", X=X, y=y)
print("3. [makedata] finished!")

#
# 災害画像と非災害画像の画像サイズをそろえる
#

import glob
import os
from PIL import Image

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
    

# 画像サイズ
photo_size = 100
    
# ディレクトリがdisasterとnon_disasterなので，これをループするようにする
prefix_name = ["", "non_"]

for pre in prefix_name:
    path = "../disaster_decision/" + pre + "disaster/*"

    # ファイル名を取得(lsと同じ)
    files = glob.glob(path)

    for filename in files:

        img = Image.open(filename)
        img = img.convert("RGB")
        img_new = expand2square(img, (0, 0, 0)).resize((photo_size, photo_size))
        img_new.save(filename)

print("finished!")

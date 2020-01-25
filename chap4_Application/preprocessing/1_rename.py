#
# 災害画像と非災害画像の名前を連番に変更する
#

import glob
import os

# ディレクトリがdisasterとnon_disasterなので，これをループするようにする
prefix_name = ["", "non_"]

for pre in prefix_name:
    path = "../disaster_decision/" + pre + "disaster/*"

    # ファイル名を取得(lsと同じ)
    files = glob.glob(path)
    #print(files)

    basename = pre + "disasterIMG_"

    ct = 0
    for filename in files:

        print(filename)

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
    print("[%s] is finished!"%(pre))

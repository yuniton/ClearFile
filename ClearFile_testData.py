from PIL import Image
import os, glob
import numpy as np
import random, math
from tensorflow.python import confusion_matrix
from sklearn.metrics import accuracy_score



# 画像が保存されているディレクトリのパス


root_dir = "TEST"
# 画像が保存されているフォルダ名
categories = ["不良品","良品"]

X = [] # 画像データ
Y = [] # ラベルデータ

# フォルダごとに分けられたファイルを収集
#（categoriesのidxと、画像のファイルパスが紐づいたリストを生成）
allfiles = []
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.png")
    for f in files:
        allfiles.append((idx, f))

for cat, fname in allfiles:
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((197, 197))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

x = np.array(X)
y = np.array(Y)


np.save("ClearFile_data_test_X_150.npy", x)
np.save("ClearFile_data_test_Y_150.npy", y)

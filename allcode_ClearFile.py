%cd ../content/drive/My Drive/Colab Notebooks/ClearFileModel

#ラベリングによる学習/検証データの準備

from PIL import Image
import os, glob
import numpy as np
import random, math

#画像が保存されているルートディレクトリのパス
root_dir = "Model"
# 商品名
categories = ["不良品","良品"]

# 画像データ用配列
X = []
# ラベルデータ用配列
Y = []

#画像データごとにadd_sample()を呼び出し、X,Yの配列を返す関数
def make_sample(files):
    global X, Y
    X = []
    Y = []
    for cat, fname in files:
        add_sample(cat, fname)
    return np.array(X), np.array(Y)

#渡された画像データを読み込んでXに格納し、また、
#画像データに対応するcategoriesのidxをY格納する関数
def add_sample(cat, fname):
    img = Image.open(fname)
    img = img.convert("RGB")
    img = img.resize((197, 197))
    data = np.asarray(img)
    X.append(data)
    Y.append(cat)

#全データ格納用配列
allfiles = []

#カテゴリ配列の各値と、それに対応するidxを認識し、全データをallfilesにまとめる
for idx, cat in enumerate(categories):
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.png")
    for f in files:
        allfiles.append((idx, f))

#シャッフル後、学習データと検証データに分ける
random.shuffle(allfiles)
th = math.floor(len(allfiles) * 0.8)
train = allfiles[0:th]
test  = allfiles[th:]
# print(len(train))
# print(len(test))
X_train, y_train = make_sample(train)
X_test, y_test = make_sample(test)
xy = (X_train, X_test, y_train, y_test)

#データを保存する（データの名前を「tea_data.npy」としている）
np.save("ClearFile_data.npy", xy)


#モデルの構築

from keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(197,197,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #過学習を防ぐ
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dense(2,activation="sigmoid")) #分類先の種類分設定

#モデル構成の確認
model.summary()

#モデルのコンパイル



from keras import optimizers

model.compile(loss="binary_crossentropy",
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=["acc"])

#データの準備

from keras.utils import np_utils
import numpy as np

categories = ["不良品","良品"]
nb_classes = len(categories)

X_train, X_test, y_train, y_test = np.load("ClearFile_data.npy", allow_pickle=True)

#データの正規化
X_train = X_train.astype("float") / 255
X_test  = X_test.astype("float") / 255

#kerasで扱えるようにcategoriesをベクトルに変換
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test  = np_utils.to_categorical(y_test, nb_classes)


#モデルの学習
# from preparation import X_train, y_train, X_test, y_test

model = model.fit(X_train,
                  y_train,
                  epochs=10,
                  batch_size=8,
                  validation_data=(X_test,y_test))

import matplotlib.pyplot as plt

acc = model.history['acc']
val_acc = model.history['val_acc']
loss = model.history['loss']
val_loss = model.history['val_loss']
plt.ylim([0.3,1.01])

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('ClearFile_accuracy_graph')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('ClearFile_loss_graph')

#モデルの保存

json_string = model.model.to_json()
open('ClearFile_modelResult.json', 'w').write(json_string)

#重みの保存

hdf5_file = "ClearFile_modelResult.hdf5"
model.model.save_weights(hdf5_file)

from PIL import Image
import os, glob
import numpy as np
import random, math


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

# モデルの精度を測る

#評価用のデータの読み込み
import numpy as np

eval_X = np.load("ClearFile_data_test_X_150.npy")
eval_Y = np.load("ClearFile_data_test_Y_150.npy")


#Yのデータをone-hotに変換
from keras.utils import np_utils

eval_X = eval_X.astype("float")

# eval_Y = eval_Y.astype("float") / 255
test_Y = np_utils.to_categorical(eval_Y,2)
score = model.model.evaluate(x=eval_X,y=test_Y,batch_size=8)

print('loss=', score[0])
print('accuracy=', score[1])

#綾鷹を選ばせるプログラム
import re
from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('ClearFile_modelResult.json').read())
#保存した重みの読み込み
model.load_weights('ClearFile_modelResult.hdf5')

categories = ["不良品","良品"]

# %cd TEST/不良品
%cd TEST/良品

IMG=os.listdir()#順序関係ない
p = re.compile(r'\d+')
im=sorted(IMG, key=lambda s: int(p.search(s).group()))
# print(im)

# %cd ../..

# # # print(len(im))
for i in range(len(im)):
  img_path =str(im[i])
  img = image.load_img(img_path,target_size=(197, 197,3))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  #予測
  features = model.predict(x)

  #予測結果によって処理を分ける
  if features[0,1] == 1:
    print (im[i]+"は良品です。")

  else:
    print(im[i]+"は不良品です。")


%cd ../..



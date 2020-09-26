#モデルのコンパイル



from keras import optimizers

from ClearFile_model import model

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
from preparation import X_train, y_train, X_test, y_test

model = model.fit(X_train,
                  y_train,
                  epochs=10,
                  batch_size=6,
                  validation_data=(X_test,y_test))
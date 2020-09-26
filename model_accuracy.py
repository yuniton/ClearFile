# モデルの精度を測る

#評価用のデータの読み込み
import numpy as np

from model_compile import model
from ClearFile_testData import x, y

eval_X = np.load("ClearFile_data_test_X_150.npy")
eval_Y = np.load("ClearFile_data_test_Y_150.npy")

#Yのデータをone-hotに変換
from keras.utils import np_utils

test_Y = np_utils.to_categorical(eval_Y,2)

score = model.model.evaluate(x=eval_X,y=test_Y)

print('loss=', score[0])
print('accuracy=', score[1])


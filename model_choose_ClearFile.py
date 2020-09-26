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
    print ("良品です。")

  else:
    print("不良品です。")



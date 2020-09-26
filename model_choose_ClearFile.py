#綾鷹を選ばせるプログラム

from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

#保存したモデルの読み込み
model = model_from_json(open('ClearFile_modelResult.json').read())
#保存した重みの読み込み
model.load_weights('ClearFile_modelResult.hdf5')

categories = ["不良品","良品"]

# IMG=os.listdir()#順序関係ない
# IMG.sort()#sortする
# print(IMG)

# # 処理なし　未知
# IMG=['DSC_5960.png', 'DSC_5965.png', 'DSC_5970.png', 'DSC_5972.png', 'DSC_5977.png', 'DSC_5982.png', 'DSC_5987.png', 'DSC_5988.png', 'DSC_5993.png', 'DSC_5998.png', 'DSC_6003.png', 'DSC_6004.png', 'DSC_6009.png', 'DSC_6014.png', 'DSC_6019.png', 'DSC_6020.png',
#  'DSC_6025.png', 'DSC_6030.png', 'DSC_6035.png', 'DSC_6036.png', 'DSC_6041.png', 'DSC_6046.png', 'DSC_6051.png', 'DSC_6052.png', 'DSC_6057.png', 'DSC_6062.png', 'DSC_6067.png', 'DSC_6068.png', 'DSC_6073.png', 'DSC_6078.png', 'DSC_6083.png', 'DSC_6084.png', 
#  'DSC_6089.png', 'DSC_6094.png', 'DSC_6099.png', 'DSC_6100.png', 'DSC_6105.png', 'DSC_6110.png', 'DSC_6115.png', 'DSC_6116.png', 'DSC_6121.png', 'DSC_6126.png', 'DSC_6131.png', 'DSC_6132.png', 'DSC_6137.png', 'DSC_6142.png', 'DSC_6147.png', 'DSC_6148.png',
#   'DSC_6153.png', 'DSC_6158.png']

# 処理なし　TESTDATA 最終　既存
# IMG=['DSC_5398.png', 'DSC_5403.png', 'DSC_5408.png', 'DSC_5413.png', 'DSC_5414.png', 'DSC_5419.png', 'DSC_5424.png', 'DSC_5431.png', 'DSC_5432.png', 'DSC_5483.png', 'DSC_5488.png', 'DSC_5493.png', 'DSC_5494.png', 'DSC_5499.png', 'DSC_5504.png', 'DSC_5509.png',
#      'DSC_5510.png', 'DSC_5635.png', 'DSC_5640.png', 'DSC_5645.png', 'DSC_5646.png', 'DSC_5651.png', 'DSC_5656.png', 'DSC_5661.png', 'DSC_5662.png', 'DSC_5734.png', 'DSC_5739.png', 'DSC_5744.png', 'DSC_5745.png', 'DSC_5750.png', 'DSC_5755.png', 'DSC_5760.png', 
#      'DSC_5761.png', 'DSC_5766.png', 'DSC_5771.png', 'DSC_5776.png', 'DSC_5777.png', 'DSC_5794.png', 'DSC_5799.png', 'DSC_5804.png', 'DSC_5805.png', 'DSC_5810.png', 'DSC_5815.png', 'DSC_5820.png', 'DSC_5821.png', 'DSC_5826.png', 'DSC_5831.png', 'DSC_5836.png', 
#      'DSC_5837.png', 'DSC_5842.png']


# 処理なし　良品
# IMG=['1.png のコピー.png', '2.png のコピー.png', '3.png のコピー.png', '4.png のコピー.png', 'DSC_5845.png', 'DSC_5846.png', 'DSC_5847.png', 'DSC_5848.png', 'DSC_5849.png', 'DSC_5850.png', 'DSC_5851.png', 'DSC_5852.png', 'DSC_5853.png', 'DSC_5854.png', 'DSC_5855.png', 
# 'DSC_5856.png', 'DSC_5857.png', 'DSC_5858.png', 'DSC_5859.png', 'DSC_5860.png', 'DSC_5861.png', 'DSC_5862.png', 'DSC_5863.png', 'DSC_5864.png', 'DSC_5865.png', 'DSC_5866.png', 'DSC_5867.png', 'DSC_5868.png', 'DSC_5869.png', 'DSC_5870.png', 'DSC_5871.png', 
# 'DSC_5872.png', 'DSC_5873.png', 'DSC_5874.png', 'DSC_5875.png', 'DSC_5876.png', 'DSC_5877.png', 'DSC_5878.png', 'DSC_5879.png', 'DSC_5880.png', 'DSC_5881.png', 'DSC_5882.png', 'DSC_5883.png', 'DSC_5884.png', 'DSC_5885.png', 'DSC_5886.png', 'DSC_5887.png', 
# 'DSC_5888.png', 'DSC_5889.png', 'DSC_5890.png']


# # 処理あり　良品
IMG=['diff10.png', 'diff11.png', 'diff12.png', 'diff13.png', 'diff14.png', 'diff15.png', 'diff16.png', 'diff17.png', 'diff18.png', 'diff19.png', 'diff20.png', 'diff21.png', 'diff22.png', 'diff23.png', 'diff24.png', 'diff25.png', 'diff26.png', 'diff27.png', 'diff28.png', 'diff29.png', 'diff30.png', 'diff31.png', 
'diff32.png', 'diff33.png', 'diff34.png', 'diff35.png', 'diff36.png', 'diff37.png', 'diff38.png', 'diff39.png', 'diff4.png', 'diff40.png', 'diff41.png', 'diff42.png', 'diff43.png', 'diff44.png', 'diff45.png', 'diff46.png', 'diff47.png', 'diff48.png', 'diff49.png', 'diff5.png', 'diff50.png', 'diff51.png', 'diff52.png',
 'diff53.png', 'diff6.png', 'diff7.png', 'diff8.png', 'diff9.png']

# 処理あり　不良品 既存
# IMG=['diff0.png', 'diff1.png', 'diff100.png', 'diff101.png', 'diff102.png', 'diff103.png', 'diff104.png', 'diff105.png', 'diff106.png', 'diff107.png', 'diff108.png', 'diff19.png', 'diff2.png', 'diff20.png', 'diff21.png', 'diff22.png', 'diff23.png', 'diff24.png', 'diff25.png', 'diff26.png',
#      'diff3.png', 'diff4.png', 'diff5.png', 'diff57.png', 'diff58.png', 'diff59.png', 'diff6.png', 'diff60.png', 'diff61.png', 'diff62.png', 'diff63.png', 'diff64.png', 'diff7.png', 'diff8.png', 'diff81.png', 'diff82.png', 'diff83.png', 'diff84.png', 'diff85.png', 'diff86.png', 'diff87.png', 'diff88.png',
#      'diff89.png', 'diff90.png', 'diff91.png', 'diff92.png', 'diff96.png', 'diff97.png', 'diff98.png', 'diff99.png']

# 処理あり　不良品　未知のデータ
# IMG=['diff0.png', 'diff1.png', 'diff2.png', 'diff3.png', 'diff4.png', 'diff5.png', 'diff6.png', 'diff7.png', 'diff8.png', 'diff9.png', 'diff10.png', 'diff11.png', 'diff12.png', 'diff13.png', 'diff14.png', 'diff15.png', 'diff16.png', 'diff17.png', 'diff18.png',
# 'diff19.png', 'diff20.png', 'diff21.png', 'diff22.png', 'diff23.png', 'diff24.png', 'diff25.png', 'diff26.png', 'diff27.png', 'diff28.png', 'diff29.png', 'diff30.png', 'diff31.png', 
# 'diff32.png', 'diff33.png', 'diff34.png', 'diff35.png', 'diff36.png', 'diff37.png', 'diff38.png', 'diff39.png', 'diff4.png', 'diff40.png', 'diff41.png', 'diff42.png', 'diff43.png', 
# 'diff44.png', 'diff45.png', 'diff46.png', 'diff47.png', 'diff48.png', 'diff49.png']


# # # print(len(IMG))
for i in range(len(IMG)):
  img_path =str(IMG[i])
  img = image.load_img(img_path,target_size=(197, 197 3))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  #予測
  features = model.predict(x)

  #予測結果によって処理を分ける
  if features[0,1] == 1:
    print ("良品です。")

  else:
    print("不良品です。")



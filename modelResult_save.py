#モデルの保存
from model_compile import model

json_string = model.model.to_json()
open('ClearFile_modelResult.json', 'w').write(json_string)

#重みの保存

hdf5_file = "ClearFile_modelResult.hdf5"
model.model.save_weights(hdf5_file)
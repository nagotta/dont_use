import os
"""
train.zipを解凍し、内部のtrainをNotebookと同じフォルダーにコピーし、
訓練データとテストデータに振り分ける以下のコードを実行
"""
num_train = 1000                    # 訓練データに使用するdog,catの各枚数
num_validation = 400                # テストデータに使用するdog,catの各枚数

source_dir = "./train"              # 元画像のtrainフォルダーの場所
train_dir = "./data/train"          # 訓練データの保存先
valid_dir = "./data/validation"     # テストデータの保存先

os.makedirs("%s/dogs" % train_dir)  # train_dir内にdogsフォルダーを作成
os.makedirs("%s/cats" % train_dir)  # train_dir内にcatsフォルダーを作成
os.makedirs("%s/dogs" % valid_dir)  # valid_dir内にdogsフォルダーを作成
os.makedirs("%s/cats" % valid_dir)  # valid_dir内にcatsフォルダーを作成

# 訓練データの用意 #
# cat,dogの画像1000枚をtrain_dirのdogs,catsに移動
for i in range(num_train):
    #「cat.1.jpg～cat.1000.jpg」を
    #「cat.0001.jpg～cat.1000.jpg」にリネームして移動
    os.rename("%s/cat.%d.jpg" % (source_dir, i + 1),
              "%s/cats/cat%04d.jpg" % (train_dir, i + 1))
    #「dog.1.jpg～dog.1000.jpg」を
    #「dog.0001.jpg～dog.2000.jpg」にリネームして移動
    os.rename("%s/dog.%d.jpg" % (source_dir, i + 1),
              "%s/dogs/dog%04d.jpg" % (train_dir, i + 1))

# テストデータの用意 #
# dog,catの画像の1001枚目以降の400枚をvalid_dirのdogs,catsに移動
for i in range(num_validation):
    #「cat.1001.jpg～cat.1400.jpg」を
    #「cat.0001.jpg～cat.0400.jpg」にリネームして移動
    os.rename("%s/cat.%d.jpg" % (source_dir, num_train + i + 1),
              "%s/cats/cat%04d.jpg" % (valid_dir, i + 1))
    #「dog.1001.jpg～dog.1400.jpg」を
    #「dog.0001.jpg～dog.0400.jpg」にリネームして移動
    os.rename("%s/dog.%d.jpg" % (source_dir, num_train + i + 1),
              "%s/dogs/dog%04d.jpg" % (valid_dir, i + 1))
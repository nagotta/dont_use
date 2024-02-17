###############
#画像の機械学習
###############
import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils.np_utils import to_categorical

#飲み物の名前リスト
drink_name = ["コカ・コーラ", "綾鷹", "アクエリアス"]

# 教師データのラベル付け
X_train = [] 
Y_train = []
i = 0
#飲み物の名前ごとに処理する
for name in drink_name:
    #ファルダーの中身の画像を一覧にする
    img_file_name_list=os.listdir("./data/"+name)
    #確認
    print(len(img_file_name_list))
    #画像ファイルごとに処理
    for img_file_name in img_file_name_list:
        #パスを結合
        n=os.path.join("./data/"+name+"/"+img_file_name)
        img = cv2.imread(n)
        #色成分を分割
        b,g,r = cv2.split(img)
        #色成分を結合
        img = cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)
    i += 1

# テストデータのラベル付け
X_test = [] # 画像データ読み込み
Y_test = [] # ラベル（名前）
#飲み物の名前ごとに処理する
i = 0
for name in drink_name:
    img_file_name_list=os.listdir("./test/"+name)
    #確認
    print(len(img_file_name_list))
    #ファイルごとに処理
    for img_file_name in img_file_name_list:
        n=os.path.join("./test/" + name + "/" + img_file_name)
        img = cv2.imread(n)
        #色成分を分割
        b,g,r = cv2.split(img)
        #色成分を結合
        img = cv2.merge([r,g,b])
        X_test.append(img)
        # ラベルは整数値
        Y_test.append(i)
    i += 1
#配列化
X_train=np.array(X_train)
X_test=np.array(X_test)

#ラベルをone-hotベクトルにする？
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

# モデルの定義
model = Sequential()
#畳み込みオートエンコーダーの動作
#ここの64は画像サイズ
#画像サイズがあっていないと、エラーが発生する
#3×3のフィルターに分ける
model.add(Conv2D(input_shape=(64, 64, 3), filters=32,kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
#2×2の範囲で最大値を出力
model.add(MaxPooling2D(pool_size=(2, 2)))
#畳み込みオートエンコーダーの動作
#3×3のフィルターに分ける
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
#2×2の範囲で最大値を出力
model.add(MaxPooling2D(pool_size=(2, 2)))
#畳み込みオートエンコーダーの動作
#3×3のフィルターに分ける
model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                 strides=(1, 1), padding="same"))
#2×2の範囲で最大値を出力
model.add(MaxPooling2D(pool_size=(2, 2)))
#1次元配列に変換
model.add(Flatten())
#出力の次元数を256にする
model.add(Dense(256))
#非線形変形の処理をするらしい
model.add(Activation("sigmoid"))
#出力の次元数を128にする
model.add(Dense(128))
#非線形変形の処理をするらしい
model.add(Activation('sigmoid'))
#出力の次元数を3にする
#今回3種類のジュースなので、3
model.add(Dense(3))
#非線形変形の処理をするらしい
model.add(Activation('softmax'))

# コンパイル
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 学習
history = model.fit(X_train, y_train, batch_size=32, #画像の枚数に近い2^n
                    epochs=110, verbose=1, validation_data=(X_test, Y_test))#epochs90-110

# 汎化制度の評価・表示
score = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#モデルを保存
model.save("./drink_img_model.h5")
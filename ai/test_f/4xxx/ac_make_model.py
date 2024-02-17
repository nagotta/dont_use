import numpy as np
import os
import cv2
import keras.backend as K 
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from keras.models import Model


FM_name = ["men", "women"]

X_train = []
Y_train = []

i = 0
for name in FM_name:
    #ファルダーの中身の画像を一覧にする
    img_file_name_list=os.listdir("./newdata/resize_val/train/"+name)
    #確認
    print(name,"画像枚数",len(img_file_name_list))
    #画像ファイルごとに処理
    for img_file_name in img_file_name_list:
        #パスを結合
        n=os.path.join("./newdata/resize_val/train/"+name+"/"+img_file_name)
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
for name in FM_name:
    img_file_name_list=os.listdir("./newdata/resize_val/validation/"+name)
    #確認
    print(name,"画像枚数",len(img_file_name_list))
    #ファイルごとに処理
    for img_file_name in img_file_name_list:
        n=os.path.join("./newdata/resize_val/validation/" + name + "/" + img_file_name)
        img = cv2.imread(n)
        #色成分を分割
        b,g,r = cv2.split(img)
        #色成分を結合
        img = cv2.merge([r,g,b])
        X_test.append(img)
        # ラベルは整数値
        Y_test.append(i)
    i += 1

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

input_layer = Input(shape=(64,64,3))

x = Conv2D(filters = 32, kernel_size = 3, strides = 1, padding = 'same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = 'same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)


x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.5)(x)

x = Dense(2)(x)
output_layer = Activation('softmax')(x)

model = Model(input_layer, output_layer)
model.summary()


# コンパイル
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train
          , Y_train
          , batch_size=8
          , epochs=10
          , shuffle=True
          , validation_data = (X_test, Y_test))

model.layers[6].get_weights()

score = model.evaluate(X_test, Y_test, batch_size=8, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#モデルを保存
model.save("./m_or_w2num.h5")

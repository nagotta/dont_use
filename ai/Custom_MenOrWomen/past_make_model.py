from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2

img_w, img_h = 64, 64        # 画像をリサイズするときのサイズ
batch_size = 64                # ミニバッチのサイズ
FM_name = ["men","women"]

# X_train = []
# Y_train = []
# i = 0
# for name in FM_name:
#     #ファルダーの中身の画像を一覧にする
#     img_file_name_list=os.listdir("./data/assets/"+name)
#     #確認
#     print(len(img_file_name_list))
#     #画像ファイルごとに処理
#     for img_file_name in img_file_name_list:
#         #パスを結合
#         n=os.path.join("./data/assets/"+name+"/"+img_file_name)
#         img = cv2.imread(n)
#         #色成分を分割
#         b,g,r = cv2.split(img)
#         #色成分を結合
#         img = cv2.merge([r,g,b])
#         X_train.append(img)
#         Y_train.append(i)
#     i += 1

# # テストデータのラベル付け
# X_test = [] # 画像データ読み込み
# Y_test = [] # ラベル（名前）
# #飲み物の名前ごとに処理する
# i = 0
# for name in FM_name:
#     img_file_name_list=os.listdir("./data/validation/"+name)
#     #確認
#     print(len(img_file_name_list))
#     #ファイルごとに処理
#     for img_file_name in img_file_name_list:
#         n=os.path.join("./data/validation/" + name + "/" + img_file_name)
#         img = cv2.imread(n)
#         #色成分を分割
#         b,g,r = cv2.split(img)
#         #色成分を結合
#         img = cv2.merge([r,g,b])
#         X_test.append(img)
#         # ラベルは整数値
#         Y_test.append(i)
#     i += 1


# 訓練データを読み込んで加工するジェネレーターを生成
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,         # RGB値を0～1の範囲に変換
    rotation_range=15,         # 15度の範囲でランダムに回転させる
    zoom_range=0.2,            # ランダムに拡大
    horizontal_flip=True       # 水平方向にランダムに反転
    )

# テストデータを読み込むジェネレーターを生成
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# 訓練データをミニバッチの数だけ生成
train_generator = train_datagen.flow_from_directory(
    'data/assets',               # 訓練データのフォルダー
    target_size=(img_w, img_h), # 画像をリサイズ
    batch_size=batch_size,      # ミニバッチのサイズ
    classes = FM_name,
    class_mode='categorical')        # 二値分類なのでbinaryを指定

# テストデータをミニバッチの数だけ生成
validation_generator = test_datagen.flow_from_directory(
    'data/validation',          # テストデータのフォルダー
    target_size=(img_w, img_h), # 画像をリサイズ
    batch_size=batch_size,      # ミニバッチのサイズ
    classes = FM_name,
    class_mode='categorical')        # 二値分類なのでbinaryを指定

print(train_generator.class_indices)

# モデルを構築
model = Sequential()

# （第1層）畳み込み層
model.add(
    Conv2D(
        filters=32,                # フィルターの数は32
        kernel_size=(3, 3),        # 3×3のフィルターを使用
        input_shape=(64, 64, 3), # 入力データの形状
        padding='same',            # ゼロパディングを行う
        activation='relu'          # 活性化関数はReLU
        ))
# （第2層）プーリング層
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
# ドロップアウト25％
model.add(Dropout(0.25))

# （第3層）畳み込み層
model.add(
    Conv2D(
        filters=32,            # フィルターの数は32
        kernel_size=(3, 3),    # 3×3のフィルターを使用
        activation='relu'      # 活性化関数はReLU
        ))
# （第4層）プーリング層
model.add(
    MaxPooling2D(pool_size=(2, 2))
)
# ドロップアウト25％
model.add(Dropout(0.25))

# （第5層）畳み込み層
model.add(
    Conv2D(filters=64,         # フィルターの数は64
           kernel_size=(3, 3), # 3×3のフィルターを使用
           activation='relu'   # 活性化関数はReLU
          ))
# （第6層）プーリング層
model.add(
    MaxPooling2D(pool_size=(2, 2)))
# ドロップアウト25％
model.add(Dropout(0.25))

# 出力層への入力を4階テンソルから2階テンソルに変換する
model.add(Flatten())

# （第7層）全結合層
model.add(
    Dense(64,                  # ニューロン数は64
          activation='relu'))  # 活性化関数はReLU
# ドロップアウト50％
model.add(Dropout(0.5))

# （第8層）出力層
model.add(
    Dense(
        1,                     # ニューロン数は1個
        activation='sigmoid')) # 活性化関数はsigmoid

# モデルのコンパイル
model.compile(
    loss='binary_crossentropy', # バイナリ用の交差エントロピー誤差
    metrics=['accuracy'],       # 学習評価として正解率を指定
    # Adamアルゴリズムで最適化
    optimizer=optimizers.Adam(),
)

# モデルのサマリを表示
model.summary()

epochs = 10             # 学習回数
batch_size = batch_size # 設定済みのミニバッチのサイズ

# 学習を行う
history = model.fit(
    train_generator,    # 訓練データ
    epochs=epochs,      # 学習回数
    verbose=1,          # 学習の進捗状況を出力する
    # テストデータ
    validation_data=validation_generator,
)

score = model.evaluate(validation_generator, batch_size=32, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))

#モデルを保存
model.save("./m_or_w.h5")

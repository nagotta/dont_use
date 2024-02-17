# keras.utilsからnp_utilsをインポート
from keras.utils import np_utils
# MNISTデータセットをインポート
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
# MNISTデータセットの読み込み
(x_trains, y_trains), (x_tests, y_tests) = mnist.load_data()

# 訓練データ
# 60000x28x28の3次元配列を60000×28×28×1の4次元配列に変換
x_trains = x_trains.reshape(60000, 28, 28, 1)
x_trains = x_trains.astype('float32') # float32型に変換
x_trains /= 255                      # 0から1.0の範囲に変換
correct = 10                        # 正解ラベルの数
# 正解ラベルを1-of-K符号化法で変換
y_trains = np_utils.to_categorical(y_trains, correct)

# テストデータ
# 10000x28x28の3次元配列を10000×28×28×1の4次元配列に変換
x_tests = x_tests.reshape(10000, 28, 28, 1)
# テストデータをfloat32(浮動小数点数)型に変換
x_tests = x_tests.astype('float32') # float32型に変換
x_tests /= 255                     # 0から1.0の範囲に変換
# 正解ラベルをワンホット表現に変換
y_tests = np_utils.to_categorical(y_tests, correct)
model = Sequential()                      # Sequentialオブジェクトの生成

# （第1層）畳み込み層1
# ニューロン数：16
# 出力：1ニューロンあたり(28, 28, 1)の3階テンソルを16個出力
model.add(Conv2D(filters=16,              # フィルターの数は16
                 kernel_size=(3, 3),      # 3×3のフィルターを使用
                 input_shape=(28, 28, 1), # 入力データのサイズ
                 padding='same',          # ゼロパディングを行う
                 activation='relu'        # 活性化関数はReLU
                ))

# （第2層）畳み込み層2
# ニューロン数：32
# 出力：1ニューロンあたり(28, 28, 1)の3階テンソルを32個出力
model.add(Conv2D(filters=32,              # フィルターの数は32
                 kernel_size=(3, 3),      # 3×3のフィルターを使用
                 padding='same',          # ゼロパディングを行う
                 activation='relu'        # 活性化関数はReLU
                ))

# （第3層）プーリング層1
# ユニット数：32
# 出力：1ユニットあたり(14, 14, 1)の3階テンソルを32個出力
model.add(
    MaxPooling2D(pool_size=(2, 2)))       # 縮小対象の領域は2×2

# （第4層）畳み込み層3
# ニューロン数：64
# 出力：1ニューロンあたり(14, 14, 1)の3階テンソルを64個出力
model.add(Conv2D(filters=64,              # フィルターの数は64
                 kernel_size=(3, 3),      # 3×3のフィルターを使用
                 padding='same',          # ゼロパディングを行う
                 activation='relu'        # 活性化関数はReLU
                ))

# （第5層）プーリング層2
# 出力：1ユニットあたり(7, 7, 1)の3階テンソルを64個出力
model.add(
    MaxPooling2D(pool_size=(2, 2)))       # 縮小対象の領域は2×2

# ドロップアウト層
# 出力：1ユニットあたり(7, 7, 1)の3階テンソルを64個出力
model.add(Dropout(0.5))                   # ドロップアウトは25％

# Flaten層
# (7, 7)から(49)に変換する
# 出力：要素数(3136)の1階テンソル
model.add(Flatten())

# （第6層）全結合層
# ニューロン数：128
# 出力：要素数(128)の1階テンソルを出力
model.add(Dense(128,                      # ニューロン数は128
                activation='relu'         # 活性化関数はReLU
               ))

# （第7層）出力層
# ニューロン数：10
# 出力：要素数(10)の1階テンソルを出力
model.add(Dense(10,                       # 出力層のニューロン数は10
                activation='softmax'      # 活性化関数はソフトマックス
               ))

# Sequentialオブジェクトのコンパイル
model.compile(
    loss='categorical_crossentropy',      # 損失の基準は交差エントロピー誤差
    optimizer=Adam(),                     # 学習方法をAdamにする
    metrics=['accuracy']                  # 学習評価として正解率を指定
    )

model.summary()
# 学習を行って結果を出力
history = model.ﬁt(x_trains,            # 訓練データ
                   y_trains,            # 正解ラベル 
                   batch_size=100,      # 勾配計算に用いるミニバッチの数 
                   epochs=20,           # 学習を繰り返す回数
                   verbose=1,           # 学習の進捗状況を出力する
                   validation_data=(
                       x_tests, y_tests # テストデータの指定
                   ))
# プロット図のサイズを設定
plt.ﬁgure(ﬁgsize=(15, 6))
# プロット図を縮小して図の間のスペースを空ける
plt.subplots_adjust(wspace=0.5)

# 1×2のグリッドの左(1,2,1)の領域にプロット
plt.subplot(1, 2, 1)
# 訓練データの損失(誤り率)をプロット
plt.plot(history.history['loss'],
         label='training',
         color='black')
# テストデータの損失(誤り率)をプロット
plt.plot(history.history['val_loss'],
         label='test',
         color='red')
plt.ylim(0, 1)       # y軸の範囲
plt.legend()         # 凡例を表示
plt.grid()           # グリッド表示
plt.xlabel('epoch')  # x軸ラベル
plt.ylabel('loss')   # y軸ラベル

# 1×2のグリッドの右(1,2,21)の領域にプロット
plt.subplot(1, 2, 2)
# 訓練データの正解率をプロット
plt.plot(history.history['accuracy'],
         label='training',
         color='black')
# テストデータの正解率をプロット
plt.plot(history.history['val_accuracy'],
         label='test',
         color='red')
plt.ylim(0.5, 1)     # y軸の範囲
plt.legend()         # 凡例を表示
plt.grid()           # グリッド表示
plt.xlabel('epoch')  # x軸ラベル
plt.ylabel('accuracy')    # y軸ラベル
plt.show()
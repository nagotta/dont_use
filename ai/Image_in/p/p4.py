from keras.datasets import mnist
# keras.utilsからnp_utilsをインポート
from keras.utils import np_utils
# keras.modelsからSequentialをインポート
from keras.models import Sequential
# keras.layersからDense、Activationをインポート
from keras.layers import Dense, Activation
from tensorflow.keras import optimizers
# keras.optimizersからAdamをインポート
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_trains, y_trains), (x_tests, y_tests) = mnist.load_data()
print(x_trains.shape)  # (60000, 28, 28) 訓練データ用の画像が60000
print(y_trains.shape)  # 出力：(60000,) 訓練データの正解ラベルが60000
print(x_tests.shape)   # 出力：(10000, 28, 28) テスト用の画像が10000
print(y_tests.shape)   # 出力：(10000,) テスト用の正解ラベルが10000
# 正解ラベルのクラスの数
classes = 10

# 訓練データ
# 60000x28x28の3次元配列を60000×784の2次元配列に変換
x_trains = x_trains.reshape(60000, 784)
# 訓練データをfloat32(浮動小数点数)型に変換
x_trains = x_trains.astype('float32')
# データを255で割って0から1.0の範囲に変換
x_trains = x_trains / 255
# 正解ラベルをワンホット表現に変換
y_trains = np_utils.to_categorical(y_trains, classes)

# テストデータ
# 10000x28x28の3次元配列を10000×784の2次元配列に変換
x_tests = x_tests.reshape(10000, 784)
# テストデータをfloat32(浮動小数点数)型に変換
x_tests = x_tests.astype('float32')
# データを255で割って0から1.0の範囲に変換
x_tests = x_tests / 255
# 正解ラベルをワンホット表現変換
y_tests = np_utils.to_categorical(y_tests, classes)
model = Sequential()                 # Sequentialのインスタンス化
model.add(Dense(200,                 # 隠れ層のニューロン数は200
                input_dim=784,       # 入力層のニューロン数は784
                activation='sigmoid' # 活性化関数はシグモイド
               ))
model.add(Dense(10,                  # 出力層のニューロン数は10
                activation='softmax' # 活性化関数はソフトマックス
               ))
model.compile(
    loss='categorical_crossentropy', # 誤差関数は交差エントロピー誤差
    optimizer=Adam(),                # 学習方法をAdamにする
    metrics=['accuracy']             # 学習評価として正解率を指定
    )
model.summary() # ニューラルネットワークのサマリー（概要）を出力
# 学習を行って結果を出力
history = model.fit(
    x_trains,         # 訓練データ
    y_trains,         # 正解ラベル
    epochs=10,        # 学習を繰り返す回数
    batch_size=100,   # 勾配計算に用いるミニバッチの数
    verbose=1,        # 学習の進捗状況を出力する
    validation_data=(
    x_tests, y_tests  # テストデータの指定
    ))

# プロット図のサイズを設定
plt.ﬁgure(ﬁgsize=(15, 6))
# プロット図を縮小して図の間のスペースを空ける
plt.subplots_adjust(wspace=0.2)

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
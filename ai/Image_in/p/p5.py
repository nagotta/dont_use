import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
# CIFAR-10データを読み込む
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 訓練データのピクセル値を0～1の範囲にに変換
X_train = X_train.astype('float32')
X_train /= 255.0
# テストデータのピクセル値を0～1の範囲にに変換
X_test = X_test.astype('float32')
X_test /= 255.0

# 正解ラベルを10クラスのワンホット表現に変換
classes = 10
Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)
# CNNを構築
model = Sequential()

# （第1層）畳み込み層1
model.add(
    Conv2D(
        filters=32,                # フィルターの数は32
        kernel_size=(3, 3),        # 3×3のフィルターを使用
        input_shape=(32, 32, 3),   # 入力データの形状
        padding='same',            # ゼロパディングを行う
        activation='relu'          # 活性化関数はReLU
        ))

# （第2層）畳み込み層2
model.add(
    Conv2D(
        filters=32,                # フィルターの数は32
        kernel_size=(3, 3),        # 3×3のフィルターを使用
        padding='same',            # ゼロパディングを行う
        activation='relu'          # 活性化関数はReLU
        ))

# （第3層）プーリング層1：ウィンドウサイズは2×2
model.add(
    MaxPooling2D(pool_size=(2, 2))
)

# ドロップアウト層1：ドロップアウトは25％
model.add(Dropout(0.25))

# （第4層）畳み込み層3
model.add(
    Conv2D(filters=64,             # フィルターの数は64
           kernel_size=(3, 3),     # 3×3のフィルターを使用
           padding='same',
           activation='relu'       # 活性化関数はReLU
           ))

# （第5層）畳み込み層4
model.add(
    Conv2D(filters=64,             # フィルターの数は64
           kernel_size=(3, 3),     # 3×3のフィルターを使用
           padding='same',         # ゼロパディングを行う
           activation='relu'       # 活性化関数はReLU
           ))

# （第6層）プーリング層2：ウィンドウサイズは2×2
model.add(
    MaxPooling2D(
        pool_size=(2, 2)
    ))

# ドロップアウト層2　ドロップアウトは25％
model.add(Dropout(0.25))

# Flatten層　4階テンソルから2階テンソルに変換
model.add(Flatten())

# （第7層）全結合層
model.add(
    Dense(512,                     # ニューロン数は512
          activation='relu'))      # 活性化関数はReLU
          

# ドロップアウト層3：ドロップアウトは50％
model.add(Dropout(0.5))

# （第8層）出力層
model.add(
    Dense(classes,                 # 出力層のニューロン数はclasses
          activation='softmax'))   # 活性化関数はソフトマックス


# Sequentialオブジェクトのコンパイル
model.compile(
    loss='categorical_crossentropy', # 損失関数は交差エントロピー誤差
    optimizer='adam',                # 最適化をAdamアルゴリズムで行う
    metrics=['accuracy']             # 学習評価として正解率を指定
    )

# モデルのサマリを表示
model.summary()
# 学習を行う

epochs = 100     # 学習回数
batch_size = 128 # ミニバッチのサイズ

history = model.fit(X_train,               # 訓練データ
                    Y_train,               # 正解ラベル 
                    batch_size=batch_size, # 勾配計算に用いるミニバッチの数 
                    epochs=epochs,         # 学習を繰り返す回数
                    verbose=1,             # 学習の進捗状況を出力する
                    validation_data=(
                        X_test, Y_test),   # テストデータの指定                    
                   shuffle=True)
def plot_history(history):
    plt.ﬁgure(ﬁgsize=(8, 10))
    plt.subplots_adjust(hspace=0.3)
    # 精度の履歴をプロット
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'],"-",label="accuracy")
    plt.plot(history.history['val_accuracy'],"-",label="val_accuracy")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc="lower right")

    # 損失の履歴をプロット
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'],"-",label="loss",)
    plt.plot(history.history['val_loss'],"-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.show()
    
# 学習の過程をグラフにする
plot_history(history)
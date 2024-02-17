from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np

# 訓練データ
train = np.array([[0, 0], # 0と1の組み合わせの行列(4データ,2列)
                  [0, 1],
                  [1, 0],
                  [1, 1]])
label = np.array([[0],    # 正解ラベル(4データ,1列)
                  [1],
                  [1],
                  [0]])
# Sequentialオブジェクトを生成
model = Sequential()

# （第1層）隠れ層
model.add(
    Dense(units=3,               # 隠れ層のニューロン数は3
          input_dim=2,           # 入力層のニューロン数は2
          activation='sigmoid')) # 活性化関数はシグモイド

# （第2層）出力層
model.add(
    Dense(units=1,               # 出力層のニューロン数は1
          activation='sigmoid')) # 活性化関数はシグモイド

model.compile(
    loss='binary_crossentropy', # 誤差関数にバイナリ用のクロスエントロピーを指定
    optimizer=optimizers.SGD(lr=0.1),      # 勾配降下法を指定
)

 # ニューラルネットワークのサマリー（概要）を出力
model.summary()

history = model.fit(
    train,        # 訓練データ
    label,        # 正解ラベル
    epochs=3000,  # 学習回数
    batch_size=4, # ミニバッチのサイズ（今回はすべて使用）
    verbose=0,    # 学習の進捗状況を出力しない
)
# predict_classes()で出力のみを行う
# 0.5を閾値として0または1を取得
classes = model.predict_classes(train, batch_size=4)
# 出力された値そのものを取得
prob = model.predict_proba(train, batch_size=4)

print('Output:')
print(classes)
print('Output Probability:')
print(prob)
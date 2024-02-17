import numpy as np
from keras.utils import to_categorical
# テスト用のデータ生成
data = np.random.randint(low=0, high=5, size=10)
print(data)
# One-Hotベクトルに変換
print(to_categorical(data))
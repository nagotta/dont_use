from keras.preprocessing.image import ImageDataGenerator

img_w, img_h = 150, 150        # 画像をリサイズするときのサイズ
batch_size = 32                # ミニバッチのサイズ

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
    'data/train',               # 訓練データのフォルダー
    target_size=(img_w, img_h), # 画像をリサイズ
    batch_size=batch_size,      # ミニバッチのサイズ
    class_mode='binary')        # 二値分類なのでbinaryを指定

# テストデータをミニバッチの数だけ生成
validation_generator = test_datagen.flow_from_directory(
    'data/validation',          # テストデータのフォルダー
    target_size=(img_w, img_h), # 画像をリサイズ
    batch_size=batch_size,      # ミニバッチのサイズ
    class_mode='binary')        # 二値分類なのでbinaryを指定

print(train_generator.class_indices)
print(validation_generator.class_indices)
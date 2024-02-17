import cv2
import numpy as np
from PIL import Image

#画像を読み込む
img = cv2.imread("./assets/download.jpg")

#RGBからHSVに変換
#HSVがGRAYだと白黒になる
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

#色の範囲指定
#BGRで判断
BGR_lower = np.array([10,120,90])
BGR_upper = np.array([100, 209, 255])

#指定した色に基づいたマスク画像の生成
#指定した色とそれ以外の色に分ける 
mask = cv2.inRange(hsv, BGR_lower, BGR_upper)
#確認
cv2.imwrite("./assets/check.jpg",mask)
#AND演算を行っているらしい
output = cv2.bitwise_and(hsv, hsv, mask = mask)
# 結果のファイルを作成
cv2.imwrite("./assets/mask_fruits.jpg", output)
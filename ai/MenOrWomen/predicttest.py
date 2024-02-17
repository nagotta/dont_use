import numpy as np
import cv2
from keras.models import  load_model
from PIL import Image
import os

face_cascade_path = './haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

#バウンディングボックス座標データを取得
def Get_Bounding_Box(image):
    img_b = cv2.imread(image)
    if img_b is None:
        print("No Object")
        return -1
    #ここでオブジェクトを検出
    #バウンディングボックスもろもろ検出してる
    img_b_gray = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    #バウンディングボックスの座標情報をゲット
    face = face_cascade.detectMultiScale(img_b_gray)
    bbox = []
    for x, y, w, h in face:
        bbox.append([x,y,x+w,y+h])
    print(face)
    #バウンディングボックス座標データを返却
    return bbox

#画像を抜き取り、リサイズ
def detect_object(image, img_url):
    bbox_Coordinate = Get_Bounding_Box(img_url)
    #配列の中が空かどうか判断
    #空だったら、実行しない
    if len(bbox_Coordinate[0]) == 0:
        print("no object")
    else:
        os.makedirs("./stock_room", exist_ok=True)
        #画像読み込み
        img = Image.open(img_url)
        #画像を切り取る
        #座標を格納する箱
        item_position_list = []
        #座標を一個ずつ読み取る
        for item_position in bbox_Coordinate:
            for item_coordinate in item_position:
                item_position_list.append(item_coordinate)
            #取得した座標のところだけを抜き取る
            img_crop = img.crop(item_position_list)
            #画像出力
            img_crop.save("./stock_room/stock.jpg")
            #座標リストを空にする
            item_position_list.clear()
        #サイズ変更
        image_resize = cv2.imread("./stock_room/stock.jpg")
        image_set = cv2.resize(image_resize, (64,64))
    return np.expand_dims(image_set,axis=0)

def detect_who(img):
    #予測
    name=""
    print("predict:",model.predict(img))
    nameNumLabel=np.argmax(model.predict(img))
    print("argmax:",nameNumLabel)
    if nameNumLabel== 0: 
        name="男性"
    elif nameNumLabel==1:
        name="女性"
    return name


if __name__ == '__main__':
    model = load_model('./m_or_w2num.h5')
    # 判別したい画像
    image_url = "./emab.png" #判定したい画像
    image=cv2.imread(image_url)
    if image is None:
        print("Not open:")
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    whoImage=detect_object(image,image_url)
    print(detect_who(whoImage))
    #plt.imshow(whoImage)
    #plt.show()
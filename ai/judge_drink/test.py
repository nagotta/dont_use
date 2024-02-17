import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import  load_model
from PIL import Image
import cvlib as cv
from cvlib.object_detection import draw_bbox
import sys

#バウンディングボックス座標データを取得
def Get_Bounding_Box(image):
    #コーラの画像を読み込む
    #ここでオブジェクトを検出
    #バウンディングボックスもろもろ検出してる
    bbox, label, conf = cv.detect_common_objects(image)
    #バウンディングボックスの描画
    output_image = draw_bbox(image, bbox, label, conf)
    #確認
    # cv2.imwrite("./assets_kora/output_image_ko_ra.jpg",output_image)
    #バウンディングボックスの座標情報をゲット
    bbox=np.array(bbox,dtype="int64")
    #確認用
    print(bbox)
    #バウンディングボックス座標データを返却
    return bbox

#画像を抜き取り、リサイズ
def detect_object(image, img_url):
    bbox_Coordinate = Get_Bounding_Box(image)
    #配列の中が空かどうか判断
    #空だったら、実行しない
    if len(bbox_Coordinate) == 0:
        print("no object")
        sys.exit()
    else:
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
    print(model.predict(img))
    nameNumLabel=np.argmax(model.predict(img))
    if nameNumLabel== 0: 
        name="コカ・コーラ"
    elif nameNumLabel==1:
        name="綾鷹"
    elif nameNumLabel==2:
        name="アクエリアス"
    return name


if __name__ == '__main__':
    model = load_model('./drink_img_model.h5')
    image_url = "./akueriasu/000003.jpg"
    image=cv2.imread(image_url)
    if image is None:
        print("Not open:")
        sys.exit()
    b,g,r = cv2.split(image)
    image = cv2.merge([r,g,b])
    whoImage=detect_object(image,image_url)
    print(detect_who(whoImage))
    #plt.imshow(whoImage)
    #plt.show()
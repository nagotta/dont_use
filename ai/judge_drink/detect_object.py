##########################
#写真から物体だけを切り取る
##########################
import cv2
import glob
import os
import cvlib as cv
import numpy as np
from PIL import Image
from cvlib.object_detection import draw_bbox

#バウンディングボックス座標データを取得
def Get_Bounding_Box(img_path):
    #コーラの画像を読み込む
    img_b = cv2.imread(img_path)
    if img_b is None:
        print("No Object")
        return -1
    #ここでオブジェクトを検出
    #バウンディングボックスもろもろ検出してる
    bbox, label, conf = cv.detect_common_objects(img_b)
    #バウンディングボックスの描画
    output_image = draw_bbox(img_b, bbox, label, conf)
    #確認
    cv2.imwrite("./assets_kora/output_image_ko_ra.jpg",output_image)
    #バウンディングボックスの座標情報をゲット
    bbox=np.array(bbox,dtype="int64")
    #確認用
    print(bbox)
    #バウンディングボックス座標データを返却
    return bbox

#物体だけを取り出す
def Cut_draw(img_path, bbox_Coordinate, img_number, drink_name):
    #画像読み込み
    img = Image.open(img_path).convert("RGB")
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
        img_crop.save("./images/" + drink_name + "/cut_image_drink{0}.jpg".format(img_number))
        #座標リストを空にする
        item_position_list.clear()

if __name__ == '__main__':
    #飲み物リスト
    drink_list = ["コカ・コーラ","綾鷹","アクエリアス","トランプ"]
    #飲み物事に処理を行う
    for drink_name in drink_list:
        #使用する画像のパスを指定する
        img_path = "./assets/" + drink_name + "/*.jpg"
        #１つずつ画像パスを読み取る
        img_jpg = glob.glob(img_path)
        #画像番号
        img_number = 1
        #フォルダーが存在するか確認
        os.makedirs("./images/" + drink_name, exist_ok=True)
        #画像を１つずつ処理する
        for img in img_jpg:
            #バウンディングボックスの座標データを取得&格納
            bbox_Coordinate = Get_Bounding_Box(img)
            if type(bbox_Coordinate) is not int:
                #画像の抜き取り
                Cut_draw(img, bbox_Coordinate, img_number, drink_name)
                img_number += 1
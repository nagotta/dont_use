######################
#画像の水増し作業を行う
######################
import os
import cv2
import glob
from scipy import ndimage

#飲み物リスト
drink_list = ["コカ・コーラ","綾鷹","アクエリアス"]
#飲み物事に処理を行う
for drink_name in drink_list:

    #フォルダーを作成している
    os.makedirs("./data/" + drink_name, exist_ok=True)

    #読み取る画像のフォルダを指定
    in_dir = "./images/" + drink_name + "/*.jpg"
    #出力先のフォルダを指定
    out_dir = "./data/" + drink_name

    #./assets_kora/drink/のフォルダの画像のディレクトリをすべて配列に格納している
    img_jpg = glob.glob(in_dir)
    #./assets_kora/drink/のファイルを一覧にする
    #img_file_name_list =os.listdir("./)

    #画像の個数分繰り返し作業を行う
    for i in range(len(img_jpg)):
        img =cv2.imread(str(img_jpg[i]))
        #--------
        #回転処理
        #--------
        for ang in [-10,0,10]:
            #ang配列を回転させて、画像の角度を変えているらしい
            img_rot = ndimage.rotate(img,ang)
            #64×64サイズにしてる
            img_rot = cv2.resize(img_rot,(64,64))
            #パスを結合している
            fileName = os.path.join(out_dir, str(i) + "_" + str(ang) + ".jpg")
            #画像を出力
            cv2.imwrite(str(fileName),img_rot)
            #--------
            #閾値処理
            #--------
            #閾値を変更している
            #閾値を決め、値化の方法(今回はTHRESH_TOZERO)を決めている
            img_thr = cv2.threshold(img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
            #パスを結合
            fileName = os.path.join(out_dir, str(i) + "_" + str(ang) + "thr.jpg")
            #画像を出力
            cv2.imwrite(str(fileName),img_thr)
            #----------
            #ぼかし処理
            #----------
            #カーネルサイズ(5×5)とガウス関数を指定する
            #カーネルサイズはモザイクの粗さ的なもの
            #ガウス関数はよくわからない
            img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
            #パスを結合
            fileName = os.path.join(out_dir, str(i) + "_" + str(ang) + "filter.jpg")
            #画像を出力
            cv2.imwrite(str(fileName), img_filter)
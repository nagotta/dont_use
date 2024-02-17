import os
import cv2
import glob

# flist = ["validation"]
FM_list = ["men2","women"]
# for fname in flist:
for FM_name in FM_list:

    #フォルダーを作成している
    os.makedirs("./add_testdata/okdata2/"+ FM_name, exist_ok=True)

    #読み取る画像のフォルダを指定
    in_dir = "./add_testdata/picked2/add_testdata/"+FM_name+"/*.jpg"
    #出力先のフォルダを指定
    out_dir = "./add_testdata/okdata2/"+FM_name

    #./assets_kora/drink/のフォルダの画像のディレクトリをすべて配列に格納している
    img_jpg = glob.glob(in_dir)
    #./assets_kora/drink/のファイルを一覧にする

    #画像の個数分繰り返し作業を行う
    for i in range(len(img_jpg)):
        img =cv2.imread(str(img_jpg[i]))
        #64×64サイズにしてる
        img_rot = cv2.resize(img,(64,64))
        #パスを結合している
        fileName = os.path.join(out_dir+"/"+"y"+str(i)+".jpg")
        #画像を出力
        cv2.imwrite(str(fileName),img_rot)
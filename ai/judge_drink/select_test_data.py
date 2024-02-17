###################
#テストデータを選択
###################
import shutil
import random
import glob
import os
import cv2
#飲み物リスト
drink_list = ["コカ・コーラ","綾鷹","アクエリアス"]
os.makedirs("./test", exist_ok=True)

for drink_name in drink_list:
    img_dir = "./images/" + drink_name + "/*"
    img_jpg=glob.glob(img_dir)
    img_file_name_list=os.listdir("./images/" + drink_name + "/")
    #img_file_name_listをシャッフル、そのうち2割をtest_imageディテクトリに入れる
    random.shuffle(img_jpg)
    os.makedirs('./test/' + drink_name, exist_ok=True)
    for t in range(len(img_jpg)//5):
        img_move = cv2.imread(str(img_jpg[t]))
        img_move = cv2.resize(img_move,(64,64))
        cv2.imwrite("./test/" + drink_name + "/select" + str(t) + ".jpg", img_move)

        #shutil.move(str(img_jpg[t]), "./test/" + drink_name)
import pyxel #pyxelライブラリーをインポート
from random import randint #randomからrandintライブラリーをインポート​
WINDOW_H = 120 #ウィンドウの高さを設定
WINDOW_W = 160 #ウィンドウの横幅を設定
chara_H = 16 #キャラの高さ
chara_W = 16 #キャラの横幅

class App:
  #初期設定
  def __init__(self):
    self.speed = 2 #キャラクターの移動速度を設定
    self.IMG_ID0 = 0 #imageIDを設定
    self.p_player = Player() #self.p_playerにclass Playerを代入
    self.coin = []#コインを格納する箱
    self.score = 0#点数

    pyxel.init(WINDOW_W,WINDOW_H, caption="練習") #ウィンドウを生成

    pyxel.image(self.IMG_ID0).load(0,0,"assets/cat_16x16.png")#assetsの中の画像をロードする

    pyxel.run(self.update,self.draw)#updateとdrawを実行

  #ゲーム内の処理を大体ここで行う
  def update(self):
    #Qをおしたら、終了
    if pyxel.btn(pyxel.KEY_Q):
      pyxel.quit()
    
    #キャラの動きを設定
    if pyxel.btn(pyxel.KEY_UP):
      self.p_player.player_y = self.p_player.player_y - self.speed
    if pyxel.btn(pyxel.KEY_DOWN):
      self.p_player.player_y = self.p_player.player_y + self.speed
    if pyxel.btn(pyxel.KEY_LEFT):
      self.p_player.player_x = self.p_player.player_x - self.speed
    if pyxel.btn(pyxel.KEY_RIGHT):
      self.p_player.player_x = self.p_player.player_x + self.speed

    #コインの当たり判定
    coin_atk = len(self.coin)
    for c in range (coin_atk):
      if ((self.coin[c].item_x+5 >= self.p_player.player_x) and (self.coin[c].item_x-15 <= self.p_player.player_x)
        and (self.coin[c].item_y >= self.p_player.player_y) and (self.coin[c].item_y - 20 <= self.p_player.player_y)):
        del self.coin[c]
        self.score += 15
        break
    
    #コインを出現
    if len(self.coin) >= 0 and len(self.coin)  < 6:
      new_coin = Item()
      self.coin.append(new_coin)

  def draw(self):
    pyxel.cls(0)

    #文字列を表示
    pyxel.text(1,2,"練習",9)
    #x座標　y座標　文字列　色の順に設定
  #プレイヤーの描画
    pyxel.blt(self.p_player.player_x, self.p_player.player_y, self.IMG_ID0, 0, 0, chara_W, chara_H, 5)
    #()の中は、x座標　y座標　画像　描画の開始点2つ　画像の高さ　画像の横幅　色の順に設定

    #コインの描画
    for coin in self.coin:
      pyxel.circ(coin.item_x, coin.item_y, 7, 9)
      pyxel.circ(coin.item_x, coin.item_y, 5, 8)
      pyxel.circ(coin.item_x, coin.item_y, 2, 10)

    #得点表示
    pyxel.text(1, 2, "score:" + str(self.score), 9)

class Player:
  #初期位置の設定
  def __init__(self):
    self.player_x = 20
    self.player_y = 60

  #座標を更新
  def update(self,x,y):
    self.player_x = x
    self.player_y = y

class Item:
    def __init__(self):
        self.item_x = randint(15, 145)#15~145の間の数を生成
        self.item_y = randint(15, 105)#15~145の間の数を生成

App()
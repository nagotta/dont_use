import matplotlib.pyplot as plt
import random
import math

# シミュレーションの空間領域の設定
MAX_X = 25
MIN_X = -25
MAX_Y = 25
MIN_Y = -25

# 岩場のクラス
class Rock:

    # 岩場の初期状態の設定
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1  # 左上
        self.x2 = x2  # 右下
        self.y1 = y1  # 左上
        self.y2 = y2  # 右下

    # 岩場のプロット
    def plot_rock(self):
        plt.plot([self.x1, self.x2, self.x2, self.x1, self.x1], [self.y1, self.y1, self.y2, self.y2, self.y1], 'k--')

# エージェントのクラス
class Agent:

    # エージェントの初期状態の設定
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # エージェントの次の状態を計算
    def next(self, agents):
        # 領域外に移動した場合はループする
        if self.x < MIN_X:
            self.x += (MAX_X - MIN_X)
        elif self.x > MAX_X:
            self.x -= (MAX_X - MIN_X)
        if self.y < MIN_Y:
            self.y += (MAX_Y - MIN_Y)
        elif self.y > MAX_Y:
            self.y -= (MAX_Y - MIN_Y)

    def nearest_agent(self, agents):
        min_distance = float('inf')
        min_agent = None
        for agent in agents:
            if id(self) == id(agent):
                continue
            #  カテゴリ1のエージェントとの距離を調べる
            my_x = self.x
            my_y = self.y
            you_x = agent.x
            you_y = agent.y
            distance = (my_x - you_x) ** 2 + (my_y - you_y) ** 2
            if distance < self.range and distance < min_distance:
                min_distance = distance
                min_agent = agent
        return min_agent

    # エージェントの状態を出力
    def __str__(self):
        return f'{type(self)}\tx={self.x}\ty={self.y}'    
    

class Shark(Agent):
    def __init__(self, x, y, range):
        super().__init__(x, y)
        self.range = range  # エージェントの近傍
        self.direction = random.random() * 360

    def next(self, agents):
        agent = self.nearest_agent(agents)
        if type(agent) is Shark:
            # 視野内にサメがいたとき
            # 離れる
            self.direction = math.degrees(math.atan2(self.y - agent.y, self.x - agent.x))
        elif type(agent) is Fish:
            # 視野内に魚がいたとき
            if agent.hide == False:  # 岩場に隠れていないとき
                # 近づく
                self.direction = math.degrees(math.atan2(agent.y - self.y, agent.x - self.x))
                
        # 移動
        self.x += math.cos(math.radians(self.direction)) * 0.5
        self.y += math.sin(math.radians(self.direction)) * 0.5
        super().next(agents)



class Fish(Agent):

    def __init__(self, x, y, range):
        super().__init__(x, y)
        self.range = range  # エージェントの近傍
        self.direction = random.random() * 360
        self.hide = False

    def next(self, agents):
        agent = self.nearest_agent(agents)
        if type(agent) is Shark:
            # 視野内にサメがいたとき
            # 離れる
            self.direction = math.degrees(math.atan2(self.y - agent.y, self.x - agent.x))
            self.x += math.cos(math.radians(self.direction)) * 3.0
            self.y += math.sin(math.radians(self.direction)) * 3.0
        elif self.hide == True:  # 岩場に隠れているとき
            # ゆっくり移動
            self.x += math.cos(math.radians(self.direction)) * 0.2
            self.y += math.sin(math.radians(self.direction)) * 0.2
        else:
            # 移動
            self.x += math.cos(math.radians(self.direction)) * 0.5
            self.y += math.sin(math.radians(self.direction)) * 0.5
        super().next(agents)
    
    # 岩場に隠れているかどうかの判定
    def hide_check(self, rock):
        if math.fabs(rock.x1) <= math.fabs(self.x) <= math.fabs(rock.x2) and math.fabs(rock.y1) <= math.fabs(self.y) <= math.fabs(rock.y2):
            self.hide = True
        else:
            self.hide = False


if __name__ == '__main__':
    STEP_NUM = 1000  # 時間
    FISH_NUM = 300  # エージェントの個数
    SHARK_NUM = 10  # エージェントの個数
    # カテゴリ0のエージェントのx, y座標のリスト
    xlist0 = []
    ylist0 = []
    # カテゴリ1のエージェントのx, y座標のリスト
    xlist1 = []
    ylist1 = []

    # 追加
    # 岩場の作成
    rocks = []
    rocks.append(Rock(10, 20, 10, 20))
    rocks.append(Rock(-10, -20, -10, -20))

    # エージェントの配列（Agent N個）
    agents = []
    for _ in range(FISH_NUM):
        agents.append(Fish(random.uniform(-5, 5), random.uniform(-5, 5), 10))
        for rock in rocks:
            agents[-1].hide_check(rock)
            tmp = agents[-1].hide
            if tmp == True:
                break
    for _ in range(SHARK_NUM):
        agents.append(Shark(random.uniform(-5, 5), random.uniform(-5, 5), 10))

    # 時間分ループ
    for t in range(STEP_NUM):
        plt.clf()  # グラフをクリア

        for agent in agents:
            # エージェントの次の状態を計算
            agent.next(agents)
            # エージェントのx座標とy座標を取得（描画用）
            if type(agent) == Shark:
                xlist0.append(agent.x)
                ylist0.append(agent.y)
            elif type(agent) == Fish:
                xlist1.append(agent.x)
                ylist1.append(agent.y)
                for rock in rocks:
                    agent.hide_check(rock)
                    tmp = agent.hide
                    if tmp == True:
                        break

        # シミュレーショの領域（40x40ピクセル）を設定
        plt.axis([MIN_X, MAX_X, MIN_Y, MAX_Y])
        # エージェントをグラフにプロット
        plt.plot(xlist0, ylist0, '.')
        plt.plot(xlist1, ylist1, '+')

        # 追加
        # 岩場を描画
        for rock in rocks:
            rock.plot_rock()

        # 可視化のため一時停止
        plt.pause(0.1)
        # 座標をクリア
        xlist0.clear()
        ylist0.clear()
        xlist1.clear()
        ylist1.clear()
        
        # レポート用画像出力
        if t==5 or t==30 or t==50 or t==80 or t==100 or t==1:
            plt.savefig(f'output/img{t:02}.png')

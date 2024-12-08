def str_cells(cells):
    line = ['■' if cell == 1 else '□' for cell in cells]
    return ''.join(line)

def check_3cells(_s,s,s_,rule_list):
    if _s == 1:
        if s == 1:
            if s_ == 1:
                # 111
                return rule_list[0]
            else:
                # 110
                return rule_list[1]
        else:
            if s_ == 1:
                # 101
                return rule_list[2]
            else:
                #100
                return rule_list[3]
    else:
        if s == 1:
            if s_ == 1:
                # 011
                return rule_list[4]
            else:
                # 010
                return rule_list[5]
        else:
            if s_ == 1:
                # 001
                return rule_list[6]
            else:
                #000
                return rule_list[7]
                

# セルの初期化
import random
cells = [random.randint(0, 1) for _ in range(101)]
# cells = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print(str_cells(cells))

# rule_listの作成:111,110,101,100,011,010,001,000の順番
# rule_list = [0,0,0,1,1,1,1,0]  # ルール30
# rule_list = [0,0,0,0,1,1,0,0]  # ルール12
rule_list = [0,1,1,0,1,1,1,0]  # ルール110
#rule_list = [0,1,1,1,1,1,1,0]  # ルール126

#  返り値を代入する変数 r
r = 0
for i in range(100):
    # 次の状態のセルを保存する配列を用意
    next_cells = []
    # 周期境界条件 cells[-1]は配列の最後の値
    r = check_3cells(cells[-1], cells[0], cells[1],rule_list)
    next_cell = r
    next_cells.append(next_cell)
    # 配列の1番目から配列の末尾-1番目まで
    for j in range(1, len(cells) - 1):
        # if文を使って次のセルの状態を決定する
        r = check_3cells(cells[j-1], cells[j], cells[j+1],rule_list)
        next_cell = r
        next_cells.append(next_cell)
    # 周期境界条件
    r = check_3cells(cells[-2], cells[-1], cells[0],rule_list)
    next_cell = r
    next_cells.append(next_cell)
    # セルを次の状態に置き換え
    cells = next_cells
    # 出力
    print(str_cells(next_cells))    
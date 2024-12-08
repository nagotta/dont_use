def str_cells(cells):
    line = ['■' if cell == 1 else '□' for cell in cells]
    return ''.join(line)

# セルの初期化
import random
cells = [random.randint(0, 1) for _ in range(51)]
for i in range(30):
    # 次の状態のセルを保存する配列を用意
    next_cells = []
    # 周期境界条件 cells[-1]は配列の最後の値
    cells[-1], cells[0], cells[1]
    next_cell =
    next_cells.append(next_cell)
    # 配列の1番目から配列の末尾-1番目まで
    for j in range(1, len(cells) - 1):
        # if文を使って次のセルの状態を決定する
        cells[j-1], cells[j], cells[j+1]
        next_cell =
        next_cells.append(next_cell)
    # 周期境界条件
    cells[-2], cells[-1], cells[0]
    next_cell =
    next_cells.append(next_cell)
    # セルを次の状態に置き換え
    cells = next_cells
    # 出力
    print(str_cells(next_cells))    
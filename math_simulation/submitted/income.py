# 購入パターンの定義
purchase_patterns = [
    [1, 1, 3, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 2, 1, 1, 2, 3, 1, 3], # Case1
    [2, 3, 1, 1, 3, 2, 1, 3, 2, 3, 2, 1, 3, 1, 3, 3, 2, 1, 3, 2], # Case2
    [2, 1, 2, 3, 1, 1, 3, 3, 2, 1, 1, 3, 1, 1, 2, 3, 1, 3, 3, 3], # Case3
    [2, 3, 3, 2, 1, 2, 2, 2, 1, 2, 3, 3, 3, 3, 1, 3, 2, 2, 1, 1], # Case4
    [3, 2, 1, 3, 1, 2, 3, 3, 2, 2, 2, 3, 2, 3, 3, 3, 3, 2, 1, 1] # Case5
]

def calc_charge_and_deposit(c):
    if c == 1:
        return [70, 0]
    elif c == 2:
        return [20, 0]
    elif c == 3:
        return [0, 30]
    else:
        pass # 到達しない

import matplotlib.pyplot as plt
for p_i, pattern in enumerate(purchase_patterns):
    total_charge = 500 # 釣り銭
    xs = []
    ys = []
    for i, c in enumerate(pattern):
        xs.append(i + 1)
        charge, deposit = calc_charge_and_deposit(c)
        total_charge -= charge
        total_charge += deposit
        ys.append(total_charge)
    print(ys)
    plt.plot(xs, ys, label=f'Case{p_i + 1}')
plt.legend() # 凡例表示
# fontname='MS Gothic'は日本語表示に必要
plt.xlabel('購入者', fontname='MS Gothic')
plt.xticks(range(1, 21)) # x軸ラベル
plt.ylabel('金額', fontname='MS Gothic')
plt.yticks(range(-200, 501, 100)) # y軸ラベル
plt.show()

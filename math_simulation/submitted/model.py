import random
def rand_customer():
    rand = random.random()
    if rand <= 0.40:
        return 1
    elif rand <= 0.75:
        return 2
    elif rand <= 1.00:
        return 3
    else:
        pass # 到達しない

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
stats = []
customer_num = 25
for p_i in range(0, 50):
    total_charge = 500 # 釣り銭
    xs = [0]
    ys = [total_charge]
    for i in range(customer_num):
        xs.append(i + 1)
        charge, deposit = calc_charge_and_deposit(rand_customer())
        total_charge -= charge
        total_charge += deposit
        ys.append(total_charge)
        if total_charge < 0: 
            stats.append(i + 1)
    plt.plot(xs, ys)
plt.xlabel('人数', fontname='MS Gothic') # fontnameは日本語表示に必要
plt.xticks(range(0, customer_num+1)) # x軸ラベル
plt.ylabel('残額', fontname='MS Gothic')
plt.yticks(range(-500, 601, 100)) # y軸ラベル
plt.show()

import numpy as np
print(f"Mean:{np.mean(stats)}") 
print(f"Max:{np.max(stats)}") 
print(f"Mim:{np.min(stats)}") 

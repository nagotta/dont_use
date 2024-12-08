# x <- (a*x + c) % M
# ↓  px <- x / M
# ↓
# 同じ計算をする
# py <- x / M

# def f2(x1, x2,..., x5)
# 	return exp()



# 1-1
import random
# N=100回で実施
num = 100
in_area = 0 # 円の内部に発生した点の数
for _ in range(num):
    x = random.random() # [0, 1]の乱数発生
    y = random.random() # [0, 1]の乱数発生
# ランダム点が円の内部かどうかを判定
if x**2 + y**2 <= 1:
    in_area += 1
# 4半円のため4倍して円周率を出力
print(4 * in_area / num)

# 1-2
import random
# N=100回で実施
num = 100
in_area = 0 # 円の内部に発生した点の数
for _ in range(num):
    x = random.random() # [0, 1]の乱数発生
    y = random.random() # [0, 1]の乱数発生
# ランダム点が円の内部かどうかを判定
if x**2 + y**2 <= 1:
    in_area += 1
# 4半円のため4倍して円周率を出力
print(4 * in_area / num)
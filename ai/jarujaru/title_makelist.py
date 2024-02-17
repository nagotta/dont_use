'''
このファイルと同じ階層にタイトルを一行に一つずつ入れた{title.txt}を作成しておく。
!!!!!!!!!!!形容動詞の一部(変な→変)や口語(チョイスミスった→チョイ、スミみたいになる)が名詞として抽出されがちなので見つけたら取り除いて下さい。!!!!!!!!!!!!!
'''

import MeCab
import pandas as pd
import numpy as np
import subprocess


cmd='echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path_neologd = (subprocess.Popen(cmd, stdout=subprocess.PIPE,
                           shell=True).communicate()[0]).decode('utf-8')


m=MeCab.Tagger("-Ochasen -d "+str(path_neologd))  # NEologdへのパスを追加

FILE_NAME = "title.txt"

with open(FILE_NAME, "r", encoding="utf-8") as f:
    CONTENT = f.read()

C = CONTENT.split('\n')
C.append(0)

nouns = []
title = []
nlist = ['名詞-一般','名詞-代名詞-一般','名詞-サ変接続','名詞-固有名詞-一般','名詞-形容動詞語幹']

for ccc in C:
    if ccc == 0:
        break
    parse = m.parse(ccc)
    parse = parse.split("\n")
    del parse[-1]
    for par in parse:
        par = par.replace('\t',',')
        par = par.split(',')
        del par[-1]
        if len(par) < 2:
            continue
        if par[-2] in nlist:
            nouns.append(par[0])
            title.append(ccc)

d = {"名詞": nouns, "タイトル": title}
pd.DataFrame(d)
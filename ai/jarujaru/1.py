# 形態素解析
import MeCab
import sys

wakati = MeCab.Tagger('-Ochasen')
sentence_list = ["私は猫が好きです","私は猫が嫌いです","私は犬が好きです","私は犬が嫌いです"]
sentence_wakati_list = [wakati.parse(i).split() for i in sentence_list]
print(sentence_wakati_list)
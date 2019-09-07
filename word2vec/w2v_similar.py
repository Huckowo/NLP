# -*- coding: utf-8 -*-
# @Time    : 2019/8/16 20:53
# @Author   : YeFD
# @FileName : w2v_similar.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import time
import joblib
start = time.process_time()

w2v_model = joblib.load(r'H:\Project\Python\NLP\w2v.model')
# print(w2v_model.wv.most_similar("地球"))
# [('火星', 0.86799156665802), ('月球', 0.8565017580986023), ('宇宙', 0.8545206189155579), ('银河系', 0.849134624004364),
# ('木星', 0.8398940563201904), ('太阳系', 0.8256912231445312), ('太阳', 0.8166446685791016), ('天王星', 0.8089085817337036),
# ('大气层', 0.7934627532958984), ('金星', 0.784365177154541)]
# 11.703125
print(w2v_model.wv.most_similar("爱不释手"))
# [('惟妙惟肖', 0.8125977516174316), ('神魂颠倒', 0.8083752989768982), ('会心一笑', 0.7993052005767822),
# ('赞叹不已', 0.7992339730262756), ('肃然起敬', 0.7790569067001343), ('唱出来', 0.7649531960487366),
# ('身临其境', 0.7554298639297485), ('感人至深', 0.7539796829223633), ('惊叹不已', 0.7518457174301147),
# ('深深感动', 0.7515769004821777)]
# 11.546875
end = time.process_time()
print(end - start)
# -*- coding: utf-8 -*-
# @Time    : 2019/8/16 19:40
# @Author   : YeFD
# @FileName : load_w2v-svm_10.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import joblib
import jieba
import time
from gensim.models import Word2Vec
import numpy as np
start = time.process_time()
stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]


def del_stopwords(sentence):
    result = []
    for word in sentence:
        if word in stopwords:
            continue
        else:
            result.append(word)
    return result


def cut_sentence(sentence):
    word_list = jieba.cut(sentence)
    word_list = del_stopwords(word_list)
    return word_list


w2v_model = joblib.load(r'H:\Project\Python\NLP\w2v.model')
svm_model = joblib.load(r'H:\Project\Python\NLP\svm_10.model')


data_test = ['手机很好，功能很强大，刷脸功能特别好用，配置很高，速度很快，屏幕超级清晰，拍照很好用，电池容量大，两天一冲，给老婆买的，特别喜欢，爱不释手，质感非常好，支持华为国产手机和国产芯片，性价比超高，物流很快下次还买，家里人都是用的华为手机，非常赞',
             '京东物流速度没得说，手机非常漂亮，质感一流，手感握持感大小正合适，背面特殊工艺基本看不到指纹，使用了几天感觉运行速度还可以，基本不玩游戏，日常APP运行流程无卡顿，全面屏手势非常好用，升级123版本后流畅度进一步提升，电池也很耐用，充电速度杠杠的，一般都是半小时多就充满了，屏幕显示效果只能说一般，比起三星旗舰机还是差很一些的，尤其是暗光环境下，看见是灰蒙蒙的，只能稍微调高一些亮度就好了，还有低亮度频闪严重，白天基本没问题，很清晰透彻，期待华为速度跟进DC调光，总体上来说是综合表现非常不错的一款水桶机，值得推荐！',
             '反应速度超级快的，真的挺不错。拍照很多功能都有，真的挺好。大广角。外形超好看。待机，一个晚上待机才用0.3%的电量。真的很省电的，就算你再怎么用都可以用一天。本来我都有手机的，是小米的呢。就看了华为的新闻，被美国禁止了。所以我能力没那么多，就支持一点。就买一台。支持华为。',
             '手机还可以，就是连个贴膜都不送，有多扣，不是在意这个贴膜，你直接原装出来贴好的多好，我真心贴不好呀，差评差评差评，没有贴膜，不好。剩下的还可以，没有驾驶模式，就是导航或者蓝牙免打扰的设置没有，不**全。剩下的，还可以吧，得用着试试才知道。',
             '当日下单，次日达。手机散热感觉有点渣，五千多块的暖手宝',
             '说好的支持国产，首先第一点是重量，感觉是拿块砖头一样，还有屏幕出产不带膜，然后分辨率是3120X1440的？居然跟我3年前6S的720p的一样，再来就是核心系统，反应迟钝，可能安卓硬伤吧，唯一优点就是充电快，电量百分之二十到一百，五十分钟就充满了，海军有点多啊',
             '这质量我不想多说，1个多月的手机，要检测这，检测那，闹心，华为官方售后说屏幕没问题，主板可能有问题，要换主板，6千多的机子，这情况！都说华为质量好，对华为要刮目相看了，呵呵！'
             ]


def get_w2v(word_list):  # 传入一个由n个单词组成的word_list，返回一个由n个向量组成的list，每个向量对应一个单词
    w2v_list = []
    for word in word_list:
        word = word.replace('\n', '')  # 删除换行符
        try:
            w2v_list.append(w2v_model[word])
        except KeyError:
            continue
    return np.array(w2v_list, dtype='float')


def build_w2v(sentence_list):  # 传入一个由n个句子组成的sentence_list，返回由n个array向量组成的list，每个array向量对应一个句子
    sentence_w2v_list = []
    for sentence in sentence_list:
        word_list = cut_sentence(sentence)  # 句子分词,返回一个由n个单词组成的word_list
        word_w2v_list = get_w2v(word_list)  # 返回单词向量list
        if len(word_w2v_list) != 0:
            w2v_array = sum(np.array(word_w2v_list))/len(word_w2v_list)  # 单词向量相加，将w2v_list合并为为一个句子array向量
            sentence_w2v_list.append(w2v_array)
    return sentence_w2v_list


train_data = build_w2v(data_test)

test_predict = svm_model.predict(train_data)
test_predict_2 = svm_model.predict_proba(train_data)

print(test_predict)
print(test_predict_2)
result_list = []
for pro_list in test_predict_2.tolist():
    test_predict_2_str = ','.join('%s' %i for i in pro_list)
    # print(test_predict_2_str)
    result_list.append(test_predict_2_str)
# print(result_list)
result_str = '\n'.join(result_list)
print(result_str)
end = time.process_time()
print(end - start)
'''
[1. 1. 1. 0. 0. 0. 0.]
[[9.62226066e-06 9.99990378e-01]
 [1.01431969e-02 9.89856803e-01]
 [3.11961586e-01 6.88038414e-01]
 [8.17608276e-01 1.82391724e-01]
 [9.92151071e-01 7.84892862e-03]
 [8.78435236e-01 1.21564764e-01]
 [6.90281992e-01 3.09718008e-01]]
10.8125
'''
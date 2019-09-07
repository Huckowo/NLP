# -*- coding: utf-8 -*-
# @Time    : 2019/8/16 23:32
# @Author   : YeFD
# @FileName : score_w2v-svm_test.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import joblib
import jieba
import time
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
start = time.process_time()
pos_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\pos.txt'
neg_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\neg.txt'

pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data_all = pos + neg
label = [1]*16548 + [0]*18581
train_data, test_data, train_label, test_label = train_test_split(data_all, label, test_size=0.3, random_state=2)

stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]

w2v_model = joblib.load(r'H:\Project\Python\NLP\w2v.model')
svm_model = joblib.load(r'H:\Project\Python\NLP\svm_2.model')


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
            sentence_w2v_list += [w2v_array]
    return sentence_w2v_list


def get_acc(pre, y):
    l = len(pre)
    count = 0
    for i in range(l):
        if pre[i] == y[i]:
            count += 1
        else:
            continue
    return count/l


data_X = build_w2v(data_all[3000:30000])

predict = svm_model.predict(data_X)
predict_list = predict.tolist()
# test_predict_2 = svm_model.predict_proba(data_X)

print(predict_list)
# print(train_label)
# print(len(predict_list), len(train_label))
# print(test_predict_2)
print(get_acc(predict, label[3000:30000]))  # 0.8407338769458859
end = time.process_time()
print(end - start)
'''
2:  train 0.5261936744413237 test 0.587051452439719 all 0.855132332412182
'''
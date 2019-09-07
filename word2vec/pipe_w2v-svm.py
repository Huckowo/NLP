# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 23:11
# @Author   : YeFD
# @FileName : pipe_w2v-svm.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import jieba
import time
import numpy as np
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
pos_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\pos.txt'
neg_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\neg.txt'

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data = pos + neg  # sentence list
w2v_model = joblib.load(r'H:\Project\Python\NLP\w2v.model')


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
            sentence_w2v_list.append(w2v_array)
    return sentence_w2v_list


# print("多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model, train_X, train_label, cv=10, scoring='roc_auc')))
pos_w2v = build_w2v(pos)  # 16540
neg_w2v = build_w2v(neg)  # 18561
# data_train = pos_w2v + neg_w2v  # 35101
data_all = build_w2v(data)  # 35101

data_X = np.array(data_all)
data_label = np.concatenate((np.ones(len(pos_w2v)), np.zeros(len(neg_w2v))))

train_data, test_data, train_label, test_label = train_test_split(data_X, data_label, test_size=0.3, random_state=2)
for C_num in [4, 5, 90, 100, 150, 200, 300]:
    start = time.clock()
    svm_model = SVC(C=C_num, kernel='rbf', probability=False, gamma='auto')
# C  default=1
# kernel  default='rbf' 'linear' 'poly' 'sigmoid' 'precomputed'(不能用)
# degree  default=3  ploy函数的维度
# gamma  核函数参数 default='auto' 'scale'
# coef0  核函数参数，对poly和sigmoid有用
    svm_model.fit(train_data, train_label)
    acc = svm_model.score(test_data, test_label)
    end = time.clock()
    print("# C=", C_num, ", kernel='rbf', probability=False, gamma='auto' acc=", acc, 'time=', end-start)

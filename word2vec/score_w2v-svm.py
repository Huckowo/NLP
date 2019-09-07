# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 23:31
# @Author  : YeFD
# @FileName : score_w2v-svm.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import jieba
import time
import numpy as np
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import train_test_split

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


# print("10折交叉验证得分: ", np.mean(cross_val_score(model, train_X, train_label, cv=10, scoring='roc_auc')))
pos_w2v = build_w2v(pos)  # 16540
neg_w2v = build_w2v(neg)  # 18561
# data_train = pos_w2v + neg_w2v  # 35101
data_all = build_w2v(data)  # 35101

data_X = np.array(data_all)
data_label = np.concatenate((np.ones(len(pos_w2v)), np.zeros(len(neg_w2v))))

train_data, test_data, train_label, test_label = train_test_split(data_X, data_label, test_size=0.3, random_state=2)
for C_num in [2, 5, 10, 20, 50]:
    start = time.process_time()
    svm_model = SVC(C=C_num, kernel='rbf', probability=True, gamma='auto')
# C  default=1
# kernel  default='rbf' 'linear' 'poly' 'sigmoid' 'precomputed'(不能用)
# degree  default=3  ploy函数的维度
# gamma  核函数参数 default='auto' 'scale'
# coef0  核函数参数，对poly和sigmoid有用
    svm_model.fit(train_data, train_label)
    acc = svm_model.score(test_data, test_label)
    end = time.process_time()
    print("# C=", C_num, ", kernel='rbf', probability=Ture, gamma='auto' acc=", acc, 'time=', end-start)
# C= 3, kernel='linear', probability=False, gamma='auto' acc= 0.7344981483239958
# C= 3 , kernel='poly', probability=False, gamma='auto' acc= 0.8170164276896781
# C= 3 , kernel='sigmoid', probability=False, gamma='auto' acc= 0.5808565188491122
# C= 1 , kernel='rbf', probability=False, gamma='auto' acc= 0.802772766119077
# C= 2 , kernel='rbf', probability=True, gamma='auto'  acc= 0.813028202449909 time= 318.4113843
# C= 2 , kernel='rbf', probability=False, gamma='auto' acc= 0.8130282024499098  time= 60.67622879999999
# C= 3 , kernel='rbf', probability=False, gamma='auto' acc= 0.8217643148798784
# C= 3 , kernel='rbf', probability=True, gamma='auto'  acc= 0.8217643148798784
# C= 4 , kernel='rbf', probability=False, gamma='auto' acc= 0.828886145665179 time= 69.30246899999999
# C= 5 , kernel='rbf', probability=False, gamma='auto' acc= 0.8318298357231032 time= 69.9040468
# C= 5 , kernel='rbf', probability=True, gamma='auto' acc= 0.8318298357231032 time= 335.578125
# C= 6 , kernel='rbf', probability=False, gamma='auto' acc= 0.8352483145000474
# C= 7 , kernel='rbf', probability=False, gamma='auto' acc= 0.8402810749216598
# C= 8 , kernel='rbf', probability=False, gamma='auto' acc= 0.8437945114424081
# C= 10 , kernel='rbf', probability=False, gamma='auto' acc= 0.8488272718640205
# C= 10 , kernel='rbf', probability=True, gamma='auto' acc= 0.8488272718640205 time= 464.65625
# C= 13 , kernel='rbf', probability=False, gamma='auto' acc= 0.8538600322856329
# C= 16 , kernel='rbf', probability=False, gamma='auto' acc= 0.8576583420377932
# C= 19 , kernel='rbf', probability=False, gamma='auto' acc= 0.8608869053271294
# C= 20 , kernel='rbf', probability=True, gamma='auto' acc= 0.8627860602032096 time= 622.34375
# C= 21 , kernel='rbf', probability=False, gamma='auto' acc= 0.8638305953850537
# C= 24 , kernel='rbf', probability=False, gamma='auto' acc= 0.8673440319058019
# C= 27 , kernel='rbf', probability=False, gamma='auto' acc= 0.8695280600132941
# C= 27 , kernel='rbf', probability=False, gamma='scale' acc= 0.8644003418478777
# C= 30 , kernel='rbf', probability=False, gamma='auto' acc= 0.8728515810464343
# C= 33 , kernel='rbf', probability=False, gamma='auto' acc= 0.8758902288481626
# C= 38 , kernel='rbf', probability=False, gamma='auto' acc= 0.8789288766498908
# C= 41 , kernel='rbf', probability=False, gamma='auto' acc= 0.8794036653689108
# C= 50 , kernel='rbf', probability=False, gamma='auto' acc= 0.8845313835343273
# C= 50 , kernel='rbf', probability=True, gamma='auto' acc= 0.8845313835343273 time= 1057.03125
# C= 60 , kernel='rbf', probability=False, gamma='auto' acc= 0.8883296932864875 time= 226.5007976
# C= 70 , kernel='rbf', probability=False, gamma='auto' acc= 0.8885196087740955 time= 252.8688599000000
# C= 80 , kernel='rbf', probability=False, gamma='auto' acc= 0.8907985946253917 time= 667.6503709
# C= 90 , kernel='rbf', probability=False, gamma='auto' acc= 0.8935523691957079 time= 288.83528
# C= 100 , kernel='rbf', probability=False, gamma='auto' acc= 0.8959263127908081 time= 307.3593958
# C= 150 , kernel='rbf', probability=False, gamma='auto' acc= 0.8992498338239483 time= 390.1399878
# C= 200 , kernel='rbf', probability=False, gamma='auto' acc= 0.9025733548570886 time= 446.77518280000004
# C= 300 , kernel='rbf', probability=False, gamma='auto' acc= 0.9069414110720729 time= 541.4914894000001


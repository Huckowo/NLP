# -*- coding: utf-8 -*-
# @Time    : 2019/8/16 17:35
# @Author   : YeFD
# @FileName : build-model_w2v-svm_5.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import time
import jieba
import numpy as np
from sklearn.svm import SVC
import joblib
start = time.process_time()
stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
pos_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\pos.txt'
neg_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\neg.txt'

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data = pos + neg
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


def get_w2v(word_list):
    w2v_list = []
    for word in word_list:
        word = word.replace('\n', '')
        try:
            w2v_list.append(w2v_model[word])
        except KeyError:
            continue
    return np.array(w2v_list, dtype='float')


def build_w2v(sentence_list):
    sentence_w2v_list = []
    for sentence in sentence_list:
        word_list = cut_sentence(sentence)
        word_w2v_list = get_w2v(word_list)
        if len(word_w2v_list) != 0:
            w2v_array = sum(np.array(word_w2v_list))/len(word_w2v_list)
            sentence_w2v_list.append(w2v_array)
    return sentence_w2v_list


pos_w2v = build_w2v(pos)
neg_w2v = build_w2v(neg)
train_data = build_w2v(data)
train_label = np.concatenate((np.ones(len(pos_w2v)), np.zeros(len(neg_w2v))))
train_X = np.array(train_data)

svm_model = SVC(C=5, kernel='rbf', probability=True, gamma='auto')
svm_model.fit(train_X, train_label)


def save_mod():
    joblib.dump(svm_model, 'svm_5.model')


save_mod()
end = time.process_time()
print('time=', end - start)
# svm_5.model time= 969.53125

# -*- coding: utf-8 -*-
# @Time    : 2019/8/16 19:01
# @Author   : YeFD
# @FileName : build-model_w2v-svm_2.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import time
import jieba
from gensim.models import Word2Vec
import numpy as np
from sklearn.svm import SVC
import joblib
from sklearn.model_selection import cross_val_score
start = time.process_time()
stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
pos_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\pos.txt'
neg_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\neg.txt'

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
data = pos + neg  # sentence list
w2v_model = joblib.load(r'H:\Project\Python\NLP\w2v.model')
# label = [1]*16548 + [0]*18581
# print(stopwords)


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


# pos_cut = cut_words(pos)
# neg_cut = cut_words(neg)
# data_all = pos_cut + neg_cut
# print(len(pos_cut))
'''
# pos_del = del_stopwords(pos_cut)
# neg_del = del_stopwords(neg_cut)
pos_txt = ' '.join(pos_cut)
with open(r'H:\比赛\一般课题\资料\数据\sentiment正负\pos_cut.txt', 'w', encoding="utf-8") as file_pos:
    file_pos.write(pos_txt)
'''
# w2v_model = Word2Vec.load(r'F:\BaiduNetdiskDownload\10G训练好的词向量\60维\Word60.model')


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
            sentence_w2v_list.append(w2v_array)  # 每个句子为一个向量
    return sentence_w2v_list


pos_w2v = build_w2v(pos)  # 16540 list[array([])]
neg_w2v = build_w2v(neg)  # 18561
# data_train = pos_w2v + neg_w2v  # 35101
train_data = build_w2v(data)  # 35101

# print(len(pos_w2v))  # 16540
# print(len(neg_w2v))  # 18561
# print(len(data_train))  # 35101

train_label = np.concatenate((np.ones(len(pos_w2v)), np.zeros(len(neg_w2v))))

train_X = np.array(train_data)

svm_model = SVC(C=50, kernel='rbf', probability=True, gamma='auto')
svm_model.fit(train_X, train_label)


def save_mod():
    # joblib.dump(w2v_model, 'w2v.model')
    joblib.dump(svm_model, 'svm_50.model')


save_mod()
end = time.process_time()

print('time=', end - start)

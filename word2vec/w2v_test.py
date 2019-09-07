# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 19:10
# @Author   : YeFD
# @FileName : w2v_test.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
from gensim.models import Word2Vec
import jieba
import numpy as np
import joblib
stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
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


data_test = ['手机很好，功能很强大，刷脸功能特别好用，配置很高，速度很快，屏幕超级清晰，拍照很好用，电池容量大，两天一冲，'
             '给老婆买的，特别喜欢，爱不释手，质感非常好，支持华为国产手机和国产芯片，性价比超高，物流很快下次还买，家里人都是用的华为手机，非常赞']
sentence_test = '手机很好，功能很强大，刷脸功能特别好用，配置很高，速度很快，屏幕超级清晰，拍照很好用，电池容量大，两天一冲，给老婆买的，' \
       '特别喜欢，爱不释手，质感非常好，支持华为国产手机和国产芯片，性价比超高，物流很快下次还买，家里人都是用的华为手机，非常赞'
# print(mod.wv.most_similar("地球"))

word_list_t = cut_sentence(sentence_test)
# print(word_list)
w2v_list_t = get_w2v(word_list_t)  # 由单词向量组成的list
# print(len(w2v_list_t))  # 45
# print(w2v_list_t)
sentence_list = []
sentence_w2v_array = np.zeros(60).reshape((1, 60))  # 一维数组
for word in w2v_list_t:
    sentence_w2v_array += word
sentence_w2v_array /= len(w2v_list_t)
# sentence_w2v_array = sum(np.array(w2v_list_t))/len(w2v_list_t)  # 句子向量list
print(sentence_w2v_array)
# sentence_list.append(sentence_w2v_array)
# print(sentence_list)
# w2v = build_w2v(word_list)  # [13*array]
# w2v = build_w2v(data_test)  # [1*array]
# print(w2v)

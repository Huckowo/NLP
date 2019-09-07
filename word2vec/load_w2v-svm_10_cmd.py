# -*- coding: utf-8 -*-
# @Time    : 2019/8/18 11:40
# @Author   : YeFD
# @FileName : load_w2v-svm_10_cmd.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz

import time
import jieba
import numpy as np
import joblib


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


def build_w2v(sent_list):
    sent_w2v_list = []
    for sent in sent_list:
        word_list = cut_sentence(sent)
        word_w2v_list = get_w2v(word_list)
        if len(word_w2v_list) != 0:
            w2v_array = sum(np.array(word_w2v_list))/len(word_w2v_list)
            sent_w2v_list.append(w2v_array)
    return sent_w2v_list


if __name__ == '__main__':
    stopwords_path = r'H:\Project\Python\NLP\stopword2.txt'
    stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
    time1 = time.process_time()
    print('load w2v_model')
    w2v_model = joblib.load(r'H:\Project\Python\NLP\w2v.model')
    time2 = time.process_time()
    print('used time:', time2 - time1)
    print('load svm_model')
    svm_model = joblib.load(r'H:\Project\Python\NLP\svm_10.model')
    time3 = time.process_time()
    print('used time:', time3 - time2)

    sentence_list = []
    sentence = input('input a sentence:')
    sentence_list.append(sentence)
    sentence_w2v = build_w2v(sentence_list)
    predict_pro = svm_model.predict_proba(sentence_w2v)
    time4 = time.process_time()
    print(predict_pro, 'used time:', time4 - time3)
    predict = svm_model.predict(sentence_w2v)
    time5 = time.process_time()
    print(predict, 'used time:', time5 - time4)
    print('total time:', time5 - time1)

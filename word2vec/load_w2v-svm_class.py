# -*- coding: utf-8 -*-
# @Time    : 2019/8/17 18:09
# @Author   : YeFD
# @FileName : load_w2v-svm_class.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import jieba
import joblib
import numpy as np
# stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
# stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]


class Get(object):
    def __init__(self):
        self.w2v_model = joblib.load(r'H:\Project\Python\NLP\w2v.model')
        self.svm_model = joblib.load(r'H:\Project\Python\NLP\svm_10.model')
        self.stopwords = [line.strip() for line in open(r'H:\比赛\一般课题\资料\数据\stopword2.txt', 'r', encoding='utf-8').readlines()]

    def _get_w2v(self, word_list):
        w2v_list = []
        for word in word_list:
            word = word.replace('\n', '')
            try:
                w2v_list.append(self.w2v_model[word])
            except KeyError:
                continue
        return np.array(w2v_list, dtype='float')

    def _del_stopwords(self, sentence):
        result = []
        for word in sentence:
            if word in self.stopwords:
                continue
            else:
                result.append(word)
        return result

    def cut_sentence(self, sentence):
        word_list = jieba.cut(sentence)
        word_list = self._del_stopwords(word_list)
        return word_list

    def build_w2v(self, sentence_list):  # 传入一个由n个句子组成的sentence_list，返回由n个array向量组成的list，每个array向量对应一个句子
        sentence_w2v_list = []
        for sentence in sentence_list:
            word_list = self.cut_sentence(sentence)  # 句子分词,返回一个由n个单词组成的word_list
            word_w2v_list = self._get_w2v(word_list)  # 返回单词向量list
            if len(word_w2v_list) != 0:
                w2v_array = sum(np.array(word_w2v_list)) / len(word_w2v_list)  # 单词向量相加，将w2v_list合并为为一个句子array向量
                sentence_w2v_list.append(w2v_array)
        return sentence_w2v_list

    def get_pro(self, sentence_list):
        sentence_w2v = self.build_w2v(sentence_list)
        predict = self.svm_model.predict_proba(sentence_w2v)
        return predict.tolist()

    def get_result(self, sentence_list):
        sentence_w2v = self.build_w2v(sentence_list)
        predict = self.svm_model.predict(sentence_w2v)
        return predict.tolist()

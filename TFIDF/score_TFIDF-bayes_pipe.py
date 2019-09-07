# -*- coding: utf-8 -*-
# @Time    : 2019/8/16 0:00
# @Author   : YeFD
# @FileName : score_TFIDF-bayes_pipe.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import time
import jieba
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

start = time.process_time()
stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
pos_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\pos.txt'
neg_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\neg.txt'

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]
# print(stopwords)
# print(len(neg_train))


def del_stopwords(sentence):
    result = []
    for word in sentence:
        if word in stopwords:
            continue
        else:
            result.append(word)
    return result


def cut_words(sentence_list):
    result = []
    for sentence in sentence_list:
        word_list = jieba.cut(sentence)  # 分词-精确切分 cut_all=False
        word_list = del_stopwords(word_list)
        sentence_cut = ' '.join(word_list)
        result.append(sentence_cut)
    return result


pos_cut = cut_words(pos)
neg_cut = cut_words(neg)
# print(len(pos_cut))
'''
# pos_del = del_stopwords(pos_cut)
# neg_del = del_stopwords(neg_cut)
pos_txt = ' '.join(pos_cut)
with open('H:\\比赛\\一般课题\\资料\\数据\\sentiment正负\\pos_cut.txt', 'w', encoding="utf-8") as file_pos:
    file_pos.write(pos_txt)
'''

TFIDF_model = TfidfVectorizer(min_df=2, max_features=None, strip_accents='unicode', analyzer='word',
                              token_pattern=r'\w{1,}',
                              ngram_range=(1, 4),
                              use_idf=1,
                              smooth_idf=1,
                              sublinear_tf=1,
                              stop_words=None)

label = [1] * 16548 + [0] * 18581
data_all = pos_cut + neg_cut  # 16548 18581
len_pos = len(pos_cut)

# print(len(data_all))  # 35129
# print(len(label))  # 35129

train_data, test_data, train_label, test_label = train_test_split(data_all, label, test_size=0.3, random_state=2)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)  # 默认
model = MultinomialNB()
pipe = make_pipeline(TFIDF_model, model)
# pipe.steps
pipe.fit(train_data, train_label)
acc = pipe.score(test_data, test_label)
end = time.process_time()
print(acc, 'time=', end - start)
# 0.8974285985387608 time= 32.078125

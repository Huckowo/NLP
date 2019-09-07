# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 23:22
# @Author   : YeFD
# @FileName : build-model_TFIDF-bayes.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import jieba
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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

TFIDF_model.fit(data_all)
data_train = TFIDF_model.transform(data_all)
# train_data, test_data, train_label, test_label = train_test_split(data_train, label, test_size=0.3, random_state=2)


MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)  # 默认
model = MultinomialNB()
model.fit(data_train, label)


# model.fit(train_data, train_label)
# acc = model.score(test_data, test_label)
# print(acc)


def save_mod():
    joblib.dump(TFIDF_model, 'TFIDF.model')
    joblib.dump(model, 'bayes.model')


save_mod()

print("10折交叉验证得分: ", np.mean(cross_val_score(model, data_train, label, cv=10, scoring='roc_auc')))


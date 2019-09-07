# -*- coding: utf-8 -*-
# @Time    : 2019/8/14 23:30
# @Author   : YeFD
# @FileName : score_TFIDF-bayes.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
import jieba
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

stopwords_path = r'H:\比赛\一般课题\资料\数据\stopword2.txt'
pos_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\pos.txt'
neg_path = r'H:\比赛\一般课题\资料\数据\sentiment正负\neg.txt'
TFIDF_model = joblib.load(r'H:\Project\Python\NLP\TFIDF.model')

stopwords = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]
pos = [line.strip() for line in open(pos_path, 'r', encoding='utf-8').readlines()]
neg = [line.strip() for line in open(neg_path, 'r', encoding='utf-8').readlines()]


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

label = [1]*16548 + [0]*18581
data_all = pos_cut + neg_cut  # 16548 18581
data_train = TFIDF_model.transform(data_all)
train_data, test_data, train_label, test_label = train_test_split(data_train, label, test_size=0.3, random_state=2)

MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)  # 默认
model = MultinomialNB()

model.fit(train_data, train_label)
acc = model.score(test_data, test_label)
print(acc)  # 0.9346237783470918

# -*- coding: utf-8 -*-
# @Time    : 2019/8/18 18:49
# @Author   : YeFD
# @FileName : build-model_w2v.py
# @Software : PyCharm 2019.1.3
# @Processor: Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
from gensim.models import Word2Vec
import joblib
w2v_model = Word2Vec.load(r'F:\BaiduNetdiskDownload\10G训练好的词向量\60维\Word60.model')
joblib.dump(w2v_model, 'w2v.model')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 19:25:10 2023

@author: apple
"""

import pandas as pd
import numpy as np
import re
import sys
import jieba.analyse
import jieba.posseg as psg
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder


reviews = pd.read_csv('data/Test_reviews.csv')

#评论去重

reviews[['Reviews']].duplicated().sum()
reviews = reviews[['Reviews']].drop_duplicates()

#print(reviews)
reviews.reset_index(drop=True,inplace=True)

#数据清洗
content = reviews['Reviews']
# 编译匹配模式
pattern = re.compile('[a-zA-Z0-9]')
# re.sub用于替换字符串中的匹配项
content = content.apply(lambda x : pattern.sub('',x))


#评论分词
# 自定义简单的分词函数
worker = lambda s : [[x.word,x.flag] for x in psg.cut(s)]   # 单词与词性
seg_word = content.apply(worker)
#print(seg_word)

# 将词语转化为数据框形式，一列是词，一列是词语所在的句子id，最后一列是词语在该句子中的位置
 # 每一评论中词的个数
n_word = seg_word.apply(lambda x: len(x)) 
# 构造词语所在的句子id
n_content = [[x+1]*y for x,y in zip(list(seg_word.index), list(n_word))]
# 将嵌套的列表展开，作为词所在评论的id
index_content = sum(n_content, [])    # []指定相加的参数

seg_word = sum(seg_word,[])
# 词
word = [x[0] for x in seg_word]
# 词性
nature = [x[1] for x in seg_word]
# 构造数据框
result_review = pd.DataFrame({'index_content': index_content,
                      'word' : word,
                      'nature': nature})
# 删除标点符号
result_review = result_review[result_review['nature'] != 'x']

# 删除停用词
# 加载停用词
stop_path = open('data/stoplist.txt','r',encoding='utf-8')
stop = [x.replace('\n','') for x in stop_path.readlines()]
# 得到非停用词序列
word = list(set(word) - set(stop))
# 判断表格中的单词列是否在非停用词列中
result_review = result_review[result_review['word'].isin(word)]


# 构造各词在评论中的位置列
n_word = list(result_review.groupby(by=['index_content'])['index_content'].count())
index_word = [list(np.arange(0,x)) for x in n_word]
index_word = sum(index_word,[])
result_review['index_word'] = index_word
result_review.reset_index(drop=True,inplace=True)

#LDA分析 确认评论中体现出的商品主题
n_features = 1000 #提取1000个特征词语
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(word)

n_topics = 7 # 手动指定分类数
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                doc_topic_prior=0.1,
                                topic_word_prior=0.01,
                               random_state=0)
lda.fit(tf)

n_top_words = 25
tf_feature_names = tf_vectorizer.get_feature_names()

#输出LDA分析结果，方便进行后续对评论的数据挖掘
def print_topic_words(model, feature_names, n_top_words):
    tword = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        tword.append(topic_w)
        print(topic_w)
    return tword

topic_word = print_topic_words(lda, tf_feature_names, n_top_words)

#计算困惑度，确定最佳聚类数量
plexs = []
scores = []
n_max_topics = 10
for i in range(1,n_max_topics):
    lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50,random_state=0)
    lda.fit(tf)
    plexs.append(lda.perplexity(tf))
    scores.append(lda.score(tf))

n_t=9                                      
x=list(range(1,n_t))
plt.plot(x,plexs[1:n_t])
plt.xlabel("number of topics")
plt.ylabel("perplexity")

x=list(range(1,n_t))
plt.plot(x,scores[1:n_t])
plt.xlabel("number of topics")
plt.ylabel("score")

# tf-idf获取商品评论的top10关键词，关键词的数量选取可参考上述困惑度计算结果
def getKeywords_tfidf(data,stopkey,topK):
    idList, reviewList = data['id'], data['Reviews']
    corpus = [] # 将所有评论输出到一个list中
    for index in range(len(idList)):
        l = []
        pos = ['n', 'nz', 'v', 'vd', 'vn', 'l', 'a', 'd']  # 定义选取的词性
        seg = jieba.posseg.cut(data)  # 分词
        for i in seg:
            if i.word not in stopkey and i.flag in pos:  # 去停用词 + 词性筛选
                l.append(i.word)
        text = l # 文本预处理
        text = " ".join(text) # 连接成字符串，空格分隔
        corpus.append(text)

    # 1、构建词频矩阵，将文本中的词语转换成词频矩阵
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus) # 词频矩阵,a[i][j]:表示j词在第i个文本中的词频
    # 2、统计每个词的tf-idf权值
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    # 3、获取词袋模型中的关键词
    word = vectorizer.get_feature_names_out()
    # 4、获取tf-idf矩阵，a[i][j]表示j词在i篇文本中的tf-idf权重
    weight = tfidf.toarray()
    # 5、打印词语权重
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        ids.append(idList[i])
        df_word,df_weight = [],[] # 当前文章的所有词汇列表、词汇对应权重列表
        for j in range(len(word)):
            print (word[j],weight[i][j])
            df_word.append(word[j])
            df_weight.append(weight[i][j])
        df_word = pd.DataFrame(df_word,columns=['word'])
        df_weight = pd.DataFrame(df_weight,columns=['weight'])
        word_weight = pd.concat([df_word, df_weight], axis=1) # 拼接词汇列表和权重列表
        word_weight = word_weight.sort_values(by="weight",ascending = False) # 按照权重值降序排列
        keyword = np.array(word_weight['word']) # 选择词汇列并转成数组格式
        word_split = [keyword[x] for x in range(0,topK)] # 抽取前topK个词汇作为关键词
        word_split = " ".join(word_split)
        #keys.append(word_split.encode("utf-8"))
        keys.append(word_split)

    result = pd.DataFrame({"id": ids, "title": titles, "key": keys},columns=['id','title','key'])
    return result

# 按照不同的词性分别进行聚类
word_nouns = []
word_verbs = []
word_adjectives = []
word_adverb = []

# 停用词表
stopkey = stop

#得到各词性的聚类结果，基于这些结果和其互联信息可挖掘出消费者最高频的消费场景
nouns = result_review[result_review['nature'] == 'n']
nouns = getKeywords_tfidf(nouns, stop, 10)
word_nouns.append(nouns)

verbs = result_review[result_review['nature'] == 'v']
verbs = getKeywords_tfidf(verbs, stop, 10)
word_verbs.append(verbs)

adjectives = result_review[result_review['nature'] == 'a']
adjectives = getKeywords_tfidf(adjectives, stop, 10)
word_adjectives.append(adjectives)

adverb = result_review[result_review['nature'] == 'd']
adverb = getKeywords_tfidf(adverb, stop, 10)
word_adverb.append(adverb)

# 计算词语之间的互信息
bigram_measures = BigramAssocMeasures()
trigram_measures = TrigramAssocMeasures()

# 关键词之间的互信息PMI计算

word_all = []
word_all.extend(word_nouns, word_verbs, word_adjectives, word_adverb)

finder = BigramCollocationFinder.from_words(word_all)
finder.apply_freq_filter(2)
for bigram, score in finder.score_ngrams(bigram_measures.pmi):
    print(bigram[0], bigram[1], score)

finder = TrigramCollocationFinder.from_words(word_all)
finder.apply_freq_filter(2)
for trigram, score in finder.score_ngrams(trigram_measures.pmi):
    print(trigram[0], trigram[1], trigram[2], score)
    
#也可以自行设计PMI计算

class PMI:
    def __init__(self, document):
        self.document = document
        self.pmi = {}
        self.miniprobability = float(1.0) / document.__len__()
        self.minitogether = float(0)/ document.__len__()
        self.set_word = self.getset_word()

    def calcularprobability(self, document, wordlist):

        """
        :param document:
        :param wordlist:
        :function : 计算单词的document frequency
        :return: document frequency
        """

        total = document.__len__()
        number = 0
        for doc in document:
            if set(wordlist).issubset(doc):
                number += 1
        percent = float(number)/total
        return percent

    def togetherprobablity(self, document, wordlist1, wordlist2):

        """
        :param document:
        :param wordlist1:
        :param wordlist2:
        :function: 计算单词的共现概率
        :return:共现概率
        """

        joinwordlist = wordlist1 + wordlist2
        percent = self.calcularprobability(document, joinwordlist)
        return percent

    def getset_word(self):

        """
        :function: 得到document中的词语词典
        :return: 词语词典
        """
        list_word = []
        for doc in self.document:
            list_word = list_word + list(doc)
        set_word = []
        for w in list_word:
            if set_word.count(w) == 0:
                set_word.append(w)
        return set_word

    def get_dict_frq_word(self):

        """
        :function: 对词典进行剪枝,剪去出现频率较少的单词
        :return: 剪枝后的词典
        """
        dict_frq_word = {}
        for i in range(0, self.set_word.__len__(), 1):
            list_word=[]
            list_word.append(self.set_word[i])
            probability = self.calcularprobability(self.document, list_word)
            if probability > self.miniprobability:
                dict_frq_word[self.set_word[i]] = probability
        return dict_frq_word

    def calculate_nmi(self, joinpercent, wordpercent1, wordpercent2):
        """
        function: 计算词语共现的nmi值
        :param joinpercent:
        :param wordpercent1:
        :param wordpercent2:
        :return:nmi
        """
        return (joinpercent)/(wordpercent1*wordpercent2)

    def get_pmi(self):
        """
        function:返回符合阈值的pmi列表
        :return:pmi列表
        """
        dict_pmi = {}
        dict_frq_word = self.get_dict_frq_word()
        print(dict_frq_word)
        for word1 in dict_frq_word:
            wordpercent1 = dict_frq_word[word1]
            for word2 in dict_frq_word:
                if word1 == word2:
                    continue
                wordpercent2 = dict_frq_word[word2]
                list_together=[]
                list_together.append(word1)
                list_together.append(word2)
                together_probability = self.calcularprobability(self.document, list_together)
                if together_probability > self.minitogether:
                    string = word1 + ',' + word2
                    dict_pmi[string] = self.calculate_nmi(together_probability, wordpercent1, wordpercent2)
        return dict_pmi

















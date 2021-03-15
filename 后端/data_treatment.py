import nltk
import random
import collections
import json
import cleantext

from nltk.corpus import *
from nltk import *
from types import *


class data_treatment:
    # 这是一个处理输入数据的类


    def __init__(self,text):
        '''
        :param text: 从前端输入的文本
        self.text:原始文本
        self.sentences:将文本分句
        '''
        self.text = text
        self.sentences = nltk.sent_tokenize(text)



    @staticmethod
    def pretreatment(s_source):

        # 分词
        if(type(s_source) is str):
            tokenization = nltk.word_tokenize(s_source)
        #print("tokenization is : " + str(tokenization))
        else:
            tokenization = s_source #如果这个s_source是一个list
        # 标准化

        # 去除标点符号
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义标点符号列表
        cutwords1 = [word for word in tokenization if word not in interpunctuations]
        # 第二遍去除词内标点符号
        s_source = ""
        interpunctuationsInword = r'[,.:;()?&!@#$%\'/]'
        for i in range(0, len(cutwords1)):
            cutwords1[i] = re.sub(interpunctuationsInword, " ", cutwords1[i])
            s_source = s_source + " " + cutwords1[i]
        cutwords1 = nltk.word_tokenize(s_source)

        # 大小写转换
        cutwords2 = []
        for i in range(0, len(cutwords1)):
            cutwords2.append(cutwords1[i].lower())

        # 去停用词

        stops = set(stopwords.words("english"))
        cutwords3 = [word for word in cutwords2 if word not in stops]

        normalization = cutwords3
        #print("normalization is :" + str(normalization))

        stemming1 = []
        for i in range(0, len(normalization)):
            stemming1.append(PorterStemmer().stem(normalization[i]))
        #print("Porter stemming1 is: " + str(stemming1))
        return stemming1

    @staticmethod
    def pretrement_for_synonyms(s_source):

        if (type(s_source) is str):
            tokenization = nltk.word_tokenize(s_source)
            # print("tokenization is : " + str(tokenization))
        else:
            tokenization = s_source  # 如果这个s_source是一个list

        # 标准化
        # 去除标点符号
        interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '"',
                             '``']  # 定义标点符号列表
        cutwords1 = [word for word in tokenization if word not in interpunctuations]
        # 第二遍去除词内标点符号
        s_source = ""
        interpunctuationsInword = r'[,.:;()?&!@#$%\'/]'
        for i in range(0, len(cutwords1)):
            cutwords1[i] = re.sub(interpunctuationsInword, " ", cutwords1[i])
            s_source = s_source + " " + cutwords1[i]
        cutwords1 = nltk.word_tokenize(s_source)

        # 大小写转换
        cutwords2 = []
        for i in range(0, len(cutwords1)):
            cutwords2.append(cutwords1[i].lower())
        return cutwords2
    @staticmethod
    def pos_lemmatize_article(article):
        lemmatizer = WordNetLemmatizer()
        sent_token =nltk.sent_tokenize(article)
        for k in range(len(sent_token)):
            sent = sent_token[k]
            word_token = nltk.word_tokenize(sent)
            pos_tag_list = nltk.pos_tag(word_token)
            # print(pos_tag_list)
            new_sent = ""
            for i in range(0,len(pos_tag_list)):
                tag = pos_tag_list[i][1]
                word = word_token[i]
                if tag.startswith('NN'):
                    word = lemmatizer.lemmatize(word, pos='n')
                elif tag.startswith('VB'):
                    if tag.startswith('VBN'):pass
                    else:
                        word = lemmatizer.lemmatize(word, pos='v')
                elif tag.startswith('JJ'):
                    word = lemmatizer.lemmatize(word, pos='a')
                elif tag.startswith('R'):
                    word = lemmatizer.lemmatize(word, pos='r')
                else:
                    word = lemmatizer.lemmatize(word)
                word_token[i] = word
                new_sent+=word
                new_sent+= " "
            new_sent = new_sent[0:len(new_sent)-1]
            sent_token[k]=new_sent
        new_article = ""
        for i in range(len(sent_token)):
            new_article+=sent_token[i]
            new_article+=" "
        return new_article

    @ staticmethod
    def pretrement_extra_space(article):
        pattern = re.compile(r" ")
        article = cleantext.clean(article)
        interpunctuations = ['.', ',', ';', '\?', ':','!']# 定义标点符号列表
        for i in range(len(interpunctuations)):
            pattern_1 = re.compile(interpunctuations[i]+"  ")
            pattern_2 = re.compile(" "+interpunctuations[i]+" ")
            pattern_3 = re.compile(" "+interpunctuations[i]+"  ")
            article = re.sub(pattern_1,interpunctuations[i]+" ",article)
            article = re.sub(pattern_2, interpunctuations[i] + " ", article)
            article = re.sub(pattern_3, interpunctuations[i] + " ", article)
        return article








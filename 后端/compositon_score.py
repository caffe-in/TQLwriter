from nltk import *
from json import *
from nltk.corpus import *
from nltk.stem import WordNetLemmatizer
from data_treatment import data_treatment
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import *
from sklearn.externals import joblib
from gensim import corpora,models

import numpy as np
import language_tool_python
import re
import cleantext
import heapq
import nltk
import sys
import xlrd
import xlwt




class CompositionScroe:
    """
    这个类用于作文评分：一共有以下x方面

    - 词汇方面
        - 文本总数
        - 错误单词所占比例
        - 词汇总量
    - 语句方面
        - 标点符号
        - 平均词汇数目
        - 平均短语数目
        - 平均句子语法错误数目
    - 内容质量评价：使用TF-IDF抽取主题词

    """

    phrase_array1 = []
    phrase_array2 = []
    phrase_array3 = []      # 短语抽取的语料库
    phrase_file = open("data/score/phrases.txt")
    phrase_array1 = phrase_file.readlines()
    for i in range(len(phrase_array1) - 1):
        phrase_array1[i] = phrase_array1[i][:len(phrase_array1[i]) - 1]
    article_topic = {
        "love":["like","hate"],
        "education":["teacher","student"],
        "economy":["money","people"]
    }
    tool = language_tool_python.LanguageTool('en-US')

    def __init__(self,compositon_atticle,topic):
        self._composition_article = compositon_atticle   # compostion_article is the composition's content
        self._composition_sentences = nltk.sent_tokenize(self._composition_article) # composition sentences is all the sent in article
        self._clean_text = self.get_clean_text()
        #self._lemmatize_text = data_treatment.pos_lemmatize_article(cleantext.clean(self._composition_article))

        self.language_tool = CompositionScroe.tool

        # 词汇方面
        self._spell_mistake_num = 0
        self._composition_num = 0
        self._composition_total_voc = 0
        self._comma_num = 0
        self._question_mark_num = 0
        self._period_mark_num = 0
        self._stand_deviation_word_len = 0



        # 语句方面
        self._grammar_mistake_num = 0
        self._mistake_message =[]
        self._avg_word_num = 0
        self._avg_voc_num = 0
        self._avg_pharse_nun = 0
        self._avg_grammar_mistake_num = 0
        self._composition_sent_num = 0
        self._preposition_num = 0
        self._conjunction_num = 0
        self._stand_deviation_sent_num = 0

        #篇章方面
        self._paragraph_num = 0  #段落数目
        self._topic_word = []
        #使用lda
        # ldafile = open('./model/ldamodel.pkl','rb')
        # ldadict = open('./model/ldadict.pkl','rb')
        # self.ldamodel = joblib.load(ldafile)
        # self.ldadict = joblib.load(ldadict)
        # self.ldaans = 0

        #文章分数，初始分数为10

        # self.score = 10

        self.topic = topic




    def get_spell_mistake(self,sentence):
        '''

        :param sentence:作文中的一句话
        '''

        matches = self.language_tool.check(sentence)
        for match in matches:
            if match.__getattribute__("ruleIssueType") == "misspelling":
                self._spell_mistake_num+=1
                # print(match)



    def grammar_check(self,sentence):
        """

        :param
        sentence:需要进行语法检测的句子
        tool:grammar-check的工具包
        :return:
        """

        matches = self.language_tool.check(sentence)  # time slow
        """
        matches的信息如下
        offset:错误开始的偏移位置
        lenth：错误的长高度
        RULE ID:错误种类
        suggestion：修改方案
        """


        self._mistake_message.append(matches)
        ans = []
        for match in matches:
            if match.__getattribute__("ruleIssueType") =="typographical":
                continue
            else:
                ans.append(match)
        self._grammar_mistake_num += len(ans)


    def get_clean_text(self):
        clean_text = cleantext.clean(self._composition_article)
        clean_text = data_treatment.pretreatment(clean_text)
        return clean_text

    def get_word_param(self):
        for sent in self._composition_sentences:
            self.get_spell_mistake(sent)
        self._composition_num=len(nltk.word_tokenize(self._composition_article))
        self._composition_total_voc = len(set(self._clean_text))
        word_len_list = []
        for word in nltk.word_tokenize(self._composition_article):
            word_len_list.append(len(word))
            if word =='?':self._question_mark_num+=1
            elif word ==',':self._comma_num+=1
            elif word=='.':self._period_mark_num+=1
        self._stand_deviation_word_len = np.std(np.array(word_len_list),ddof=1)

    def get_sent_param(self):
        # time slow
        sent_len_list = []
        for sent in self._composition_sentences:
            sent_len_list.append(len(sent))
            self.grammar_check(sent)
            pos_tag_list = pos_tag(nltk.word_tokenize(sent))
            for pos in pos_tag_list:
                if pos[1].startswith('CC'):self._conjunction_num+=1
                elif pos[1].startswith('IN'):self._preposition_num+=1




        self._grammar_mistake_num -= self._spell_mistake_num
        sent_num = len(self._composition_sentences)
        self._avg_word_num = self._composition_num/sent_num
        self._avg_voc_num = self._composition_total_voc/sent_num
        self._avg_grammar_mistake_num = self._grammar_mistake_num/sent_num
        self._composition_sent_num = sent_num
        self._stand_deviation_sent_num = np.std(np.array(sent_len_list),ddof=1)

        # regular_patten_str = ""
        # for word in CompositionScroe.phrase_array1:
        #     regular_patten_str += " "
        #     regular_patten_str += word
        #     regular_patten_str += " "
        #     regular_patten_str += "|"
        # for word in CompositionScroe.phrase_array2:
        #     regular_patten_str += " "
        #     regular_patten_str += word
        #     regular_patten_str += " "
        #     regular_patten_str += "|"
        # for word in CompositionScroe.phrase_array2:
        #     regular_patten_str += " "
        #     regular_patten_str += word
        #     regular_patten_str += " "
        #     regular_patten_str += "|"
        # regular_patten_str = regular_patten_str[0:len(regular_patten_str) - 1]
        # regular_patten  = re.compile(regular_patten_str)
        # find_ans = set(regular_patten.findall(self._lemmatize_text))
        # print("this is find_ans",find_ans)
        # self._avg_pharse_nun = len(find_ans)/sent_num
        # self._avg_grammar_mistake_num=self._grammar_mistake_num/sent_num
        # self._composition_sent_len=sent_num

    def get_article_param(self):
        self._paragraph_num = self._composition_article.count('\t')
        # 使用路透社前1000篇的文章作为tf-idf的语料库
        # reuters_articles = reuters.fileids()
        # tf_idf_corpus = []
        # tf_idf_corpus.append(self._composition_article)
        # for i in range(0,1000):
        #     temp_list = reuters.words(reuters_articles[i])
        #     temp_str = ""
        #     for j in range(0,len(temp_list)):
        #         temp_str+=temp_list[j]+" "
        #     tf_idf_corpus.append(cleantext.clean(temp_str))
        # # 计算当前文章的tf-idf
        # vectorizer = CountVectorizer(stop_words = ["the"])
        # transforms = TfidfTransformer()
        # tfidf = transforms.fit_transform(vectorizer.fit_transform(tf_idf_corpus))
        # tfidf_array = tfidf.toarray()
        # word = vectorizer.get_feature_names()
        # compostion_tfidf =list(zip(word,tfidf_array[0]))
        # temp = heapq.nlargest(5,compostion_tfidf,key=lambda s:s[1])
        # for i in range(5):
        #     self._topic_word.append(temp[i][0])
        # #使用lda
        # ldabow = self.ldadict.doc2bow(self._clean_text)
        # self.ldaans = self.ldamodel.get_document_topics(ldabow)


    def show_all_params(self):
        print("拼写错误数目：",self._spell_mistake_num)
        print("文章长度：",self._composition_num)
        print("文章总词数：",self._composition_total_voc)

        # 语句方面
        print("语法错误：",self._grammar_mistake_num)
        print("语法错误信息：",self._mistake_message)
        print("平均每句子词汇量：",self._avg_voc_num)
        print("平均每句子短语量：",self._avg_pharse_nun)
        print("平均每句语法错误：",self._avg_grammar_mistake_num)

        # 篇章方面

        print("段落数目：",self._paragraph_num)  # 段落数目
        print("主题词汇：",self._topic_word)

    def get_score_by_rule(self):
        self.score -= self._spell_mistake_num*0.5   #一个拼写错误扣0.5分
        self.score -= self._grammar_mistake_num*0.5 #一个语法错误扣0.5分

        # 文章长度评分
        if self._composition_num<50:self.score-=3
        elif self._composition_num<100:self.score-=2
        elif self._composition_num<150:self.score-=1
        elif self._composition_num>300:self.score-=1
        elif self._composition_num>400:self.score-=2
        elif self._composition_num>500:self.score-=3

        #文章总词汇数目评分

        if self._composition_total_voc<50:self.score-=2
        elif self._composition_total_voc<75:self.score-=1
        elif self._composition_total_voc<100:self.score+=0.5
        else:self.score+=1

        #文章句子数目评分

        if self._composition_sent_len<10:self.score*=0.9
        elif self._composition_sent_len<15:self.score*=1.1
        elif self._composition_sent_len<20:self.score*=1.2

        #篇章评分：
        if self._paragraph_num is 1:self.score -=2
        elif self._paragraph_num is 2:self.score-=1
        elif self._paragraph_num is 4:self.score+=1

        for topic_word in self._topic_word:
            if topic_word in CompositionScroe.article_topic[self.topic]:self.score*=1.1
            #可更改为于主题词的wordvec的相似度

        if self.score>10:self.score=10
        if self.score<0:self.score=0



if __name__ == "__main__":
    # article = sys.argv[1]
    # topic = sys.argv[2]
    # com_score = CompositionScroe(article,1)
    # com_score.get_word_param()
    # com_score.get_sent_param()
    # com_score.get_article_param()
    # com_score.show_all_params()
    # print(com_score.score)


    read_book = xlrd.open_workbook("data/score_new/clec_st4.xls")
    write_book = xlwt.Workbook()
    read_sheet = read_book.sheets()[0]
    write_sheet = write_book.add_sheet('sheet1')
    write_sheet.col(1).width = 100 * 256
    for i in range(1,18):write_sheet.col(1).width = 50*256
    write_sheet.write(0,0,"index")
    write_sheet.write(0,1,"title")
    write_sheet.write(0,2,"theme")
    write_sheet.write(0,3,"score")
    write_sheet.write(0,4,"spell_mistake_num")
    write_sheet.write(0,5,"composition_num")
    write_sheet.write(0,6,"composition_total_voc")
    write_sheet.write(0,7,"comma_num")
    write_sheet.write(0,8,"question_mark_num")
    write_sheet.write(0,9,"period_mark_num")
    write_sheet.write(0,10,"stand_deviation_word_len")
    write_sheet.write(0,11,"grammar_mistake_num")
    write_sheet.write(0,12,"avg_word_num")
    write_sheet.write(0,13,"avg_grammar_mistake_num")
    write_sheet.write(0,14,"composition_sent_num")
    write_sheet.write(0,15,"preposition_num")
    write_sheet.write(0,16,"conjunction_num")
    write_sheet.write(0,17,"stand_deviation_sent_num")
    nrow = read_sheet.nrows

    for i in range(1,nrow):
        id = i
        title = read_sheet.cell_value(i,0)
        score = read_sheet.cell_value(i,1)
        theme = read_sheet.cell_value(i,2)
        content = read_sheet.cell_value(i,3)
        content = data_treatment.pretrement_extra_space(content)
        com_score = CompositionScroe(content,1)
        com_score.get_word_param()
        com_score.get_sent_param()
        com_score.get_article_param()
        write_sheet.write(i, 0, id)
        write_sheet.write(i, 1, title)
        write_sheet.write(i, 2, theme)
        write_sheet.write(i, 3, score)
        write_sheet.write(i, 4, com_score._spell_mistake_num)
        write_sheet.write(i, 5, com_score._composition_num)
        write_sheet.write(i, 6, com_score._composition_total_voc)
        write_sheet.write(i, 7, com_score._comma_num)
        write_sheet.write(i, 8, com_score._question_mark_num)
        write_sheet.write(i, 9, com_score._period_mark_num)
        write_sheet.write(i, 10, com_score._stand_deviation_word_len)
        write_sheet.write(i, 11, com_score._grammar_mistake_num)
        write_sheet.write(i, 12, com_score._avg_word_num)
        write_sheet.write(i, 13, com_score._avg_grammar_mistake_num)
        write_sheet.write(i, 14, com_score._composition_sent_num)
        write_sheet.write(i, 15, com_score._preposition_num)
        write_sheet.write(i, 16, com_score._conjunction_num)
        write_sheet.write(i, 17, com_score._stand_deviation_sent_num)
    write_book.save("data/score_new/score_list_st4.xls")
















    def compostion_check(self):     # 找出文本中的拼写和语法错误
        for sent in self._composition_sentences:
            self.get_spell_mistake(sent)
            self.grammar_check(sent)




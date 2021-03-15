import nltk
import nltk.corpus
import cleantext
import math
import random
import sys

from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from data_treatment import data_treatment
from matplotlib import pyplot as plt
class CompositionScore2:

    def __init__(self,mode=1,theme=1):
        if mode==1:
            file_name = "data/score/sh3_"+str(theme)+".pkl"
            if theme==2:
                file_name = "data/score/sh3_" + '1' + ".pkl"
            corpus = joblib.load(file_name)
            self.score_list = corpus[0]
            while "?" in self.score_list:
                self.score_list.remove("?")
            self.score_list = [int(i) for i in self.score_list]
            self.article_list = corpus[1]

        elif mode==2:
            file_name = "data/score/sh4" + str(theme) + ".pkl"
            corpus = joblib.load(file_name)
            while "?" in self.score_list:
                self.score_list.remove("?")
            self.score_list = [int(i) for i in self.score_list]
            self.article_list = corpus[1]
        self.vec_dimension = 0
        self.score = "A"    # A for 13-15;B for 10-12;C for 7-9;D for 4-6
        self.score_class = [4,5,6,7,8,9,10,11,12,13,14,15]
        self.Ascore = [13,14,15]
        self.Bscore = [10,11,12]
        self.Cscore = [7,8,9]
        self.Dscore = [4,5,6]
        self.class_num = 4
        self.start = 100
        self.k = 200
        self.N = len(self.article_list)
        self.m = 20
        self.small_size = 50
        if (theme == 3):
            self.start = 50
            self.k = 100
            self.N = len(self.article_list)
            self.m = 20
            self.small_size = 20
        if(theme==5):
            self.start =100
            self.k = 200
            self.N = len(self.article_list)
            self.m = 20
            self.small_size = 50
        self.phrase_feature_list = []
        # if mode==1:
        #     self.train_article_vec = joblib.load('data/score/sh3_train_article_vec.pkl')
        #     self.phrase_feature_list = joblib.load('data/score/sh3_phrase_feature_list.pkl')
        # elif mode==2:
        #     self.train_article_vec = joblib.load('data/score/sh4_train_article_vec.pkl')
        #     self.phrase_feature_list = joblib.load('data/score/sh3_phrase_feature_list.pkl')

        self.classier_list = []

    def get_vec_dimentsion(self):

        # 得到所有文章的相邻二元词组列表
        self.all_article_phrase2_list = []
        self.all_article_phrase2_llist = []
        self.all_article_phrase2_lllist = []
        for i in range(self.N):
            article = cleantext.clean(self.article_list[i], extra_spaces=True, lowercase=True)
            # article = data_treatment.pos_lemmatize_article(article)
            sents = nltk.sent_tokenize(article)
            article_phrase2_list = []
            temp = []
            for sent in sents:
                sent = cleantext.clean(sent,punct=True)
                words = nltk.word_tokenize(sent)
                for j in range(len(words)-1):
                    article_phrase2_list.append([words[j],words[j+1]])
            for phrase in article_phrase2_list[:]:

                if ',' in phrase or '.' in phrase or '\'\'' in phrase or '``' in phrase or 'n\'t' in phrase or 'it' in phrase:
                    article_phrase2_list.remove(phrase)
                else:
                    temp.append(phrase)
            for phrase in article_phrase2_list[:]:

                if article_phrase2_list.count(phrase)>1:
                    article_phrase2_list.remove(phrase)
            self.all_article_phrase2_list += article_phrase2_list
            self.all_article_phrase2_llist.append(article_phrase2_list)
            self.all_article_phrase2_lllist.append(temp)

        temp = []
        for element in self.all_article_phrase2_list:
            if element not in temp:
                temp.append(element)
        self.all_article_phrase2_list = temp

        #计算信息增益率
        p_ci = [self.score_list.count(i)/len(self.score_list) for i in self.score_class]
        p_ci[0] = sum([p_ci[i] for i in [0,1,2]])
        p_ci[1] = sum([p_ci[i] for i in [3,4,5]])
        p_ci[2] = sum([p_ci[i] for i in [6,7,8]])
        p_ci[3] = sum([p_ci[i] for i in [9,10,11]])
        p_ci = p_ci[0:4]
        prefix_constants = 0
        for i in range(self.class_num):
            if p_ci[i]==0:prefix_constants+=0
            else:
                prefix_constants+=p_ci[i]*math.log(p_ci[i])
        prefix_constants = -prefix_constants
        p_t_ci = []
        p_t=[]
        for phrase in self.all_article_phrase2_list:
            numA = 0
            numB = 0
            numC = 0
            numD = 0
            num = 0
            for i in range(len(self.all_article_phrase2_llist)):
                article_phrase2 = self.all_article_phrase2_llist[i]
                if phrase in article_phrase2:
                    num+=1
                    if self.score_list[i] in self.Ascore:numA += 1
                    elif self.score_list[i] in self.Bscore:numB += 1
                    elif self.score_list[i] in self.Cscore:numC += 1
                    elif self.score_list[i] in self.Dscore: numD += 1
            p_t.append(num/len(self.all_article_phrase2_llist))
            p_tD = numD/len(self.all_article_phrase2_llist)
            p_tC = numC /len(self.all_article_phrase2_llist)
            p_tB = numB / len(self.all_article_phrase2_llist)
            p_tA = numA / len(self.all_article_phrase2_llist)
            p_t_ci.append([p_tD,p_tC,p_tB,p_tA])


        p_tn = [1-i for i in p_t]
        p_tn_ci = []
        for i in range(len(self.all_article_phrase2_list)):
            temp= []
            for j in range(self.class_num):
                temp.append(p_ci[j]-p_t_ci[i][j])
            p_tn_ci.append(temp)

        gain_list = []

        try:
            for i in range(len(self.all_article_phrase2_list)):
                middle_coff = 0
                suffix_coff = 0
                for j in range(self.class_num):
                    if p_t_ci[i][j]==0:
                        pass
                    else:
                        middle_coff+=(p_t_ci[i][j]*math.log(p_t_ci[i][j]))
                    if p_tn_ci[i][j]==0:
                        pass
                    else:
                        suffix_coff += (p_tn_ci[i][j] * math.log(p_tn_ci[i][j]))


                middle_coff*=p_t[i]
                suffix_coff*=p_tn[i]
                gain_list.append(prefix_constants+middle_coff+suffix_coff)
        except ValueError:
            print(i," ",j)
            print(p_t_ci[i][j])
        temp = list(zip(self.all_article_phrase2_list,gain_list))
        temp = sorted(temp,key=lambda x: x[1],reverse=True)

        # x = [i for i in range(len(temp))]
        # y = [i[1] for i in temp]
        # plt.plot(x,y)
        self.phrase_feature_list = temp[self.start:self.k]
        # return temp


    def get_train_article_vec(self):
        N = self.N
        train_article_vec = []
        for i in range(N):
            temp =[]
            for j in range(self.start,self.k):
                phrase = self.phrase_feature_list[j-self.start][0]
                df = self.all_article_phrase2_lllist[i].count(phrase)

                temp.append(math.log(N/(1+df)))
            train_article_vec.append(temp)

        self.train_article_vec = train_article_vec

    def get_small_train_article_vec(self,topic="love"):
        self.small_tag_list = []
        self.small_train_article_vec = []
        for i in range(self.small_size):
            self.small_tag_list.append(self.score_list[random.randint(0,self.N-1)])
            self.small_train_article_vec.append(self.train_article_vec[random.randint(0,self.N-1)])

    def get_bayesian_classier(self):
        classier_list = []
        for i in range(self.m):
            extraction_list = []
            extraction_tag_list =[]
            for j in range(self.small_size):
                index = random.randint(0,self.N-1)
                extraction_list.append(self.train_article_vec[index])
                extraction_tag_list.append(self.score_list[index])
            train_set = extraction_list+self.small_train_article_vec
            tag_set = extraction_tag_list+self.small_tag_list
            for i in range(len(tag_set)):
                if tag_set[i] in self.Ascore:tag_set[i]='A'
                elif tag_set[i] in self.Bscore:tag_set[i]='B'
                elif tag_set[i] in self.Cscore:tag_set[i]='C'
                elif tag_set[i] in self.Dscore:tag_set[i]='D'
            mnb = MultinomialNB()
            mnb.fit(train_set,tag_set)
            classier_list.append(mnb)
        self.classier_list = classier_list

    def get_test_artcile_vec(self,article):
        article = cleantext.clean(article, extra_spaces=True, lowercase=True)
        # article = data_treatment.pos_lemmatize_article(article)
        sents = nltk.sent_tokenize(article)
        article_phrase2_list = []
        temp = []
        for sent in sents:
            sent = cleantext.clean(sent, punct=True)
            words = nltk.word_tokenize(sent)
            for j in range(len(words) - 1):
                article_phrase2_list.append([words[j], words[j + 1]])
        for phrase in article_phrase2_list:
            if ',' in phrase or '.' in phrase or '\'\'' in phrase or '``' in phrase or 'n\'t' in phrase or 'it' in phrase:
                article_phrase2_list.remove(phrase)
        test_article_vec = []
        N = self.N
        for i in range(self.start,self.k):
            phrase = self.phrase_feature_list[i-self.start][0]
            df = article_phrase2_list.count(phrase)
            test_article_vec.append(math.log(N / (1 + df)))
        self.test_article_vec = test_article_vec

    def get_score(self):
        test_vec = self.test_article_vec
        score_list = []
        class_list = [0,0,0,0]
        for i in range(len(self.classier_list)):
            score = self.classier_list[i].predict([test_vec])
            score_list.append(score.tolist())
        for i in range(len(score_list)):
            if score_list[i][0] is 'D':class_list[0]+=1
            elif score_list[i][0] is 'C':class_list[1]+=1
            elif score_list[i][0] is 'B':class_list[2]+=1
            elif score_list[i][0] is 'A': class_list[3]+=1
        index = class_list.index(max(class_list))
        if index==0:self.score="D"
        elif index==1:self.score="C"
        elif index==2:self.score="B"
        elif index==3:self.score="A"


    def get_preformance_para(self):
        extraction_list = []
        extraction_tag_list = []
        for j in range(30):
            index = random.randint(0, self.N - 1)
            extraction_list.append(self.train_article_vec[index])
            extraction_tag_list.append(self.score_list[index])
        test_article_list = extraction_list
        test_tag_list = extraction_tag_list
        predict_list = []
        for i in range(len(test_tag_list)):
            temp = []
            for j in range(len(self.classier_list)):
                temp.append(self.classier_list[j].predict([test_article_list[i]]))
            predict_list.append(temp)

        print(predict_list)
        print(test_tag_list)



if __name__ == "__main__":
    #article = sys.argv[1]
    test = CompositionScore2(mode=1,theme=6)
    test.get_vec_dimentsion()
    test.get_train_article_vec()
    # joblib.dump(test.train_article_vec,"data/score/sh3_5_train_article_vec.pkl")
    # joblib.dump(test.phrase_feature_list,'data/score/sh3_5_phrase_feature_list.pkl')
    test.get_small_train_article_vec()
    test.get_bayesian_classier()
    predict_list = []
    for i in range(len(test.score_list)):

        test.get_test_artcile_vec(test.article_list[i])
        test.get_score()
        predict_list.append(test.score)


    print(predict_list)
    print(test.score_list)









import nltk
import gensim
import cleantext
import re
import xlrd
import sys
from gensim.models import word2vec
from data_treatment import data_treatment
from nltk.corpus import reuters
from nltk.corpus import wordnet as wn
from sklearn.externals import joblib
from nltk.stem import WordNetLemmatizer
class Synonyms_suggestion:

    def __init__(self,model_type):
        self.model1_path = "E:\\programming\\NLP\\TQCorpus\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin"
        self.model2_path = "data\\txt\\Economist\\Economist.txt"

        if model_type == 0:
            pass
        if model_type==1:
            self.model1 = gensim.models.KeyedVectors.load_word2vec_format(self.model1_path, binary=True)
        elif model_type==2:
            self.f_model2_in = open(self.model2_path,"r",encoding="ISO-8859-1")   #使用ISO解码
            all_string = self.f_model2_in.read()
            all_list = nltk.sent_tokenize(all_string)
            train_sentences_model2 = []
            for i in range(0,len(all_list)):
                train_sentences_model2.append(cleantext.clean(all_list[i]))
            train_sentences_model3 = list(reuters.sents())
            for sent in train_sentences_model3:
                train_sentences_model2.append(sent)
            self.model2 = word2vec.Word2Vec(train_sentences_model2, min_count=2, window=3, size=300)

        self.lemmatizer = WordNetLemmatizer()   # 词性还原器
        # 载入人工规则替换词表
        self.artificial_word_book = xlrd.open_workbook("data/suggestion/word_by_rule.xls")
        self.artificial_word_sheet_vec = self.artificial_word_book.sheet_by_index(0)
        self.artificial_word_sheet_adj = self.artificial_word_book.sheet_by_index(1)
        self.artificial_word_vec = []
        self.artificial_word_adj = []
        for i in range(0,self.artificial_word_sheet_vec.ncols):
            temp_list = self.artificial_word_sheet_vec.col_values(i)[2:]
            temp_list = [w.lower() for w in temp_list]
            temp_list = [w for w in temp_list if w != ' ' and w != '']
            for i in range(len(temp_list)):
                temp_list[i] = self.lemmatizer.lemmatize(temp_list[i], pos='v')
            self.artificial_word_vec.append(temp_list)
        for i in range(0,self.artificial_word_sheet_adj.ncols):
            temp_list = self.artificial_word_sheet_adj.col_values(i)[2:]
            temp_list = [w.lower() for w in temp_list]
            temp_list = [w for w in temp_list if w != ' ' and w != '']
            self.artificial_word_adj.append(temp_list)


    def suggestion_word(self,word,sentence,model=2):

        # 词性处理
        sentence = nltk.word_tokenize(sentence)
        pos_tag_list = nltk.pos_tag(sentence)
        tag = pos_tag_list[sentence.index(word)][1]
        word = word.lower()

        # suggestion by artificial rule
        suggestion_list_artificial_rule = []
        if tag.startswith('VB'):
            word = self.lemmatizer.lemmatize(word, pos='v')
            for i in range(0, len(self.artificial_word_vec)):
                if word in self.artificial_word_vec[i]:
                    suggestion_list_artificial_rule = self.artificial_word_vec[i]
                    break

        elif tag.startswith('JJ'):
            word = self.lemmatizer.lemmatize(word, pos='a')
            for i in range(0, len(self.artificial_word_adj)):
                if word in self.artificial_word_adj[i]:
                    suggestion_list_artificial_rule = self.artificial_word_adj[i]
                    break

        elif tag.startswith('R'):
            word = self.lemmatizer.lemmatize(word, pos='r')
            for i in range(0, len(self.artificial_word_vec)):
                if word in self.artificial_word_adj[i]:
                    suggestion_list_artificial_rule = self.artificial_word_adj[i]
                    break
        else:
            word = self.lemmatizer.lemmatize(word, pos='n')

        # suggestion by wordnet

        if tag.startswith('NN'):
            word_meaning_list = wn.synsets(word, pos=wn.NOUN)
        elif tag.startswith('VB'):
            word_meaning_list = wn.synsets(word, pos=wn.VERB)
        elif tag.startswith('JJ'):
            word_meaning_list = wn.synsets(word, pos=wn.ADJ)
        elif tag.startswith('R'):
            word_meaning_list = wn.synsets(word, pos=wn.ADV)
        else:
            word_meaning_list = wn.synsets(word)
        suggestion_ans_wordnet = []
        for word_meaning in word_meaning_list:
            lemmas_ans_wordnet = []
            word_meaning_hypernyms = word_meaning.hypernyms()
            word_meaning_hyponyms = word_meaning.hyponyms()
            word_meaning_similar = word_meaning.similar_tos()
            lemmas_ans_wordnet+=word_meaning_hyponyms
            lemmas_ans_wordnet+=word_meaning_hypernyms
            lemmas_ans_wordnet+=word_meaning_similar
            for i in range(len(lemmas_ans_wordnet)):
                syn = lemmas_ans_wordnet[i]
                suggestion_ans_wordnet.append(str(syn.lemmas()[0].name()))
        suggestion_ans_wordnet = data_treatment.pretrement_for_synonyms(suggestion_ans_wordnet)


        # suggestion by word2vec
        suggestion_list_word2vec = []
        if model==0:
            suggestion_list_word2vec = []
        if model==1:
            suggestion_list_word2vec = self.model1.most_similar([word],topn=20)
        elif model==2:
            suggestion_list_word2vec = self.model2.most_similar([word],topn=20)
        suggestion_ans_word2vec = []
        for i in range (0,len(suggestion_list_word2vec)):
            suggestion_ans_word2vec.append(suggestion_list_word2vec[i][0])
        suggestion_ans_word2vec = data_treatment.pretrement_for_synonyms(suggestion_ans_word2vec)
        ## 去除_号
        for i in range(len(suggestion_ans_word2vec)):
            word = suggestion_ans_word2vec[i]
            word = word.replace("_", " ")
            if tag.startswith('NN'):
                word = self.lemmatizer.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                word = self.lemmatizer.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                word = self.lemmatizer.lemmatize(word, pos='a')
            elif tag.startswith('R'):
                word = self.lemmatizer.lemmatize(word, pos='r')
            else:
                word = self.lemmatizer.lemmatize(word)
            suggestion_ans_word2vec[i] = word
        suggestion_ans_word2vec = list(set(suggestion_ans_word2vec))
        final_ans = []


        final_ans+=suggestion_list_artificial_rule
        final_ans += suggestion_ans_wordnet
        final_ans += suggestion_ans_word2vec


        return final_ans


if __name__=="__main__":

    word = sys.argv[1]
    sent = sys.argv[2]
    mode = sys.argv[3]
    # word = "love"
    # sent = "i love you"
    # mode = 0
    synonyms = Synonyms_suggestion(int(mode))
    syn_list = synonyms.suggestion_word(word,sent,int(mode))
    print(syn_list)
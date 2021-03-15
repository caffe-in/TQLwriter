import nltk
import xlrd
import string
import sys

class sent_suggestion:

    def __init__(self):
        corpus_book = xlrd.open_workbook(r'data/suggestion/economoist_corpus.xls')
        corpus_sheet = corpus_book.sheets()[0]
        nrows = corpus_sheet.nrows
        ncols = corpus_sheet.ncols
        self.corpus_theme1 = ""
        self.corpus_theme2 = ""
        self.corpus_theme3 = ""
        self.corpus_theme4 = ""
        self.corpus_theme5 = ""
        self.corpus_theme6 = ""
        self.corpus_theme_others = ""
        for i in range(1,nrows):
            theme_list = [0]*3
            article = corpus_sheet.cell_value(i,4)
            theme_list[0] = corpus_sheet.cell_value(i, 0)
            theme_list[1] = corpus_sheet.cell_value(i, 1)
            theme_list[2] = corpus_sheet.cell_value(i, 2)
            for j in range(len(theme_list)):
                if int(theme_list[j])==1:
                    self.corpus_theme1+=article
                    self.corpus_theme1+=" "
                    self.corpus_theme_others +=article
                    self.corpus_theme_others +=" "
                elif int(theme_list[j])==2:
                    self.corpus_theme2 += article
                    self.corpus_theme2 += " "
                    self.corpus_theme_others += article
                    self.corpus_theme_others += " "
                elif int(theme_list[j])==3:
                    self.corpus_theme3 += article
                    self.corpus_theme3 += " "
                    self.corpus_theme_others += article
                    self.corpus_theme_others += " "
                elif int(theme_list[j])==4:
                    self.corpus_theme4 += article
                    self.corpus_theme4 += " "
                    self.corpus_theme_others += article
                    self.corpus_theme_others += " "
                elif int(theme_list[j])==5:
                    self.corpus_theme5 += article
                    self.corpus_theme5 += " "
                    self.corpus_theme_others += article
                    self.corpus_theme_others += " "
                elif int(theme_list[j])==6:
                    self.corpus_theme6 += article
                    self.corpus_theme6 += " "
                    self.corpus_theme_others += article
                    self.corpus_theme_others += " "


    def get_concordance(self,word,theme):
        self.corpus_theme_list = [self.corpus_theme1,self.corpus_theme2,self.corpus_theme3,
                                  self.corpus_theme4,self.corpus_theme5,self.corpus_theme6,
                                  self.corpus_theme_others]
        token = nltk.word_tokenize(self.corpus_theme_list[theme])
        text = nltk.Text(token)
        sent_list = text.concordance_list(word)
        sent_ans = []
        punc = [',', '.', ':', ';', '?', '(', ')', '[',
                ']', '&', '!', '*', '@', '#', '$', '%',
                '\'','"']
        punc_pass = ['’','“','”','）','（']
        for i in range(len(sent_list)):
            temp = sent_list[i]
            temp_left = temp.left
            temp_right = temp.right
            sent = ""
            for j in range(len(temp_left)):
                if temp_left[j] in punc:
                    sent = sent[0:len(sent)-1]
                    sent+=temp_left[j]
                    sent+=" "
                elif temp_left[j] in punc_pass:
                    continue
                else:
                    sent+=temp_left[j]+" "
            sent += word + " "
            for j in range(len(temp_right)):
                if temp_right[j] in punc:
                    sent = sent[0:len(sent) - 1]
                    sent+=temp_right[j]
                    sent+=" "
                elif temp_right[j] in punc_pass:
                    continue
                else:
                    sent+=temp_right[j]+" "
            sent_list[i] =nltk.sent_tokenize(sent)
        for i in range(len(sent_list)):
            for j in  range(len((sent_list[i]))):
                if word in sent_list[i][j]:
                    sent_ans.append(sent_list[i][j])
        self.sent_suggestion = sent_ans
        return sent_ans


if __name__=="__main__":
    sent_sugg = sent_suggestion()
    word = sys.argv[1]
    theme = int(sys.argv[2])
    # word = "love"
    # theme =1
    sent_list = sent_sugg.get_concordance(word,theme)
    print(sent_list)

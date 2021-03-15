import nltk
import sklearn
import xlrd
import numpy as np
import math
import sys
import data_treatment

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from compositon_score import CompositionScroe
class composition_score3:

    def __init__(self,mode='st_3'):
        if mode =='st_3':
            self.workbook = xlrd.open_workbook('data/score_new/score_list_st3.xls')
        elif mode == 'st_4':
            self.workbook = xlrd.open_workbook('data/score_new/score_list_st4.xls')
        sheet = self.workbook.sheets()[1]
        #sheet
        #[index,title,theme,score,]
        nrow = sheet.nrows
        self.score_list = [float(sheet.cell_value(i,3)) for i in range(1,nrow)]
        self.attribute_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        self.attribute_num = 14
        self.attribute_name_list = sheet.row(0)[4:]
        self.R = 0.08
        for i in range(1,nrow):
            for j in range(len(self.attribute_list)):
                self.attribute_list[j].append(float(sheet.cell_value(i,j+4)))

    def get_pearson_coff(self):
        pearson_coff = [self.score_list]+self.attribute_list[0:len(self.attribute_list)]
        pccf = np.corrcoef(pearson_coff)
        self.pearson_coff_matrix = pccf
        return pccf
    def get_final_attribute_list(self):
        self.final_attribute_list = []
        self.final_attribute_name_list = []
        temp_list = self.pearson_coff_matrix.tolist()
        scaler = MinMaxScaler()
        for i in range(1,len(temp_list[0])):
            if math.fabs(temp_list[0][i])>=self.R:
                self.final_attribute_name_list.append(self.attribute_name_list[i-1])
                self.final_attribute_list.append(self.attribute_list[i-1])
        self.final_attribute_list_normalization = scaler.fit_transform(np.array(self.final_attribute_list).T)
    def get_score_test(self):
        model = LinearRegression()
        train_attribute = self.final_attribute_list_normalization[0:int(len(self.final_attribute_list_normalization)*0.8),:]
        train_score = np.array(self.score_list)[0:int(len(self.score_list)*0.8)]
        test_attribute = self.final_attribute_list_normalization[int(len(self.final_attribute_list_normalization)*0.8):,:]
        test_score = np.array(self.score_list)[int(len(self.score_list)*0.8):]
        model.fit(train_attribute,train_score)
        self.linear_model = model
        ans = model.predict(test_attribute)
        print(model.score(train_attribute,train_score))
        print(model.predict(test_attribute))
        print(test_score)
        lenth =0
        for i in range(len(test_score)):
            ans[i] = ans[i] - test_score[i]
        print(ans)
        ans_list = ans.tolist()
        length = 0
        for i in range(len(ans_list)):
            if math.fabs(ans_list[i])<1.5:
                length+=1
        print(length/len(ans_list))

    def get_score(self,feature_list):
        scaler = MinMaxScaler()
        for i in range(len(feature_list)):
            self.final_attribute_list[i] = [feature_list[i]]+self.final_attribute_list[i]
        feature_list = scaler.fit_transform(np.array(self.final_attribute_list).T)[0,:]
        model = LinearRegression()
        train_attribute = self.final_attribute_list_normalization[0:int(len(self.final_attribute_list_normalization) * 0.8), :]
        train_score = np.array(self.score_list)[0:int(len(self.score_list) * 0.8)]
        # test_attribute = self.final_attribute_list_normalization[int(len(self.final_attribute_list_normalization) * 0.8):, :]
        # test_score = np.array(self.score_list)[int(len(self.score_list) * 0.8):]
        model.fit(train_attribute, train_score)
        self.linear_model = model
        ans = model.predict(feature_list.reshape(1,-1))[0]
        print(round(ans,1))



if __name__=="__main__":

    # article = sys.argv[1]
    # mode = sys.argv[2]
    # article = " As we well known , the change of the life expectancy and the infant mortaility is very big in developing countries. Seen from the picture, we can find that in 1960 life expectancy was Forty years old. in developing countries. But in 1990. It has risen to sixty years old: on the other hand. the infant mortaility was 200 deaths per 1,000 births, in 1960, But in 1990 it becomes 100 deaths per 1,000 births. 	The important point of causing this is that the developing countries' medical technology make a great development than before. in the last , the situation was very difficult and the people didn't solve the eatness and clother . Going to see doctor was inusion  . Now the people's life become good. So life expectancy and infant morlality is good. 	On the other hand, Now the sports is developed in developing coutries , sports can give health to people. 	So health gains in developing countries. "
    # article = data_treatment.data_treatment.pretrement_extra_space(article)
    # cs1 = CompositionScroe(article,1)
    # cs1.get_word_param()
    # cs1.get_sent_param()
    # cs1.get_article_param()
    cs3 = composition_score3(mode='st_3')
    cs3.get_pearson_coff()
    cs3.get_final_attribute_list()
    # feature_list = []
    #
    # for feature in cs3.final_attribute_name_list:
    #     temp = cs1.__getattribute__('_'+feature.value)
    #     feature_list.append(temp)


    cs3.get_score_test()




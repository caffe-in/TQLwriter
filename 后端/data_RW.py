import json


class data_RW:

    '''
    this is a class for Corpus generation and Processing data
    '''

    json_filename = ""
    list_bad = []
    composition_bad_num = 0
    composition_good_num = 0
    f = open("data/json/data.json", "a")

    def __init__(self, good_num, bad_num):
        json_filename = "data/json/data.json"
        f = open(r"data/json/data.json", "a")  # 不会删除文件添加
        self.composition_bad_num = bad_num
        self.composition_good_num = good_num

    def write_json(self):

        for i in range(0, self.composition_good_num):
            try:

                fin_name = "data//txt//good_composition//file" + str(i) + ".txt"
                fin = open(fin_name, "r", encoding="utf-8")
                fin_str = fin.read()
                dict = {"tag":"good","text":fin_str}
                self.f.writelines(json.dumps(dict) + '\n')
            except Exception as error:
                print(fin_name)
                print(error)
        for i in range(0, self.composition_bad_num):
            fin_name = "data//txt//bad_composition//file" + str(i) + ".txt"
            fin = open(fin_name, "r", encoding="utf-8")
            fin_str = fin.read()
            dict = {"tag":"bad","text":fin_str}
            self.f.writelines(json.dumps(dict) + '\n')








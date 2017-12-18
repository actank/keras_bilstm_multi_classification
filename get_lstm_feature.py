from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
import numpy as np


feature_data = []
ins_num = 0
label = []
with open("./train_data") as f:
    for line in f:
        l = line.split('\t')[0].strip()
        if l == '春':
            l = 1
        elif l == '夏':
            l = 2
        elif l == '秋':
            l = 3
        elif l == '冬':
            l = 4
        else:
            l = 0
        label.append(str(l))
        ins_num += 1


def get_dictionary():
    texts = []
    with open("./train_data") as f:
        for line in f:
            l = line.split('\t')[1].strip()
            l = l.split(" ")
            l = [i.strip() for i in l if i.find("gram")==-1]
            texts.append(l)
    dictionary = corpora.Dictionary(texts)
    dictionary.save('train_lstm.dict')
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('train_lstm.mm', corpus) 
    return

def get_sample():
    dic = corpora.Dictionary.load('train_lstm.dict')
    corpus = corpora.MmCorpus('train_lstm.mm')
    max_len = 0
    ins = 0
    with open("train_data") as f:
        for line in f:
            line = line.split("\t")[1]
            line = line.split(" ")
            line = [i.strip() for i in line if i.find("gram")==-1]
            if len(line) > max_len:
                max_len =  len(line)
    with open("train_data") as f:
        for line in f:
            line = line.split("\t")[1].strip()
            line = line.split(" ")
            line = [i.strip() for i in line if i.find("gram")==-1]
            #去重m
            m = {}
            tmp = [label[ins]]
            for i in range(len(line)):
                l = line[i].strip()
                feature_id = dic.token2id[l]
                tmp.append(str(feature_id))
            if len(tmp) < max_len:
                tmp.extend([str(0) for j in range(max_len-len(tmp))])
            else:
                tmp = tmp[:max_len]
            feature_data.append(tmp)
            ins += 1
    return
def output():
    with open("./train_lstm.data", "w") as f, open("./submit_lstm.data", "w") as f1, open("./train_data_index_lstm", "w") as f3, open("./submit_data_index_lstm", "w") as f4, open("./train_data") as f5, open("./goods_info") as f6:
        for i in feature_data:
            l = i[0]
            b = i[1:]
            #b = sorted(b,key=lambda d : int(d.split(":")[0]))
            if l in ('1','2','3','4'):
                f.write(str(int(l)-1) + "," + ",".join(b) + "\n")
                f3.write(f5.readline().strip() + " " + f6.readline().split("\t")[0] + "\n")
            else:
                f1.write(",".join(b) + "\n")
                f4.write(f5.readline().strip() + " " + f6.readline().split("\t")[0] + "\n")
    

    return


if __name__ == "__main__":
    #get_dictionary()
    get_sample()
    output()

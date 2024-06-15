import pickle
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import random
import os
import jieba
import re
import collections
from tqdm import tqdm
from transformers import BertTokenizer, AutoTokenizer
import torch.nn as nn
from tokenizer import MyTokenizer
import json
from setting import CONTRA_WAY, CONTRASTIVE, PRETRAIN,DEVICE, MODEL


RANDOM_SEED = 19
torch.manual_seed(RANDOM_SEED)

def text_cleaner(text):
    def load_stopwords(filename):
        stopwords = []
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.replace("\n", "")
                stopwords.append(line)
        return stopwords

    stop_words = load_stopwords("data/stopwords.txt")

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    text = re.sub("[0-9\.]+元", "", text)
    stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    word_tokens = jieba.cut(text)

    def str_find_list(string, words):
        for word in words:
            if string.find(word) != -1:
                return True
        return False

    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]
    return " ".join(text)

#只清洗数据，不去停用词。
def text_cleaner2(text):
    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    # 换行替换为#, 空格替换为&
    text = text.replace("#", "").replace("$", "").replace("&", "")
    text = text.replace("\n", "").replace(" ", "")

    return text

#用的时候改改属性即可，中文的需要bert来tokenizer
class myDataset(Dataset):
    def __init__(self, justice, charge, charge_num, charge_label, article , article_num, article_label):
        self.justice = justice
        self.charge= torch.LongTensor(charge)
        self.charge_num= torch.LongTensor(charge_num)
        self.article= torch.LongTensor(article)
        self.article_num= torch.LongTensor(article_num)
        self.charge_label= torch.LongTensor(charge_label)
        self.article_label= torch.LongTensor(article_label)

    def __getitem__(self, idx):
        #为了匹配bert_tokenizer的返回结果
        if CONTRASTIVE and CONTRA_WAY=="supcon2":
            input_ids = [self.justice["input_ids"][idx], self.justice["input_ids"][idx]]
            # input_ids = torch.cat([self.justice["input_ids"][idx], self.justice["input_ids"][idx]], dim=0)
        else:
            input_ids = self.justice["input_ids"][idx]
        return {"justice":
                {
                    "input_ids": input_ids,
                    "token_type_ids": self.justice["token_type_ids"][idx],
                    "attention_mask": self.justice["attention_mask"][idx],
                }, "charge": self.charge[idx],"charge_num":self.charge_num[idx],"charge_label":self.charge_label[idx],
                "article": self.article[idx],"article_num":self.article_num[idx],"article_label":self.article_label[idx]}

    def __len__(self):
        return len(self.charge)

def get_split_dataset(idx,justice,charge,charge_num,charge_label,article,article_num,article_label):
    #此处根据ID返回对应的数据，因为dataset需要根据id进行划分
    justice_cur = {
        "input_ids": justice["input_ids"][idx],
        "token_type_ids": justice["token_type_ids"][idx],
        "attention_mask": justice["attention_mask"][idx],
    }
    charge_cur=pd.Series(charge)[idx].tolist()
    charge_num_cur=pd.Series(charge_num)[idx].tolist()
    article_cur=pd.Series(article)[idx].tolist()
    article_num_cur=pd.Series(article_num)[idx].tolist()
    charge_label_cur=pd.Series(charge_label)[idx].tolist()
    article_label_cur=pd.Series(article_label)[idx].tolist()
    return myDataset(justice_cur,charge_cur,charge_num_cur,charge_label_cur,article_cur,article_num_cur,article_label_cur)

def simple_load_multi_data(filename, seq_len, embedding_path, text_clean:bool):
    #LA:将错误行跳过
    #重组数据集
    source=[]
    with open(filename,'r',encoding="utf-8") as f:
        for line in f:
            source.append((json.loads(line)))
    df_data=pd.DataFrame(source)
    charge=[]
    article=[]
    for i in range(len(df_data)):
        article.append(df_data["meta"].iloc[i]["relevant_articles"])
        charge.append(df_data["meta"].iloc[i]["accusation"])
    df=pd.DataFrame()
    df["justice"]=df_data["fact"]
    df["charge"]=charge
    df["article"]=article
    path, file = os.path.split(filename)
    stem, suffix = os.path.splitext(file)

    # 读入csv数据，根据上面text_clean参数是否过stopwords
    if text_clean:
        file = stem+"_clean"+suffix
        clean_data_path = os.path.join(path, file)
        if not os.path.exists(clean_data_path):
            for i in tqdm(range(len(df))):
                # df["charge"][i] = text_cleaner(df["charge"][i])
                df["justice"][i] = text_cleaner(df["justice"][i])
            # df.to_csv(clean_data_path, index=False, sep=",")
            df.to_json(clean_data_path,orient="records",force_ascii=False)
            df=pd.read_json(clean_data_path,orient="records")

        if len(df) != len(df.dropna()):
            print("before drop nan, data num: ", len(df))
            print("after drop nan, data num: ", len(df.dropna()))
        df = df.dropna()
        df = df.reset_index()

    #读取字典
    maps = {}
    c2i_path = "data/label2id_clean/c2i_clean.json"
    a2i_path = "data/label2id_clean/a2i_clean.json"
    with open(c2i_path) as f:
        c2i = json.load(f)
        maps["charge2idx"] = c2i
        maps["idx2charge"] = {str(v): k for k, v in c2i.items()}

    with open(a2i_path) as f:
        a2i = json.load(f)
        maps["article2idx"] = a2i
        maps["idx2article"] = {str(v): k for k, v in a2i.items()}

#读取对应关系，并把它们转换为id
    with open("data/relationship_clean/c2a_clean.json") as f:
        c2a_final = json.load(f)
        c2a_tmp = dict()
        for key, value in c2a_final.items():
            c2a_tmp[maps["charge2idx"][key]] = maps["article2idx"][value]
        maps["c2a"]=c2a_tmp
    dic = collections.defaultdict(list)
    for k,v in maps["c2a"].items():
        dic[v].append(k)
    maps["a2c"] = dic

    # 将fact和opinion文本进行tokenize
    if MODEL=="Bert":
        pkl_path = "data/pkl_bert_811clean/"+stem+".pkl"
    elif MODEL=="Electra" or MODEL=="Al_Trans":
        pkl_path = "data/pkl_electra_single/"+stem+".pkl"
    else:
        pkl_path = "data/pkl_811clean/"+stem+".pkl"

    if not os.path.exists(pkl_path):
        if MODEL=="Bert":
            tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
        elif MODEL=="Electra" or MODEL=="Al_Trans":
            tokenizer=AutoTokenizer.from_pretrained("electra-small")
        else:
            tokenizer=MyTokenizer(embedding_path)

        justice = df["justice"].tolist()
        justice = tokenizer(justice, return_tensors="pt",
                         padding="max_length", max_length=seq_len, truncation=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(justice, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pkl_path, "rb") as f:
            justice = pickle.load(f)

    #将标签转换为one_hot
    def one_hot_labels(labels_index, arg_map):
        label=[0]*len(arg_map)
        for item in labels_index:
            label[int(item)] = 1
        return label
    
    charge=df["charge"].tolist()
    article=df["article"].tolist()
    charge_num=[[] for _ in range(len(df))]
    article_num=[[] for _ in range(len(df))]
    charge_label=[[] for _ in range(len(df))]
    article_label=[[] for _ in range(len(df))]
    for i in range(len(charge)):
        charge_num[i]=len(charge[i])-1
        article_num[i]=len(article[i])-1
        charge[i]=one_hot_labels([maps["charge2idx"][el] for el in charge[i]], maps["charge2idx"])
        article[i]=one_hot_labels([maps["article2idx"][str(el)] for el in article[i]], maps["article2idx"])
        charge_label[i] = range(len(maps["charge2idx"]))
        article_label[i] = range(len(maps["article2idx"]))

    random.seed(RANDOM_SEED)
    idx = list(range(len(df)))
    random.shuffle(idx)  
    dataset = get_split_dataset(idx, justice, charge, charge_num, charge_label, article, article_num, article_label)

    return dataset, maps

def load_details(maps, art_details_path, char_details_path, art_len, char_len, embedding_path):
    article_details=[]
    with open(art_details_path,'r',encoding="utf-8") as f:
        data=json.load(f)
        for key, value in data.items():
            if key in list(maps["article2idx"].keys()):
                article_details.append(value.replace(" ",""))
    charge_details=[]
    with open(char_details_path,'r',encoding="utf-8") as f:
        data=json.load(f)
        for key, value in data.items():
            if key in list(maps["charge2idx"].keys()):
                charge_details.append(value["定义"].replace(" ",""))

    if MODEL=="Bert":
        tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")
    elif MODEL=="Electra" or MODEL=="Al_Trans":
        tokenizer=AutoTokenizer.from_pretrained("electra-small")
    else:
        tokenizer=MyTokenizer(embedding_path)
    print(tokenizer.vocab_size)
    article_details = tokenizer(article_details, return_tensors="pt",
                        padding="max_length", max_length=art_len, truncation=True)
    charge_details = tokenizer(charge_details, return_tensors="pt",
                        padding="max_length", max_length=char_len, truncation=True)
    article_details = tocuda(article_details)
    charge_details = tocuda(charge_details)

    return article_details, charge_details

def tocuda(data):
    if type(data) is dict:
        for k in data:
            data[k] = data[k].to(DEVICE)
    else:
        data = data.to(DEVICE)
    return data
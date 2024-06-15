from gensim.models import Word2Vec
import torch
from tqdm import tqdm
import numpy as np
import jieba
import re

class MyTokenizer:
    #往word2vec生成的word2id中加入特殊token，加在最前面
    def __init__(self,embedding_path) -> None:
        model=Word2Vec.load(embedding_path)
        self.special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        self.id2word=self.special_tokens+model.wv.index_to_key
        self.word2id=model.wv.key_to_index
        #其他对应的id往后移动特殊token总数的长度
        for k in self.word2id.keys():
            self.word2id[k]+=len(self.special_tokens)
        #特殊token对应id加在最前
        for i in range(len(self.special_tokens)):
            self.word2id[self.special_tokens[i]]=i

        #这个size在模型中会用到
        self.vocab_size = len(self.word2id)
        #更改新的向量size
        self.vector_size=model.wv.vector_size
        special_token_size=np.zeros((len(self.special_tokens),self.vector_size))
        self.vectors=model.wv.vectors
        self.vectors=np.concatenate((special_token_size,self.vectors))

    def load_embedding(self):
        return self.vectors
    
    def __call__(self, *args, **kwds):
        return self.encode(*args, **kwds)

    def encode(self,sents,max_length=512,return_tensors="ls",padding="max_length",truncation=True):
        input_ids=[]
        token_type_ids=[]
        attention_mask=[]

        for sent in tqdm(sents):
            #将词映射到对应id
            sent=sent.replace(" ", "")
            sent=jieba.lcut(sent)
            sent=[self.word2id[w] if w in self.word2id.keys() else self.word2id["[UNK]"] for w in sent]
            #句子开头加上SOS，句尾加上EOS
            sent=[self.word2id["[SOS]"]]+sent+[self.word2id["[EOS]"]]
            #padding
            sent+=[0]*max_length
            #截取
            sent=sent[:max_length]

            input_ids.append(sent)
            token_type_ids.append([0]*max_length)
            attention_mask.append([0]*max_length)

        if return_tensors=="pt":
            input_ids = torch.LongTensor(input_ids)
            token_type_ids = torch.LongTensor(token_type_ids)
            attention_mask = torch.LongTensor(attention_mask)
        
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
    #检验encode
    def decode(self,token_ids):
        res=[]
        for token in tqdm(token_ids["input_ids"]):
            sent=[]
            for id in token:
                #读到终止符EOS
                if id==0:
                    break
                sent.append(self.id2word[id])
            sent=" ".join(sent)
            res.append(sent)
        return res


if __name__=="__main__":
    tokenizer=MyTokenizer("gensim_train/word2vec.model")
    tokens=tokenizer.encode(["上午 被告人 贺 宁波市 镇海区 蟹浦镇 觉渡村 滕 刘金 23 号 无人 之机 被害人 杨 家中 盗走 现金 人民币 3000 余元 上午 被告人 贺 宁波市 镇海区 贵 驷 街道 镇 骆路 522 无人 之机 被害人 叶 家中 盗走 现金 人民币 2500 余元 上午 被告人 贺 镇海区 贵 驷 街道 沙河 村三 段落 307 号 11 无人 之机 被害人 林 家中 盗走 现金 人民币 2400 余元 被告人 贺 镇海区 九龙湖 镇长 宏村 抓获 上述 被盗 钱款 现均 已 追回",
    "被告人谌某某于2021年6月6日在本市海淀区被民警抓获归案，后如实供述了上述事实。涉案物品手机一部已依法扣押"])
    sents=tokenizer.decode(tokens)
    print(tokens)
    print(sents)


        




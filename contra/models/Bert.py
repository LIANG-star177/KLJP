import torch.nn as nn
import torch
from transformers import BertTokenizer, BertModel


class Bert(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=128, hid_dim=128, maps=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        # self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.hid_dim = 768  # bert base hidden dim
        # self.fc_article = nn.Linear(self.hid_dim, self.article_class_num)
        self.fc_charge = nn.Linear(self.hid_dim, self.charge_class_num)
        self.fc_k = nn.Linear(self.hid_dim, 1)

        self.dropout = nn.Dropout(0.4)

    def forward(self, data):
        text = data["justice"]      
        out = self.bert(**text) # 爆显存，dp也爆
        out_charge = self.fc_charge(out[1])
        out_k = self.fc_k(out[1])

        return {
            # "article": out_article,
            "charge": out_charge,
            "k": out_k
        }
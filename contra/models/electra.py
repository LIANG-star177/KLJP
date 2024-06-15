import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
from setting import CONTRA_WAY, CONTRASTIVE, DEVICE, PRETRAIN, LCM,ADD_DETAILS, ADD_ATTN

class Electra(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, maps=None, article_details=None, charge_details=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim

        self.electra = AutoModel.from_pretrained("electra-small") 

        if ADD_ATTN:
            self.embedding = nn.Embedding(vocab_size, emb_dim)
            self.transformer_enc = nn.TransformerEncoderLayer(emb_dim, nhead=8, batch_first=True)
            self.article_details = article_details["input_ids"].to(DEVICE)
            self.charge_details = charge_details["input_ids"].to(DEVICE)
            self.w2a=nn.Parameter(torch.zeros(hid_dim,hid_dim))
            self.w2c=nn.Parameter(torch.zeros(hid_dim,hid_dim))
            self.fc_article = nn.Linear(self.hid_dim*2, len(maps["article2idx"]))
            self.fc_charge = nn.Linear(self.hid_dim*2, len(maps["charge2idx"]))
        else:
            self.fc_article = nn.Linear(self.hid_dim, len(maps["article2idx"]))
            self.fc_charge = nn.Linear(self.hid_dim, len(maps["charge2idx"]))
        # self.fc_penalty = nn.Linear(self.hid_dim, 1)
        self.dropout = nn.Dropout(0.4)

    def enc(self, text):
        if  CONTRASTIVE and CONTRA_WAY == "supcon2":
            text = torch.cat([text[0], text[1]], dim=0)
        x = self.electra(text)
        out = x.last_hidden_state[:,0] # [256]
        return out

    def details_attn(self, details, enc_src, w):
        enc_details = self.embedding(details)  
        enc_details = self.transformer_enc(enc_details)
        enc_details, _ = torch.max(enc_details, dim=1)   
        alpha = nn.Softmax(dim=1)(torch.matmul(torch.matmul(enc_src, w), enc_details.T)) # [64, 512, 1]  
        context = torch.matmul(alpha, enc_details)
        return context
    
    def forward(self, data):
        fact_text = data["justice"]["input_ids"]
        enc_src = self.enc(fact_text)
        if ADD_DETAILS:
            char_context = self.details_attn(self.charge_details, enc_src, self.w2c)
            art_context = self.details_attn(self.article_details, enc_src, self.w2a)
            char_context = torch.cat((enc_src, char_context), dim=-1)
            art_context = torch.cat((enc_src, art_context), dim=-1)
        else:
            char_context = enc_src
            art_context = enc_src
        out_charge = self.fc_charge(char_context)
        out_article = self.fc_article(art_context)
        # out_penalty = self.fc_penalty(fact_emb)

        return {
            "article": out_article,
            "charge": out_charge,
            "char_enc": char_context,
            "art_enc": art_context
            # "penalty": out_penalty
        }

class LCM_DIST(nn.Module):
    def __init__(self, output_dim, hid_dim=512, wvdim=256) -> None:
        super().__init__()
        # label_encoder:
        self.label_emb = nn.Embedding(output_dim,wvdim) # (n,wvdim)
        self.label_fc = nn.Linear(wvdim, hid_dim)

        self.sim_fc = nn.Linear(output_dim, output_dim)

    def forward(self, embedded, labels):
        # print(text_emb.shape,'text')  # [16,64]
        label_emb = self.label_emb(labels)
        label_emb = F.tanh(self.label_fc(label_emb))
        # print(label_emb.shape,'label')  # [16,20,64]
        doc_product = torch.bmm(label_emb, embedded.unsqueeze(-1))  # (b,n,d) dot (b,d,1) --> (b,n,1)
        # print(doc_product.shape)   # [16,20,1]
        label_sim_dict = self.sim_fc(doc_product.squeeze(-1))
        #print(label_sim_dict.shape)
        return label_sim_dict

class ProjectionHead(nn.Module):
    def __init__(self, output_dim, output_art_dim, feat_dim=128):
        super(ProjectionHead, self).__init__()
        # label_encoder:
        self.head = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_dim, feat_dim)
            )
        self.head_art = nn.Sequential(
                nn.Linear(output_art_dim, output_art_dim),
                nn.ReLU(inplace=True),
                nn.Linear(output_art_dim, feat_dim)
            )

    def forward(self, output, output_art):
        output_cl = self.head(output)
        output_art_cl = self.head_art(output_art)
        #print(label_sim_dict.shape)
        return output_cl, output_art_cl
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
from setting import ADD_ATTN, BATCH_SIZE, CONTRA_WAY, CONTRASTIVE, DEVICE, LCM, ADD_DETAILS, ADD_CNN
from transformer import Transformer
from position_encoding import PostionalEncoding
import math
import torch.nn.functional as F
# from decoder import TransformerDecoder

class CNN_Encoder(nn.Module):
    def __init__(self, hid_dim=512, dropout=0.2, kernel_size=3, num_layers=5):
        super(CNN_Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.ModuleList([nn.Conv1d(hid_dim, hid_dim, kernel_size,
                                      padding=kernel_size // 2) for _ in range(num_layers)])
        self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)
    def forward(self, src):
        src = self.dropout(src)
        cnn = src.transpose(1, 2)
        for i, layer in enumerate(self.conv):
            cnn = F.tanh(layer(cnn)+cnn)        
        return cnn.transpose(1,2)

class Electra(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, max_length=512, maps=None, article_details=None, charge_details=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim
        self.max_length = max_length

        self.electra = AutoModel.from_pretrained("electra-small") 
        # for p in self.electra.parameters():
        #     p.requires_grad = False
        self.CNN = CNN_Encoder(hid_dim=hid_dim, dropout=0.2, kernel_size=3, num_layers=5)

        self.transformer_char_dec = Transformer(
                d_model=256, nhead=4, num_encoder_layers=1,num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=False,
                return_intermediate_dec=False, 
                rm_self_attn_dec=True, rm_first_self_attn=True,)
        
        self.transformer_art_dec = Transformer(
                d_model=256, nhead=4, num_encoder_layers=1,num_decoder_layers=2, dim_feedforward=2048, dropout=0.1,
                activation="relu", normalize_before=False,
                return_intermediate_dec=False, 
                rm_self_attn_dec=True, rm_first_self_attn=True,)
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # self.pos_embedding=nn.Embedding(max_length,hid_dim)
        self.pos_embedding = PostionalEncoding(hid_dim, max_len=max_length)
        self.transformer_enc = nn.TransformerEncoderLayer(emb_dim, nhead=8, batch_first=True)
        self.article_details = article_details["input_ids"].to(DEVICE)
        self.charge_details = charge_details["input_ids"].to(DEVICE)
        self.fill_char = nn.Parameter(torch.zeros(len(maps["charge2idx"])-self.charge_details.shape[0],hid_dim))

        self.w2a=nn.Parameter(torch.zeros(hid_dim,hid_dim))
        self.w2c=nn.Parameter(torch.zeros(hid_dim,hid_dim))
        self.fc_article = GroupWiseLinear(len(maps["article2idx"]), hid_dim, bias=True)
        self.fc_charge = GroupWiseLinear(len(maps["charge2idx"]), hid_dim, bias=True)
        self.label_emb_art = nn.Embedding(len(maps["article2idx"]), hid_dim)
        self.label_emb_char = nn.Embedding(len(maps["charge2idx"]), hid_dim)


    def enc(self, text):
        if  CONTRASTIVE and CONTRA_WAY == "supcon2":
            text = torch.cat([text[0], text[1]], dim=0)
        x = self.electra(text)
        out = x.last_hidden_state # [256]
        # text = self.embedding(text)
        # hidden = self.transformer_enc(text)  # [64, 512, 256]
        # out = nn.Tanh()(hidden)
        return out

    def details_attn(self, label_emb, another, enc_src, w):
        # seq_len = enc_src.shape[1]
        # enc_src, _ = torch.max(enc_src,dim=1)
        # enc_details = self.embedding(details)  
        # enc_details = self.transformer_enc(enc_details)
        # enc_details, _ = torch.max(enc_details, dim=1)  
        label_emb = label_emb[0] 
        alpha = nn.Softmax(dim=1)(torch.matmul(torch.matmul(enc_src, w), label_emb.T)) # [64, 512, 1]  
        context = torch.matmul(alpha, label_emb)

        # context=context.unsqueeze(1).repeat(1,seq_len,1)

        return context
    
    def details2label(self, details, batch_size, char_flag):
        # enc_details = self.enc(details)
        enc_details = self.embedding(details)  
        enc_details = self.transformer_enc(enc_details)
        enc_details, _ = torch.max(enc_details, dim=1)  
        if char_flag:
            enc_details = torch.cat((enc_details, self.fill_char), dim=0)
        label_emb = enc_details.unsqueeze(0).repeat(batch_size,1,1)
        return label_emb
    
    def label_enc(self, batch_size, char_flag):
        if char_flag:
            enc_details = self.label_emb_char.weight
        else:
            enc_details = self.label_emb_art.weight
        label_emb = enc_details.unsqueeze(0).repeat(batch_size,1,1)
        return label_emb
    
    def forward(self, data):
        fact_text = data["justice"]["input_ids"]
        batch_size=fact_text.shape[0]

        enc_src = self.enc(fact_text)
        if ADD_CNN:
            enc_src = self.CNN(enc_src)
        # char_context=char_context.unsqueeze(1).repeat(1,self.max_length,1) + enc_src
        # art_context=art_context.unsqueeze(1).repeat(1,self.max_length,1) + enc_src
        # char_context = torch.cat((enc_src, char_context), dim=-1)
        # art_context = torch.cat((enc_src, art_context), dim=-1)
        if ADD_DETAILS:
            char_label_emb = self.details2label(self.charge_details, batch_size, 1)
            art_label_emb = self.details2label(self.article_details, batch_size, 0)
        else:
            char_label_emb = self.label_enc(batch_size, char_flag =1)
            art_label_emb = self.label_enc(batch_size, char_flag=0)
        
        if ADD_ATTN:
            char_context = self.details_attn(label_emb=char_label_emb, another=art_label_emb, 
                                            enc_src = enc_src, w = self.w2c)
            art_context = self.details_attn(label_emb=art_label_emb, another=char_label_emb,
                                            enc_src = enc_src, w = self.w2a)
            char_context = char_context + enc_src   
            art_context = art_context + enc_src
        else:
            char_context, art_context = enc_src, enc_src
    
        # pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(DEVICE)
        # pos_emb = self.pos_embedding(pos).to(DEVICE)
        pos_emb = self.pos_embedding(fact_text).to(DEVICE)

        char_hs = self.transformer_char_dec(char_context, char_label_emb, pos_emb)[0] # B,K,d
        art_hs = self.transformer_art_dec(char_context, art_label_emb, pos_emb)[0]
        out_charge = self.fc_charge(char_hs[-1])
        out_article = self.fc_article(art_hs[-1])

        return {
            "article": out_article,
            "charge": out_charge,
            "char_enc": char_hs,
            "art_enc": art_hs
            # "penalty": out_penalty
        }
    
class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

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
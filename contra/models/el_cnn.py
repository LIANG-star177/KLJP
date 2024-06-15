import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
from transformer import Transformer
from position_encoding import PostionalEncoding
import math
import numpy as np
# from decoder import TransformerDecoder

class El_CNN(nn.Module):
    def __init__(self, vocab_size=5000, emb_dim=300, hid_dim=128, max_length=512, maps=None, article_details=None, charge_details=None) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.charge_class_num = len(maps["charge2idx"])
        self.article_class_num = len(maps["article2idx"])
        self.hid_dim = hid_dim
        self.max_length = max_length

        self.electra = AutoModel.from_pretrained("electra-small") 

        self.dropout = nn.Dropout(0.1)
        self.filter_sizes=[3, 4, 5]
        self.num_filters=[100, 100, 100]

        self.n_last_hidden = 2
        self.is_freeze = False
        if self.n_last_hidden:
            self.linear = nn.Linear(hid_dim*self.n_last_hidden,hid_dim)

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=hid_dim,
                      out_channels=self.num_filters[i],
                      kernel_size=self.filter_sizes[i])
            for i in range(len(self.filter_sizes))
        ])
        
        self.fc_article = nn.Linear(np.sum(self.num_filters), len(maps["article2idx"]))
        self.fc_charge = nn.Linear(np.sum(self.num_filters), len(maps["charge2idx"]))

    def details_attn(self, details, enc_src, w):
        # seq_len = enc_src.shape[1]
        # enc_src, _ = torch.max(enc_src,dim=1)
        enc_details = self.embedding(details)  
        enc_details = self.transformer_enc(enc_details)
        enc_details, _ = torch.max(enc_details, dim=1)   
        alpha = nn.Softmax(dim=1)(torch.matmul(torch.matmul(enc_src, w), enc_details.T)) # [64, 512, 1]  
        context = torch.matmul(alpha, enc_details)
        # context=context.unsqueeze(1).repeat(1,seq_len,1)
        return context
    
    def details2label(self, details, batch_size):
        enc_details = self.embedding(details)  
        enc_details = self.transformer_enc(enc_details)
        enc_details, _ = torch.max(enc_details, dim=1)   
        label_emb = enc_details.unsqueeze(0).repeat(batch_size,1,1)
        return label_emb
    
    def label_enc(self, batch_size, char_flag):
        if char_flag:
            enc_details = self.label_emb_char.weight
        else:
            enc_details = self.label_emb_art.weight
        label_emb = enc_details.unsqueeze(0).repeat(batch_size,1,1)
        return label_emb
    
    def cnn_out(self, outputs):
        if self.n_last_hidden == 1:
            hidden_states = outputs.last_hidden_state
            # hidden size : [batch x max_leng x hidden_him]
        else:
            hidden_states = torch.cat(outputs[1][-self.n_last_hidden:], dim = -1)
            hidden_states = self.linear(hidden_states)

        if self.is_freeze:
            hidden_states.requres_grad = False

        hidden_states = hidden_states.permute(0,2,1)
        x_conv_list = [F.relu(conv1d(hidden_states)) for conv1d in self.conv1d_list]
        x_pool_list =  [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        return x_fc

    def forward(self, data):
        fact_text = data["justice"]["input_ids"]
        batch_size=fact_text.shape[0]
        src_len=fact_text.shape[1]
        outputs = self.electra(fact_text, output_hidden_states=True)
        # char_context = self.details_attn(self.charge_details, enc_src, self.w2c)
        # art_context = self.details_attn(self.article_details, enc_src, self.w2a)
        # char_context = char_context + enc_src
        # art_context = art_context +enc_src
        x_fc = self.cnn_out(outputs)
        out_charge = self.fc_charge(self.dropout(x_fc))
        out_article = self.fc_article(self.dropout(x_fc))

        return {
            "article": out_article,
            "charge": out_charge,
        }
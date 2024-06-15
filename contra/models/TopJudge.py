import os
from setting import BATCH_SIZE, DEVICE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

from utils.loss import MultiLabelSoftmaxLoss, log_square_loss

class LSTMEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, num_layers, batch_size, sentence_num, sentence_len):
        super(LSTMEncoder, self).__init__()

        self.data_size = emb_dim
        self.hidden_dim = hid_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sentence_num =sentence_num
        self.sentence_len= sentence_len

        self.lstm_sentence = nn.LSTM(self.data_size, self.hidden_dim, batch_first=True,
                                     num_layers=num_layers)
        self.lstm_document = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True,
                                     num_layers=num_layers)
        self.feature_len = self.hidden_dim

    def init_hidden(self):
            self.sentence_hidden = (
                torch.autograd.Variable(
                    torch.zeros(self.num_layers,
                                self.batch_size * self.sentence_num,
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(self.num_layers,
                                self.batch_size * self.sentence_num,
                                self.hidden_dim).cuda()))
            self.document_hidden = (
                torch.autograd.Variable(
                    torch.zeros(self.num_layers, self.batch_size,
                                self.hidden_dim).cuda()),
                torch.autograd.Variable(
                    torch.zeros(self.num_layers, self.batch_size,
                                self.hidden_dim).cuda()))
    def forward(self, x):
        x = x.view(self.batch_size * self.sentence_num,
                   self.sentence_len,
                   self.data_size)

        sentence_out, self.sentence_hidden = self.lstm_sentence(x, self.sentence_hidden)
        temp_out = []
        # if config.get("net", "method") == "LAST":
        #     for a in range(0, len(sentence_out)):
        #         idx = a // self.sentence_num
        #         idy = a % self.sentence_num
        #         temp_out.append(sentence_out[a][doc_len[idx][idy + 2] - 1])
        #     sentence_out = torch.stack(temp_out)
        # elif config.get("net", "method") == "MAX":
        sentence_out = sentence_out.contiguous().view(
            self.batch_size, self.sentence_num,
            self.sentence_len,
            self.hidden_dim)
        sentence_out = torch.max(sentence_out, dim=2)[0]
        sentence_out = sentence_out.view(
            self.batch_size, self.sentence_num,
            self.hidden_dim)
    
        sentence_out = sentence_out.view(self.batch_size, self.sentence_num,
                                         self.hidden_dim)

        lstm_out, self.document_hidden = self.lstm_document(sentence_out, self.document_hidden)

        self.attention = lstm_out

        # if config.get("net", "method") == "LAST":
        #     outv = []
        #     for a in range(0, len(doc_len)):
        #         outv.append(lstm_out[a][doc_len[a][1] - 1])
        #     lstm_out = torch.cat(outv)
        # elif config.get("net", "method") == "MAX":
        lstm_out = torch.max(lstm_out, dim=1)[0]

        return lstm_out

class CNN_Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hid_dim=512, dropout=0.2, max_len=512,
                 kernel_size=3, num_layers=5):
        super(CNN_Encoder, self).__init__()
        self.position_embedding = nn.Embedding(max_len, emb_dim)
        self.word_embedding = nn.Embedding(vocab_size, emb_dim)
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.ModuleList([nn.Conv1d(hid_dim, hid_dim, kernel_size,
                                      padding=kernel_size // 2) for _ in range(num_layers)])
        self.scale=torch.sqrt(torch.FloatTensor([hid_dim])).to(DEVICE)
        
    def forward(self, src):
        batch_size=src.shape[0]
        src_len=src.shape[1]
        pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(DEVICE)
        
        # Retrieving position and word embeddings 
        position_embedding = self.position_embedding(pos)
        word_embedding = self.word_embedding(src)
        embedded = self.dropout(position_embedding + word_embedding*self.scale)
        cnn = embedded.transpose(1, 2)
        for i, layer in enumerate(self.conv):
            cnn = F.tanh(layer(cnn)+cnn)        
        return cnn.transpose(1,2)

class LSTMDecoder(nn.Module):
    def __init__(self, hid_dim):
        super(LSTMDecoder, self).__init__()
        self.feature_len = hid_dim # 768

        features = self.feature_len
        self.hidden_dim = features

        # self.task_name = ["ft", "zm", "xq"]
        self.task_name = ["charge", "article"]

        self.midfc = []
        for x in self.task_name:
            self.midfc.append(nn.Linear(features, features))

        self.cell_list = [None]
        for x in self.task_name:
            self.cell_list.append(nn.LSTMCell(self.feature_len, self.feature_len))

        self.hidden_state_fc_list = []
        for a in range(0, len(self.task_name)):
            arr = []
            for b in range(0, len(self.task_name)):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr)
            self.hidden_state_fc_list.append(arr)

        self.cell_state_fc_list = []

        for a in range(0, len(self.task_name) + 1):
            arr = []
            for b in range(0, len(self.task_name) + 1):
                arr.append(nn.Linear(features, features))
            arr = nn.ModuleList(arr) # arrï¿½ï¿½16ï¿½ï¿½nn.Linear
            self.cell_state_fc_list.append(arr) #4ï¿½ï¿½moduleList Ã¿ï¿½ï¿½modulelistï¿½ï¿½4ï¿½ï¿½nn.linear

        self.midfc = nn.ModuleList(self.midfc) # 3ï¿½ï¿½nn.Linear(768,768)
        self.cell_list = nn.ModuleList(self.cell_list)# 3ï¿½ï¿½LSTMCell(input_size=768, hidden_size=768)
        self.hidden_state_fc_list = nn.ModuleList(self.hidden_state_fc_list) # 4ï¿½ï¿½moduleListï¿½ï¿½Ã¿ï¿½ï¿½moduleListï¿½ï¿½4ï¿½ï¿½nn.Linear(768,768)
        self.cell_state_fc_list = nn.ModuleList(self.cell_state_fc_list) # 4ï¿½ï¿½moduleList Ã¿ï¿½ï¿½modulelistï¿½ï¿½4ï¿½ï¿½nn.linear(768, 768)

    def init_hidden(self, bs):
        self.hidden_list = [] # [(zeros(bs,hidden_dim), zeros(bs,hidden_dim))*4]
        for a in range(0, len(self.task_name) + 1):
            self.hidden_list.append((torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).to(DEVICE)),
                                     torch.autograd.Variable(torch.zeros(bs, self.hidden_dim).to(DEVICE)))) # appendÒ»ï¿½ï¿½Ôªï¿½ï¿½
    def generate_graph(self):
        s = "[(1 2)]"
        arr = s.replace("[", "").replace("]", "").split(",")
        graph = []
        n = 0
        if (s == "[]"):
            arr = []
            n = 3
        for a in range(0, len(arr)):
            arr[a] = arr[a].replace("(", "").replace(")", "").split(" ")
            arr[a][0] = int(arr[a][0])
            arr[a][1] = int(arr[a][1])
            n = max(n, max(arr[a][0], arr[a][1]))

        n += 1
        for a in range(0, n):
            graph.append([])
            for b in range(0, n):
                graph[a].append(False)

        for a in range(0, len(arr)):
            graph[arr[a][0]][arr[a][1]] = True

        return graph
    
    def forward(self, x): 
        fc_input = x #(batch_size, emb_size) emb_size = hidden_size
        outputs = {}
        batch_size = x.size()[0]
        self.init_hidden(batch_size)
        graph = self.generate_graph()

        first = [] # [True, True, True, True]
        for a in range(0, len(self.task_name) + 1):
            first.append(True)
        for a in range(1, len(self.task_name) + 1):
            h, c = self.cell_list[a](fc_input, self.hidden_list[a]) # (h, c), ((batch_size,hidden_size),(batch_size,hidden_size))
            for b in range(1, len(self.task_name) + 1):
                if graph[a][b]:
                    hp, cp = self.hidden_list[b]
                    if first[b]:
                        first[b] = False
                        hp, cp = h, c
                    else:
                        hp = hp + self.hidden_state_fc_list[a][b](h)
                        cp = cp + self.cell_state_fc_list[a][b](c)
                    self.hidden_list[b] = (hp, cp)
            outputs[self.task_name[a - 1]] = self.midfc[a - 1](h).view(batch_size, -1)

        return outputs # {"zm":(batch_size, hidden_size), "rzrf":(batch_size, hidden_size), "qzcs":(batch_size, hidden_size), "sfqs:(batch, hidden)"}

class LJPPredictor(nn.Module):
    def __init__(self, hid_dim, maps):
        super(LJPPredictor, self).__init__()

        self.hidden_size = hid_dim # 768
        self.zm_fc = nn.Linear(self.hidden_size, len(maps["charge2idx"])) # 836ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä¸ï¿½ï¿½ï¿½ 2ï¿½Ç·ï¿½Îªï¿½ï¿½ï¿½ï¿½/ï¿½ï¿½ï¿½ï¿½ to do ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
        self.ft_fc = nn.Linear(self.hidden_size, len(maps["article2idx"]))
        # self.article_fc = nn.Linear(self.hidden_size, 183 * 2) # 183ï¿½Ç·ï¿½ï¿½ï¿½ï¿½Ä¸ï¿½ï¿½ï¿½ 2ï¿½ï¿½ï¿½Ð¸Ã·ï¿½ï¿½ï¿½/ï¿½Þ¸Ã·ï¿½ï¿½ï¿½
        # self.term_fc = nn.Linear(self.hidden_size, 1)
        # self.rzrf_fc = nn.Linear(self.hidden_size, 2) # nn.Sequential(nn.Linear(self.hidden_size, 256), nn.Linear(256, 2)) # to do ï¿½Ó²ï¿½
        # self.qzcs_fc = nn.Linear(self.hidden_size, 2) # nn.Sequential(nn.Linear(self.hidden_size, 256), nn.Linear(256, 2)) 
        # self.sfqs_fc = nn.Linear(self.hidden_size, 2) # nn.Sequential(nn.Linear(self.hidden_size, 256), nn.Linear(256, 2)) 

    def forward(self, h):
        # charge = self.charge_fc(h)
        # article = self.article_fc(h)
        # term = self.term_fc(h)

        zm = self.zm_fc(h)
        ft = self.ft_fc(h)
        # rzrf = self.qzcs_fc(h)
        # qzcs = self.qzcs_fc(h)
        # sfqs = self.sfqs_fc(h)

        # batch = h.size()[0]
        # zm = zm.view(batch, -1, 2) # (batch_size, 836, 2)
        # rzrf = rzrf.view(batch, 2) #(batch_size, 2)
        # qzcs = qzcs.view(batch, 2) #(batch_size, 2)
        # sfqs = sfqs.view(batch, 2) #(batch_size, 2)
        return {"charge": zm, "article":ft}
        # return {"zm": zm, "rzrf": rzrf, "qzcs": qzcs, "sfqs":sfqs}
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(CNNEncoder, self).__init__()

        self.emb_dim = emb_dim
        self.output_dim = self.emb_dim // 4

        self.min_gram = 2
        self.max_gram = 5
        self.convs = []
        for a in range(self.min_gram, self.max_gram + 1):
            self.convs.append(nn.Conv2d(1, self.output_dim, (a, self.emb_dim)))

        self.convs = nn.ModuleList(self.convs)
        self.feature_len = self.emb_dim
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[0]

        x = x.view(batch_size, 1, -1, self.emb_dim)

        conv_out = []
        gram = self.min_gram
        for conv in self.convs:
            y = self.relu(conv(x))
            y = torch.max(y, dim=2)[0].view(batch_size, -1)

            conv_out.append(y)
            gram += 1

        conv_out = torch.cat(conv_out, dim=1)

        return conv_out

class TopJudge(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, maps):
        super(TopJudge, self).__init__()
        # self.encoder = LSTMEncoder(emb_dim, hid_dim, 1, BATCH_SIZE, 16, 128)
        self.encoder = CNNEncoder(emb_dim)
        # self.encoder = CNNEncoder(emb_dim).to(DEVICE)
        self.decoder = LSTMDecoder(hid_dim)
       
        self.fc = LJPPredictor(hid_dim, maps)
        self.dropout = nn.Dropout(0.1)

        self.filter_sizes=[3, 4, 5]
        self.num_filters=[100, 100, 100]

        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim,
                      out_channels=self.num_filters[i],
                      kernel_size=self.filter_sizes[i])
            for i in range(len(self.filter_sizes))
        ])

        kernels = (2, 3, 4)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, hid_dim,  kernel_size=i) for i in kernels])
        self.fc1 = nn.Linear(len(kernels) * hid_dim, hid_dim)
        # self.criterion = {
        #     "zm": MultiLabelSoftmaxLoss(config, 836),
        #     "rzrf": MultiLabelSoftmaxLoss(config, 1),
        #     "qzcs": MultiLabelSoftmaxLoss(config, 1),
        #     "sfqs": MultiLabelSoftmaxLoss(config, 1),
        #     # "xq": log_square_loss
        # }
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # self.embedding = nn.Embedding(len(json.load(open(config.get("data", "word2id")))),
        #                               config.getint("model", "hidden_size")) # emb_sizeï¿½ï¿½hidden_sizeÒ»ï¿½ï¿½

    def cnn_out(self, outputs):
        hidden_states = outputs
        hidden_states = hidden_states.permute(0,2,1)
        x_conv_list = [F.relu(conv1d(hidden_states)) for conv1d in self.conv1d_list]
        x_pool_list =  [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        return x_fc
    
    def conv_and_pool(self, x, conv):
        x = x.permute(0, 2, 1)
        x = torch.nn.ReLU()(conv(x))
        x = torch.nn.MaxPool1d(x.shape[-1])(x)
        return x

    def cnn_fix(self, outputs):
        res = [self.conv_and_pool(outputs, conv) for conv in self.convs]
        tot = torch.cat(res, dim=2)
        out = tot.flatten(start_dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = torch.nn.ReLU()(out)
        return out

    def forward(self, data): # x: collate_fn ï¿½ï¿½ï¿½Øµï¿½ï¿½Öµä£¬textï¿½ï¿½Ã¿ï¿½ï¿½valueï¿½Ä¶ï¿½ï¿½ï¿½batch_sizeï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Îªmax_lenï¿½ï¿½list, data = {'text': input, 'zm': charge, 'ft': article, 'xq': term}
        x = data["justice"]["input_ids"].to(DEVICE)
        x = self.embedding(x) # (batch_size, max_len, emb_size)
        # hidden = self.cnn_fix(x)
        hidden = self.cnn_out(x)
        # hidden = self.encoder(x) # (batch_size, emb_size) emb_size = hidden_size
        hidden = self.dropout(hidden) # (batch_size, emb_size) ï¿½ï¿½pï¿½Ä¸ï¿½ï¿½ï¿½Ê¹Ò»Ð©ÖµÎª0ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ô­ï¿½ï¿½ï¿½ï¿½hiddenï¿½ï¿½ï¿½ï¿½ï¿½ï¿½aï¿½ï¿½dropoutï¿½ï¿½hiddenï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Îª1-p*a+p*0=(1-p)*aï¿½ï¿½Îªï¿½Ë±ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Òªï¿½ï¿½Ã¿ï¿½ï¿½Ôªï¿½Ø³ï¿½ï¿½ï¿½1-p
        result = self.decoder(hidden) # {"zm":(batch_size, hidden_size), "rzrf":(batch_size, hidden_size), "qzcs":(batch_size, hidden_size), "sfqs:(batch, hidden)"}

        for name in result:
            result[name] = self.fc(self.dropout(result[name]))[name] # {"zm": (batch_size, 836, 2), "rzrf":(batch_size, 2), "qzcs": (batch_size, 2), "sfqs":(batch_size, 2)}
 
        return result

import collections
import json
import os
import random
import warnings
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from contrastive_loss import (AlignConLoss, SupConLoss, SupConLoss2,
                              contrastive_loss)
from dataloader2 import RANDOM_SEED, load_details, simple_load_multi_data
# from el_trans import Electra, ProjectionHead
from models.attention_xml import Hybrid_XML
from models.cnn_trans import CNN_Trans
from models.el_trans import Al_Trans, ProjectionHead
from models.electra import Electra
from models.LSTM_attn import LSTM_attn
from models.TextCNN import TextCNN
from models.TopJudge_tmp import TopJudge
from models.transformer_attn import transformer_attn
from setting import (BATCH_SIZE, CONTRA_WAY, CONTRASTIVE, DEV_FILE, DEVICE,
                     EMB_DIM, EMBEDDING_PATH, EPOCHS, HEAD, HID_DIM, LCM,
                     MODEL, PRETRAIN, SEQ_LEN, SMOOTH, TEST_FILE, TRAIN_FILE)
from sklearn.metrics import (classification_report, f1_score, jaccard_score,
                             multilabel_confusion_matrix,
                             precision_recall_fscore_support)
from tensorboardX import SummaryWriter
from tokenizer import MyTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

# from data_loader4ee import RANDOM_SEED,simple_load_multi_data
# import matplotlib.pyplot as plt
from utils.loss import MultiLabelSoftmaxLoss

warnings.filterwarnings("ignore")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_args():
    """设置训练模型所需参数"""
    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', default='0', type=str, help='设置训练或测试时使用的显卡')
    parser.add_argument('--seed', type=int, default=22, help='随机种子')
    return parser.parse_args()

class Trainer:
    def __init__(self,load_path=None):
        args = set_args()
        setup_seed(args.seed)
        article_details_path="data/details_gen/law.json"
        charge_details_path="data/details_gen/charge_details.json"
        self.art_details_len=128
        self.charge_details_len=64
        self.epochs = EPOCHS
        self.model_name=MODEL+"_kl_al1_single"

        self.train_set,self.maps=simple_load_multi_data(TRAIN_FILE,SEQ_LEN,EMBEDDING_PATH,text_clean=False)
        self.valid_set,_=simple_load_multi_data(DEV_FILE,SEQ_LEN,EMBEDDING_PATH,text_clean=False)
        self.test_set,_=simple_load_multi_data(TEST_FILE,SEQ_LEN,EMBEDDING_PATH,text_clean=False)
        self.train_iter=DataLoader(self.train_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
        self.dev_iter=DataLoader(self.valid_set,batch_size=BATCH_SIZE,shuffle=False,drop_last=False)
        self.test_iter=DataLoader(self.test_set,batch_size=BATCH_SIZE,shuffle=False,drop_last=False)
        
        if MODEL=="Bert":
            self.tokenizer_view=BertTokenizer.from_pretrained("bert-base-chinese")
        elif MODEL=="Electra" or MODEL=="Al_Trans":
            self.tokenizer_view=AutoTokenizer.from_pretrained("electra-small")
        else:
            self.tokenizer_view=MyTokenizer(EMBEDDING_PATH)
        
        self.model_save_path="logs/{}/".format(self.model_name)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        self.writer = SummaryWriter(log_dir=self.model_save_path)
        # self.tokenizer_view=BertTokenizer.from_pretrained("bert-base-chinese")
        
        article_details, charge_details=load_details(self.maps, article_details_path, charge_details_path, self.art_details_len, self.charge_details_len,EMBEDDING_PATH)
        if MODEL=="TextCNN":
            self.model=TextCNN(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=256,maps=self.maps, embedding_path=EMBEDDING_PATH).to(DEVICE)
        elif MODEL=="LSTM":
            self.model = LSTM_attn(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=256,maps=self.maps).to(DEVICE)
        elif MODEL=="Transformer":
            self.model = transformer_attn(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=256,maps=self.maps).to(DEVICE)
        elif MODEL=="TopJudge":
            self.model = TopJudge(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=300,maps=self.maps).to(DEVICE)
        elif MODEL=="Electra":
            self.model = Electra(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=256,maps=self.maps, article_details=article_details, charge_details=charge_details).to(DEVICE)
        elif MODEL=="Al_Trans":
            self.model = Al_Trans(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=256,maps=self.maps, article_details=article_details, charge_details=charge_details).to(DEVICE)
        elif MODEL=="CNN_Trans":
            self.model = CNN_Trans(self.tokenizer_view.vocab_size,emb_dim=256,hid_dim=256,maps=self.maps, article_details=article_details, charge_details=charge_details).to(DEVICE)
        elif MODEL=="Attention_XML":
            self.model = Hybrid_XML(vocab_size=self.tokenizer_view.vocab_size,embedding_size=256,hidden_size=256,maps=self.maps).to(DEVICE)
        if HEAD:
            self.head = ProjectionHead(output_dim=HID_DIM, output_art_dim=HID_DIM, feat_dim=1).to(DEVICE)
            for p in self.head.parameters():
                p.requires_grad = True
        # print(self.model)
        print(self.count_parameters(self.model))
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4)
        # ignored_params = list(map(id, self.model.electra.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters()) 
        # self.optimizer = torch.optim.Adam([{'params': base_params},
        #     {'params': self.model.electra.parameters(), 'lr': 1e-5}], 1e-4)

    def calculate_threshold2(self,pred,label):
        zero = torch.zeros_like(pred)
        one = torch.ones_like(pred)
        threshold = 0
        pred = torch.where(pred <= threshold, zero, pred)
        pred = torch.where(pred > threshold, one, pred)
        # label, pred = self.get_multi(label, pred)
        s_p, s_r, s_f, _ =precision_recall_fscore_support(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="micro")
        m_p, m_r, m_f, _ =precision_recall_fscore_support(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="macro")
        s_j=jaccard_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="micro")
        m_j=jaccard_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="macro")
        return s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, pred

    def adjust_learning_rate(self, optimizer, shrink_factor):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
        print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],)) 

    def trans_c2a_auto(self, c_logits, maps):
    #通过邻接矩阵转换坐标轴,先将id的字典转为邻接矩阵.
        mat = np.zeros(shape = (len(maps["charge2idx"]), len(maps["article2idx"])))
        for k, v in maps["c2a"].items():
            mat[k][v] = 1
        # c_tmp = c_logits.clone()
        a_tmp = torch.matmul(c_logits, torch.Tensor(mat).to(DEVICE))
        return a_tmp
    
    def trans_a2c_auto(self, a_logits, maps):
    #通过邻接矩阵转换坐标轴,先将id的字典转为邻接矩阵.
        mat = np.zeros(shape = (len(maps["article2idx"]), len(maps["charge2idx"])))
        for k, vs in maps["a2c"].items():
            for v in vs:
                mat[k][v] = 1
        # a_tmp = a_logits.clone()
        c_tmp = torch.matmul(a_logits, torch.Tensor(mat).to(DEVICE))
        return c_tmp
    
    def supcon2_loss(self, features, trg):
        bsz = trg.shape[0]
        features = F.normalize(features, p=2, dim=1)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        # 有监督
        loss = SupConLoss2(temperature=0.07)(features, trg)
        # 无监督
        # loss = SupConLoss2(temperature=0.07)(features)
        return loss

    def contra_loss(self, output, output_art, trg, trg_art):
        if CONTRA_WAY=="supcon2":
            # output, output_art = self.head(output, output_art)
            # a_tmp = self.trans_c2a_auto(output, self.maps)
            # c_tmp = self.trans_a2c_auto(output_art, self.maps)
            # loss_al = 0.1 * self.supcon2_loss(c_tmp, trg)
            # loss_al_art = 0.1 * self.supcon2_loss(a_tmp, trg_art)
            loss = 0.1 * self.supcon2_loss(output, trg)
            loss_art = 0.1 * self.supcon2_loss(output_art, trg_art)
        elif CONTRA_WAY=="supcon":
            # a_tmp = self.trans_c2a_auto(output, self.maps)
            # c_tmp = self.trans_a2c_auto(output_art, self.maps)
            # trg_tmp, c_tmp = self.get_label(trg, c_tmp, 0)
            # trg_art_tmp, a_tmp = self.get_label(trg_art, a_tmp, 1)
            # loss_al = SupConLoss(temperature=0.07)(c_tmp, trg)
            # loss_al_art = SupConLoss(temperature=0.07)(a_tmp, trg_art)

            # trg_tmp, output_tmp = self.get_label(trg, output, 0)
            # trg_art_tmp, output_art_tmp = self.get_label(trg_art, output_art, 1)
            loss = SupConLoss(temperature=0.07)(output, trg)
            loss_art = SupConLoss(temperature=0.07)(output_art, trg_art)
        return loss, loss_art
    

    def multilabelloss():
        return  torch.nn.MultiLabelSoftMarginLoss()
    def multilabel_marginloss():
        return torch.nn.MultiLabelMarginLoss()
    def bceloss():
        return torch.nn.BCELoss()

    def criterion(self, out, label):
        return f1_score(out.cpu().argmax(1), label.cpu(), average='micro')

    def get_label_logits(self, logits):
        zero = torch.zeros_like(logits)
        one = torch.ones_like(logits)
        threshold = 0
        mask = torch.where(logits <= threshold, zero, logits)
        mask = torch.where(logits > threshold, one, mask)
        logits = logits.masked_fill(mask == 0, -1e2)   
        return logits

    def contra_al_loss(self, output, output_art):
        output = output[-1].permute(0,2,1)
        output_art = output_art[-1].permute(0,2,1)

        a_tmp = self.trans_c2a_auto(output, self.maps)
        c_tmp = self.trans_a2c_auto(output_art, self.maps)

        output = output.permute(2,0,1)
        output_art = output_art.permute(2,0,1)
        a_tmp = a_tmp.permute(2,0,1)
        c_tmp = c_tmp.permute(2,0,1)
        loss_al_c = torch.tensor(0).float().to(DEVICE)
        loss_al_a = torch.tensor(0).float().to(DEVICE)
        for i in range(len(output)):
            loss_al_c+=contrastive_loss(output[i], c_tmp[i])
        loss_al_c = loss_al_c/len(output)
        for i in range(len(output_art)):
            loss_al_a+=contrastive_loss(output_art[i], a_tmp[i])
        loss_al_a = loss_al_a/len(output)
        return loss_al_c, loss_al_a

    def compute_kl_loss(self,p, q, pad_mask=None):
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)
        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        loss = (p_loss + q_loss) / 2
        return loss

    def al_loss(self, output, output_art):
        if HEAD:
            output = output[-1]
            output_art = output_art[-1]
            output, output_art = self.head(output, output_art)
            output = output.squeeze(2)
            output_art = output_art.squeeze(2)
        a_tmp = self.trans_c2a_auto(output, self.maps)
        c_tmp = self.trans_a2c_auto(output_art, self.maps)
        # output = self.get_label_logits(output)
        # output_art = self.get_label_logits(output_art)

        if "contra" in self.model_name:
            loss_al_c = contrastive_loss(output, c_tmp)
            loss_al_a = contrastive_loss(output_art, a_tmp)
            losskl = 0.0001*(loss_al_c + loss_al_a)
        if "kl" in self.model_name:
            a_tmp = nn.LogSoftmax(dim=-1)(a_tmp)
            c_tmp = nn.LogSoftmax(dim=-1)(c_tmp)
            output = nn.Softmax(dim=-1)(output)
            output_art = nn.Softmax(dim=-1)(output_art)
            loss_al_c = nn.KLDivLoss()(c_tmp, output)
            loss_al_a = nn.KLDivLoss()(a_tmp, output_art)
            # loss_al_c=self.compute_kl_loss(c_tmp, output)
            # loss_al_a=self.compute_kl_loss(a_tmp, output_art)
            losskl = 0.5*(loss_al_c + loss_al_a)
        if "cos" in self.model_name:
            loss_al_c = -(nn.CosineSimilarity()(c_tmp, output)-1).mean()
            loss_al_a = -(nn.CosineSimilarity()(a_tmp, output_art)-1).mean()
            losskl = 0.5*(loss_al_c + loss_al_a)
        return losskl
    
    def al_sv_loss(self, output, output_art, trg, trg_art):
        # 求和后总体对齐
        a_tmp = self.trans_c2a_auto(output, self.maps)
        c_tmp = self.trans_a2c_auto(output_art, self.maps)
        loss_al_c = nn.BCEWithLogitsLoss()(c_tmp.float(),trg.float())
        loss_al_a = nn.BCEWithLogitsLoss()(a_tmp.float(),trg_art.float())
        return loss_al_c, loss_al_a
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self):
        criteria2=nn.BCEWithLogitsLoss()
        loss_vals = []
        loss_vals_eval = []
        art_loss_vals = []
        art_loss_vals_eval = []
        best_score=0        
        for epoch in range(self.epochs):
            epoch_loss= []
            train_cnt=0
            art_epoch_loss= []
            print("-"*10+"training"+"-"*10)
            tq = tqdm(self.train_iter)
            for data in tq:
                train_cnt+=1
                for k in data:
                    if type(data[k]) is dict:
                        for k2 in data[k]:
                            if k2 == "input_ids" and CONTRASTIVE and CONTRA_WAY=="supcon2":
                                data[k][k2][0] = data[k][k2][0].to(DEVICE)
                                data[k][k2][1] = data[k][k2][1].to(DEVICE)
                            else:
                                data[k][k2] = data[k][k2].to(DEVICE)
                    else:
                        data[k] = data[k].to(DEVICE)
            
                self.optimizer.zero_grad()
                out=self.model(data)

                if CONTRASTIVE and CONTRA_WAY == "supcon2":
                    # 平均两个输出求loss
                    double_out_char = out["charge"]
                    double_out_art = out["article"]
                    out["charge"] = torch.add(*torch.split(out["charge"], [BATCH_SIZE, BATCH_SIZE], dim=0))/2
                    out["article"] = torch.add(*torch.split(out["article"], [BATCH_SIZE, BATCH_SIZE], dim=0))/2

                charge=data["charge"].to(DEVICE)
                article=data["article"].to(DEVICE)

                loss=criteria2(out["charge"].float(),charge.float())
                art_loss=criteria2(out["article"].float(),article.float())

                if CONTRASTIVE:
                    # 输出logits对比
                    if CONTRA_WAY == "supcon2":
                        con_loss, con_loss_art = self.contra_loss(double_out_char, double_out_art, data["charge"], data["article"])
                        # con_loss, con_loss_art = self.contra_loss(out["char_enc"], out["art_enc"], data["charge"], data["article"])
                    else:
                        con_loss, con_loss_art = self.contra_loss(out["out_char"], out["out_art"], data["charge"], data["article"])
                    # encoder_output对比
                    # enc_out = self.model.encoder(data["justice"])
                    # con_loss, con_loss_art = self.contra_loss(enc_out, enc_out, data["charge"], data["article"])
                    loss = loss + con_loss
                    art_loss = art_loss + con_loss_art
                
                if "al1" in self.model_name:
                    #无监督对齐
                    if HEAD:
                        losskl = self.al_loss(out["char_enc"], out["art_enc"])
                    else:
                        losskl = self.al_loss(out["charge"], out["article"])
                    loss = loss + losskl

                #转换charge预测分布到article，和真实article用BCE对齐,反之相同
                if "al2" in self.model_name:
                    #有监督对齐
                    loss_al_c, loss_al_a = self.al_sv_loss(out["charge"], out["article"], charge, article)
                    loss_al_c *= 0.1
                    loss_al_a *= 0.1
                    loss = loss+  loss_al_c
                    art_loss = art_loss +  loss_al_a

                loss+=art_loss
                s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, _=self.calculate_threshold2(out["charge"],charge)
                s_p_art, s_r_art, s_f_art, s_j_art,m_p_art, m_r_art, m_f_art, m_j_art, _=self.calculate_threshold2(out["article"],article)

                loss.backward()
                epoch_loss.append(loss.item())
                art_epoch_loss.append(art_loss.item())
                #4表示保留四位小数，detach()将loss从计算图里抽离出来
                if CONTRASTIVE:
                    tq.set_postfix(epoch=epoch,train_loss=np.around(loss.cpu().detach().numpy(),4),con_loss=np.around(con_loss.cpu().detach().numpy(),4),train_mf=m_f)
                elif "al" in self.model_name:
                    tq.set_postfix(epoch=epoch,train_loss=np.around(loss.cpu().detach().numpy(),4),losskl=np.around(losskl.cpu().detach().numpy(),4),train_mf=m_f)
                else:
                    tq.set_postfix(epoch=epoch,train_loss=np.around(loss.cpu().detach().numpy(),4),train_art_loss=np.around(art_loss.cpu().detach().numpy(),4),train_mf=m_f,train_art_mf=m_f_art)
                self.optimizer.step()
                global_iter = epoch * len(self.train_iter) + train_cnt
                if train_cnt%50==0:
                    self.writer.add_scalar('train_f1', 100 * m_f, global_iter)
                    self.writer.add_scalar('loss', loss.item(), global_iter)
                    self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_iter)

            loss_vals.append(np.mean(epoch_loss))
            art_loss_vals.append(np.mean(art_epoch_loss))

            #一个epoch输出一次validation的结果
            print("-"*10+"validating"+"-"*10)
            dev_out=[]
            valid_cnt = 0
            art_dev_out=[]
            tq = tqdm(self.dev_iter)
            epoch_loss_valid= []
            art_epoch_loss_valid= []
            for data in tq:
                valid_cnt+=1
                for k in data:
                    if type(data[k]) is dict:
                        for k2 in data[k]:
                            if k2 == "input_ids" and CONTRASTIVE and CONTRA_WAY=="supcon2":
                                data[k][k2][0] = data[k][k2][0].to(DEVICE)
                                data[k][k2][1] = data[k][k2][1].to(DEVICE)
                            else:
                                data[k][k2] = data[k][k2].to(DEVICE)
                    else:
                        data[k] = data[k].to(DEVICE)
                with torch.no_grad():
                    out=self.model(data)
                
                if CONTRASTIVE and CONTRA_WAY == "supcon2":
                    # 平均两个输出求loss
                    double_out_char = out["charge"]
                    double_out_art = out["article"]
                    out["charge"] = torch.add(*torch.split(out["charge"], [data["charge"].shape[0], data["charge"].shape[0]], dim=0))/2
                    out["article"] = torch.add(*torch.split(out["article"], [data["charge"].shape[0], data["charge"].shape[0]], dim=0))/2

                charge=data["charge"].to(DEVICE)
                article=data["article"].to(DEVICE)
                loss=criteria2(out["charge"].float(),charge.float())
                art_loss=criteria2(out["article"].float(),article.float())
                loss+=art_loss

                s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, _=self.calculate_threshold2(out["charge"],charge)
                s_p_art, s_r_art, s_f_art, s_j_art,m_p_art, m_r_art, m_f_art, m_j_art, _=self.calculate_threshold2(out["article"],article)
                dev_out.append((out["charge"],charge))
                art_dev_out.append((out["article"],article))

                epoch_loss_valid.append(loss.item())
                art_epoch_loss_valid.append(art_loss.item())
                # tq.set_postfix(epoch=epoch,train_k_loss=np.around(k_loss.cpu().detach().numpy(),4),train_loss=np.around(loss.cpu().detach().numpy(),4),train_precison=precision,train_recall=recall,train_F1=F1)
                tq.set_postfix(epoch=epoch,valid_loss=np.around(loss.cpu().detach().numpy(),4),valid_sp=s_p, valid_sr=s_r,valid_sf=s_f)
                val_iter = epoch * len(self.dev_iter) + valid_cnt
                if valid_cnt%50==0:
                    self.writer.add_scalar('val_loss', loss.item(), val_iter)

            loss_vals_eval.append(np.mean(epoch_loss_valid))
            art_loss_vals_eval.append(np.mean(art_epoch_loss_valid))
            pred=torch.cat([i[0] for i in dev_out])
            truth=torch.cat([i[1] for i in dev_out])
            art_pred=torch.cat([i[0] for i in art_dev_out])
            art_truth=torch.cat([i[1] for i in art_dev_out])
            
            s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, _=self.calculate_threshold2(pred,truth)
            s_p_art, s_r_art, s_f_art, s_j_art,m_p_art, m_r_art, m_f_art, m_j_art, _=self.calculate_threshold2(art_pred,art_truth)

            print("*"*10+"charge_micro"+"*"*10)
            print("valid_precision:{0:.4f},valid_recall:{1:.4f},valid_F1:{2:.4f},valid_jaccard:{3:.4f}".format(s_p, s_r, s_f, s_j))

            print("*"*10+"charge_macro"+"*"*10)
            print("valid_precision:{0:.4f},valid_recall:{1:.4f},valid_F1:{2:.4f},valid_jaccard:{3:.4f}".format(m_p, m_r, m_f, m_j))

            print("*"*10+"article_micro"+"*"*10)
            print("valid_precision:{0:.4f},valid_recall:{1:.4f},valid_F1:{2:.4f},valid_jaccard:{3:.4f}".format(s_p_art, s_r_art, s_f_art, s_j_art))

            print("*"*10+"article_macro"+"*"*10)
            print("valid_precision:{0:.4f},valid_recall:{1:.4f},valid_F1:{2:.4f},valid_jaccard:{3:.4f}".format(m_p_art, m_r_art, m_f_art, m_j_art))

            self.writer.add_scalar('micro_f1', s_f * 100, epoch)
            self.writer.add_scalar('macro_f1', m_f * 100, epoch)
            #在性能提升的情况下保存当前epoch模型,将每个epoch的validation结果输入
            save_path=self.model_save_path+"model_{}".format(epoch)
            # with open("logs/{}/validation.txt".format(self.model_name),"a") as f:
            #     f.write(str(epoch)+" epoch"+"\n")
            #     f.write("valid f1: "+str(valid_F1)+"\n")

            if m_f+m_f_art>=best_score:
                best_score=m_f+m_f_art
                best_model_path=save_path
                torch.save(self.model,best_model_path)
            # else:
            #     self.adjust_learning_rate(self.optimizer, 0.5)
                # self.adjust_scheduler(self.scheduler, 0.5)
            # self.evaluate(model=self.model)
        
        self.evaluate(save_path=best_model_path)

    def decode(self, maps, token_ids):
        sent=[]
        for id in token_ids:
            id=id.item()
            sent.append(maps[str(id)])
        sent=" ".join(sent)
        return sent
    
    def save_decode(self, maps, truth, pred, filename):
        new_truth, new_pred=[],[]
        for i in tqdm(range(len(truth))):
            new_truth.append(self.decode(maps, np.nonzero(truth[i]).squeeze(1)))
            new_pred.append(self.decode(maps, np.nonzero(pred[i]).squeeze(1)))
        with open(filename,"a") as f:
            for i in range(len(new_truth)):
                f.write("truth: "+str(new_truth[i]).replace("[SOS]","")
                        +" pred: "+str(new_pred[i]).replace("[SOS]","")+"\n")

    def get_multi(self, trg, pred):
        multi_trg, multi_pred=[],[]
        trg = trg.tolist()
        pred = pred.tolist()
        for i in range(len(trg)):
            if trg[i].count(1)>1:
                multi_trg.append(trg[i])
                multi_pred.append(pred[i])
        return torch.LongTensor(multi_trg), torch.LongTensor(multi_pred)
    
    def get_sing(self, trg, pred):
        multi_trg, multi_pred=[],[]
        trg = trg.tolist()
        pred = pred.tolist()
        for i in range(len(trg)):
            if trg[i].count(1)==1:
                multi_trg.append(trg[i])
                multi_pred.append(pred[i].tolist())
        return torch.LongTensor(multi_trg), torch.LongTensor(multi_pred)
    
    def get_idx(self, pred):
        pred_idx = []
        for line in pred:
            cur_idx = []
            for i in range(len(line)):
                if line[i] == 1:
                    cur_idx.append(i)
            pred_idx.append(cur_idx)
        return pred_idx

    def select_articles(self, ar_lst, true_idx, trg, pred):
        top_pred = []
        top_true = []
        for i, line in enumerate(true_idx):
            for aid in line:
                if aid in ar_lst:
                    top_pred.append(pred[i])
                    top_true.append(trg[i])
                    break
        return top_pred, top_true

    def calculate_top_k(self, pred, trg, k, maps, flag="max"):
        zero = torch.zeros_like(pred)
        one = torch.ones_like(pred)
        threshold = 0
        pred = torch.where(pred <= threshold, zero, pred)
        pred = torch.where(pred > threshold, one, pred)
        trg = trg.tolist()
        pred = pred.tolist()

        trg_idx = self.get_idx(trg)
        ar_flat = [a for lst in trg_idx for a in lst]
        ar_flat = pd.Series(ar_flat)
        lst = ar_flat.value_counts()
        top_articles = lst[:k].index if flag=="max" else lst[-k:].index
        top_pred, top_true = self.select_articles(top_articles, trg_idx, trg, pred)
        print(len(top_true)/len(trg))
        print(multilabel_confusion_matrix(top_true, top_pred, labels=top_articles))

        output_dict = classification_report(top_true, top_pred, output_dict = True)
        p_all, r_all, f1_all, cnt = 0,0,0,0
        for k, v in output_dict.items():
            if k.isdigit():
                if int(k) in top_articles:
                    p_all+=v["precision"]
                    r_all+=v["recall"]
                    f1_all+=v["f1-score"]
                    cnt+=1
        ma_jac = jaccard_score(top_true, top_pred, average = "macro")
        return p_all/cnt, r_all/cnt, f1_all/cnt, ma_jac

    def evaluate(self,save_path=None,model=None):
        print("-"*10+"testing"+"-"*10)
        test_out=[]
        art_test_out=[]
        sentences_id = []
        if save_path is not None:
            model=torch.load(save_path).to(DEVICE)
        bat_cnt=0
        for data in tqdm(self.test_iter):
            bat_cnt+=1
            for k in data:
                if type(data[k]) is dict:
                    for k2 in data[k]:
                        if k2 == "input_ids" and CONTRASTIVE and CONTRA_WAY=="supcon2":
                            data[k][k2][0] = data[k][k2][0].to(DEVICE)
                            data[k][k2][1] = data[k][k2][1].to(DEVICE)
                        else:
                            data[k][k2] = data[k][k2].to(DEVICE) 
                else:
                    data[k] = data[k].to(DEVICE)
            with torch.no_grad():
                out=model(data)
            if CONTRASTIVE and CONTRA_WAY == "supcon2":
                # 平均两个输出求loss
                double_out_char = out["charge"]
                double_out_art = out["article"]
                out["charge"] = torch.add(*torch.split(out["charge"], [data["charge"].shape[0], data["charge"].shape[0]], dim=0))/2
                out["article"] = torch.add(*torch.split(out["article"], [data["charge"].shape[0], data["charge"].shape[0]], dim=0))/2

            charge=data["charge"].to(DEVICE)
            article=data["article"].to(DEVICE)
            test_out.append((out["charge"],charge))
            art_test_out.append((out["article"],article))
            sentences_id += data["justice"]["input_ids"].tolist()

        pred=torch.cat([i[0] for i in test_out])
        truth=torch.cat([i[1] for i in test_out])
        art_pred=torch.cat([i[0] for i in art_test_out])
        art_truth=torch.cat([i[1] for i in art_test_out])
        
        s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, pred_c=self.calculate_threshold2(pred,truth)
        s_p_art, s_r_art, s_f_art, s_j_art,m_p_art, m_r_art, m_f_art, m_j_art, pred_a=self.calculate_threshold2(art_pred,art_truth)

        # dic_ont_hot = {"pred_c":pred_c.tolist(), "truth_c":truth.tolist(), "pred_a":pred_a.tolist(), "truth_a":art_truth.tolist()}
        # with open("notebook/one_hot_result/{}.json".format(MODEL), "w") as f:
        #     json.dump(dic_ont_hot, f, ensure_ascii=False)
        # sentences = []
        # for i in range(len(pred_c)):
        #     sentences.append(self.tokenizer_view.decode(sentences_id[i]).replace("[PAD]","").replace(" ",""))
        dic_ont_hot = {"pred_c":pred_c.tolist(), "truth_c":truth.tolist(), "pred_a":pred_a.tolist(), "truth_a":art_truth.tolist()}
        with open("notebook/one_hot_result/{}.json".format(self.model_name), "w") as f:
            json.dump(dic_ont_hot, f, ensure_ascii=False)


        # if save_path is not None: 
        #     self.save_decode(self.maps["idx2charge"], truth, pred_c, "logs/{}/pred_charge1.txt".format(self.model_name))
        #     self.save_decode(self.maps["idx2article"], art_truth, pred_a, "logs/{}/pred_article1.txt".format(self.model_name))
        print("*"*10+"charge_micro"+"*"*10)
        print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(s_p, s_r, s_f, s_j))

        print("*"*10+"charge_macro"+"*"*10)
        print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(m_p, m_r, m_f, m_j))

        print("*"*10+"article_micro"+"*"*10)
        print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(s_p_art, s_r_art, s_f_art, s_j_art))

        print("*"*10+"article_macro"+"*"*10)
        print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(m_p_art, m_r_art, m_f_art, m_j_art))
        
        # print("=="*20)
        # top_p, top_r, top_f, top_j = self.calculate_top_k(pred,truth, k=10, maps=self.maps["idx2charge"], flag="max")
        # top_p_art, top_r_art, top_f_art,top_j_art = self.calculate_top_k(art_pred,art_truth,k=10, maps=self.maps["idx2article"], flag="max")
        # print("*"*10+"charge_max10_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p, top_r, top_f, top_j))
        # print("*"*10+"article_max10_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p_art, top_r_art, top_f_art,top_j_art))

        # top_p, top_r, top_f, top_j = self.calculate_top_k(pred,truth, k=10, maps=self.maps["idx2charge"], flag="min")
        # top_p_art, top_r_art, top_f_art,top_j_art = self.calculate_top_k(art_pred,art_truth,k=10, maps=self.maps["idx2article"], flag="min")
        # print("*"*10+"charge_min10_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p, top_r, top_f, top_j))
        # print("*"*10+"article_min10_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p_art, top_r_art, top_f_art,top_j_art))

        # top_p, top_r, top_f, top_j = self.calculate_top_k(pred,truth, k=15, maps=self.maps["idx2charge"], flag="min")
        # top_p_art, top_r_art, top_f_art,top_j_art = self.calculate_top_k(art_pred,art_truth,k=15, maps=self.maps["idx2article"], flag="min")
        # print("*"*10+"charge_min15_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p, top_r, top_f, top_j))
        # print("*"*10+"article_min15_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p_art, top_r_art, top_f_art,top_j_art))

        # top_p, top_r, top_f, top_j = self.calculate_top_k(pred,truth, k=20, maps=self.maps["idx2charge"], flag="min")
        # top_p_art, top_r_art, top_f_art,top_j_art = self.calculate_top_k(art_pred,art_truth,k=20, maps=self.maps["idx2article"], flag="min")
        # print("*"*10+"charge_min20_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p, top_r, top_f, top_j))
        # print("*"*10+"article_min20_macro"+"*"*10)
        # print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(top_p_art, top_r_art, top_f_art,top_j_art))

if __name__=="__main__":
    trainer= Trainer()
    trainer.train()
    # trainer.evaluate("logs/q2l_2d_kl_al1/model_15")


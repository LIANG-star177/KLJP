import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, jaccard_score
import numpy as np
import os
import random
import collections
import json
from dataloader2 import RANDOM_SEED,simple_load_multi_data, load_details
from setting import TRAIN_FILE,DEV_FILE, TEST_FILE, EMBEDDING_PATH, SEQ_LEN, DEVICE, EMB_DIM, HID_DIM, BATCH_SIZE, EPOCHS
from setting import LCM, SMOOTH, HEAD, CONTRASTIVE, CONTRA_WAY, MODEL, PRETRAIN
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 22

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(RANDOM_SEED)

def calculate_threshold(pred,label):
    s_p, s_r, s_f, _ =precision_recall_fscore_support(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="micro")
    m_p, m_r, m_f, _ =precision_recall_fscore_support(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="macro")
    s_j=jaccard_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="micro")
    m_j=jaccard_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="macro")
    return [s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, pred]


def calculate_threshold2(pred,label):
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
    return [s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, pred]
     


def one_hot_labels(labels_index, arg_map):
    label=[0]*len(arg_map)
    for item in labels_index:
        label[item] = 1
    return label

def transform(maps, pred, flag):
    ans=[]
    for i in tqdm(range(len(pred))):
        tmp=np.nonzero(pred[i]).squeeze(1)
        if flag=="char":
            tmp=[maps["c2a"][el.item()] for el in tmp]
            tmp=one_hot_labels(tmp,maps["article2idx"])
        else:
            lst=[]
            for el in tmp:
                lst+=maps["a2c"][el.item()]
            tmp=one_hot_labels(lst,maps["charge2idx"])
        ans.append(tmp)
    return ans

test_set,_=simple_load_multi_data(TEST_FILE,SEQ_LEN,EMBEDDING_PATH,text_clean=False)
test_iter=DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=False,drop_last=False)

def evaluate(maps, save_path=None,model=None):
    print("-"*10+"al_testing"+"-"*10)
    test_out=[]
    art_test_out=[]
    if save_path is not None:
        model=torch.load(save_path).to(DEVICE)
    bat_cnt=0
    for data in tqdm(test_iter):
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
        charge=data["charge"].to(DEVICE)
        article=data["article"].to(DEVICE)
        test_out.append((out["charge"],charge))
        art_test_out.append((out["article"],article))

    pred=torch.cat([i[0] for i in test_out])
    truth=torch.cat([i[1] for i in test_out])
    art_pred=torch.cat([i[0] for i in art_test_out])
    art_truth=torch.cat([i[1] for i in art_test_out])
    
    metric_lst=calculate_threshold2(pred,truth)
    metric_art_lst=calculate_threshold2(art_pred,art_truth)

    facke_pred_a = transform(maps, metric_lst[-1], flag="char")
    facke_pred_c = transform(maps, metric_art_lst[-1], flag="art")

    metric_lst_2=calculate_threshold(torch.Tensor(facke_pred_c),truth)
    metric_art_lst_2=calculate_threshold(torch.Tensor(facke_pred_a),art_truth)
    dic_ont_hot = {"pred_c":metric_lst_2[-1].tolist(), "truth_c":truth.tolist(), "pred_a":metric_art_lst_2[-1].tolist(), "truth_a":art_truth.tolist()}
    with open("notebook/one_hot_result/mapping.json", "w") as f:
        json.dump(dic_ont_hot, f, ensure_ascii=False)

    print("*"*10+"charge_micro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_lst[0], metric_lst[1], metric_lst[2], metric_lst[3]))
    print("*"*10+"charge_macro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_lst[4], metric_lst[5], metric_lst[6], metric_lst[7]))
    print("*"*10+"article_micro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_art_lst[0], metric_art_lst[1], metric_art_lst[2], metric_art_lst[3]))
    print("*"*10+"article_macro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_art_lst[4], metric_art_lst[5], metric_art_lst[6], metric_art_lst[7]))

    print("*"*10+"charge_micro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_lst_2[0], metric_lst_2[1], metric_lst_2[2], metric_lst_2[3]))
    print("*"*10+"charge_macro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_lst_2[4], metric_lst_2[5], metric_lst_2[6], metric_lst_2[7]))
    print("*"*10+"article_micro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_art_lst_2[0], metric_art_lst_2[1], metric_art_lst_2[2], metric_art_lst_2[3]))
    print("*"*10+"article_macro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(metric_art_lst_2[4], metric_art_lst_2[5], metric_art_lst_2[6], metric_art_lst_2[7]))
maps = {}
c2i_path = "data/label2id_clean/c2i_clean.json"
a2i_path = "data/label2id_clean/a2i_clean.json"
with open(c2i_path) as f:
    c2i = json.load(f)
    maps["charge2idx"] = c2i
    maps["idx2charge"] = {v: k for k, v in c2i.items()}

with open(a2i_path) as f:
    a2i = json.load(f)
    maps["article2idx"] = a2i
    maps["idx2article"] = {v: k for k, v in a2i.items()}
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
evaluate(maps, "logs/q2l_2d_kl_al1/model_24")
import torch

# EMBEDDING_PATH = "gensim_train/word2vec.model"
# TRAIN_FILE = 'data/811clean/train.json'  # 训练集
# DEV_FILE = 'data/811clean/valid.json'  # 验证集
# TEST_FILE = 'data/811clean/test.json'  # 测试文件

# # multi
EMBEDDING_PATH = "gensim_train/word2vec.model"
TRAIN_FILE = 'data/single/train.json'  # 训练集
DEV_FILE = 'data/single/valid.json'  # 验证集
TEST_FILE = 'data/single/test.json'  # 测试文件

# EMBEDDING_PATH = "gensim_train/word2vec4ee.model"
# TRAIN_FILE = 'data/ee/multi-classification-train.txt'  # 训练集
# DEV_FILE = 'data/ee/multi-classification-test.txt'  # 验证集
# TEST_FILE = 'data/ee/multi-classification-test.txt'  # 测试文件

MODEL = "Al_Trans"
# TextCNN,LSTM,Transformer,TopJudge,Electra,Al_Trans,Attention_XML,CNN_Trans
if MODEL=="Electra" or MODEL=="Al_Trans" or MODEL=="Bert":
    PRETRAIN = True 
else:
    PRETRAIN = False 

ADD_DETAILS = True
HEAD = False
ADD_ATTN = False
ADD_CNN = False

K_CONS = False
SMOOTH = False
BEAM = False
LCM = False
CONTRASTIVE = False
CONTRA_WAY = "supcon2" #"r-drop","supcon2","supcon"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EMB_DIM = 256
HID_DIM = 256
SEQ_LEN = 512
BATCH_SIZE = 16 if PRETRAIN else 128
EPOCHS = 25 if PRETRAIN else 100

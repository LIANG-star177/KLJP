import torch.nn as nn
import torch
from setting import DEVICE
from tokenizer import MyTokenizer
import torch.nn.functional as F

def get_embedding_layer(embedding_weights):
    word_embeddings=nn.Embedding(embedding_weights.size(0),embedding_weights.size(1))
    word_embeddings.weight.data.copy_(embedding_weights)
    word_embeddings.weight.requires_grad=False #not train
    return word_embeddings

class Hybrid_XML(nn.Module):
    def __init__(self,vocab_size=30001,embedding_size=300,hidden_size=256,maps=None,embedding_weights=None,label_emb=None):
        super(Hybrid_XML,self).__init__()
        self.embedding_size=embedding_size
        d_a = hidden_size
        self.num_labels_c = len(maps["charge2idx"])
        self.num_labels_a = len(maps["article2idx"])
        self.hidden_size=hidden_size
        
        if embedding_weights is None:
            self.word_embeddings=nn.Embedding(vocab_size,embedding_size)
        else:
            self.word_embeddings=get_embedding_layer(embedding_weights)

        self.lstm=nn.LSTM(input_size=self.embedding_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        
        #interaction-attention layer
        self.key_layer = torch.nn.Linear(2*self.hidden_size,self.hidden_size)

        self.query_layer=torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.query_layer_a=torch.nn.Linear(self.hidden_size,self.hidden_size)
        #self-attn layer
        self.linear_first = torch.nn.Linear(2*self.hidden_size,d_a)
        self.linear_second = torch.nn.Linear(d_a,self.num_labels_c)
        self.linear_first_a = torch.nn.Linear(2*self.hidden_size,d_a)
        self.linear_second_a = torch.nn.Linear(d_a,self.num_labels_a)

        #weight adaptive layer
        self.linear_weight1=torch.nn.Linear(2*self.hidden_size,1)
        self.linear_weight2=torch.nn.Linear(2*self.hidden_size,1)
        self.linear_weight1_a=torch.nn.Linear(2*self.hidden_size,1)
        self.linear_weight2_a=torch.nn.Linear(2*self.hidden_size,1)
        
        #shared for all attention component
        self.linear_final = torch.nn.Linear(2*self.hidden_size,self.hidden_size)
        self.output_layer=torch.nn.Linear(self.hidden_size,1)
        self.linear_final_a = torch.nn.Linear(2*self.hidden_size,self.hidden_size)
        self.output_layer_a=torch.nn.Linear(self.hidden_size,1)
        
        label_embedding=torch.FloatTensor(self.num_labels_c,self.hidden_size)
        label_embedding_a=torch.FloatTensor(self.num_labels_a,self.hidden_size)
        if label_emb is None:
            nn.init.xavier_normal_(label_embedding)
            nn.init.xavier_normal_(label_embedding_a)
        else:
            label_embedding.copy_(label_emb)
        self.label_embedding=nn.Parameter(label_embedding,requires_grad=False)
        self.label_embedding_a=nn.Parameter(label_embedding_a,requires_grad=False)

    def init_hidden(self,batch_size):
        if torch.cuda.is_available():
            return (torch.zeros(2,batch_size,self.hidden_size).to(DEVICE),torch.zeros(2,batch_size,self.hidden_size).to(DEVICE))
        else:
            return (torch.zeros(2,batch_size,self.hidden_size),torch.zeros(2,batch_size,self.hidden_size))
                
    def forward(self,x):
        x = x["justice"]["input_ids"].to(DEVICE)
       
        emb=self.word_embeddings(x)
        
        hidden_state=self.init_hidden(emb.size(0))
        output,hidden_state=self.lstm(emb,hidden_state)#[batch,seq,2*hidden]
        

        #get attn_key
        attn_key=self.key_layer(output) #[batch,seq,hidden]
        attn_key=attn_key.transpose(1,2)#[batch,hidden,seq]
        #get attn_query
        label_emb=self.label_embedding.expand((attn_key.size(0),self.label_embedding.size(0),self.label_embedding.size(1)))#[batch,L,label_emb]
        label_emb=self.query_layer(label_emb)#[batch,L,label_emb]
        
        #attention
        similarity=torch.bmm(label_emb,attn_key)#[batch,L,seq]
        similarity=F.softmax(similarity,dim=2)
        
        out1_c=torch.bmm(similarity,output)#[batch,L,label_emb]
    
        #self-attn output
        self_attn=torch.tanh(self.linear_first(output)) #[batch,seq,d_a]
        self_attn=self.linear_second(self_attn) #[batch,seq,L]
        self_attn=F.softmax(self_attn,dim=1)
        self_attn=self_attn.transpose(1,2)#[batch,L,seq]
        out2=torch.bmm(self_attn,output)#[batch,L,hidden]


        factor1_c=torch.sigmoid(self.linear_weight1(out1_c))
        factor2_c=torch.sigmoid(self.linear_weight2(out2))
        factor1_c=factor1_c/(factor1_c+factor2_c)
        factor2_c=1-factor1_c
        
        out=factor1_c*out1_c+factor2_c*out2
        
        out=F.relu(self.linear_final(out))
        out=self.output_layer(out).squeeze(-1)#[batch,L]


        #get attn_query
        label_emb_a=self.label_embedding_a.expand((attn_key.size(0),self.label_embedding_a.size(0),self.label_embedding_a.size(1)))#[batch,L,label_emb]
        label_emb_a=self.query_layer_a(label_emb_a)#[batch,L,label_emb]
        
        #attention
        similarity_a=torch.bmm(label_emb_a,attn_key)#[batch,L,seq]
        similarity_a=F.softmax(similarity_a,dim=2)
        
        out1_a=torch.bmm(similarity_a,output)#[batch,L,label_emb]

        self_attn_a=torch.tanh(self.linear_first_a(output)) #[batch,seq,d_a]
        self_attn_a=self.linear_second_a(self_attn_a) #[batch,seq,L]
        self_attn_a=F.softmax(self_attn_a,dim=1)
        self_attn_a=self_attn_a.transpose(1,2)#[batch,L,seq]
        out2_a=torch.bmm(self_attn_a,output)#[batch,L,hidden]

        factor1_a=torch.sigmoid(self.linear_weight1_a(out1_a))
        factor2_a=torch.sigmoid(self.linear_weight2_a(out2_a))
        factor1_a=factor1_a/(factor1_a+factor2_a)
        factor2_a=1-factor1_a
        
        out_a=factor1_a*out1_a+factor2_a*out2_a
        
        out_a=F.relu(self.linear_final_a(out_a))
        out_a=self.output_layer_a(out_a).squeeze(-1)#[batch,L]

        return {
            "article": out_a,
            "charge": out,
            # "judge": out_judge
        }

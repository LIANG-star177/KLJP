import torch
import torch.nn as nn

from model.module import BertNMT



#init params with xavier without pretrained bert params
def init_xavier(model):
    for layer in model.named_parameters():
        if 'weight' in layer[0] and 'layer_norm' not in layer[0] and 'bert' not in layer[0] and layer[1].dim() > 1:
            nn.init.xavier_uniform_(layer[1])



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def load_model(model_name, config):
    model = BertNMT(config)

    #Avoiding Update pretrained params
    for param in model.bert.parameters():
        param.requires_grad = False
    
    model.encoder.apply(init_xavier)
    model.decoder.apply(init_xavier)    

    model.to(config.device)

    print(f'{model_name} model has loaded. The model has {count_parameters(model):,} trainable parameters')

    return model
import time
import math
import torch
import torch.nn as nn



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def create_src_mask(src, pad_idx=1):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    src_mask.to(src.device)
    return src_mask



def create_trg_mask(trg, pad_idx=1):
    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(2)
    trg_sub_mask = torch.tril(torch.ones((trg.size(-1), trg.size(-1)))).bool()

    trg_mask = trg_pad_mask & trg_sub_mask.to(trg.device)
    trg_mask.to(trg.device)
    return trg_mask




def train_epoch(model, dataloader, criterion, optimizer, clip, device):
    model.train()
    epoch_loss = 0

    for _, batch in enumerate(dataloader):
        src, trg = batch[0].to(device), batch[1].to(device)

        src_mask = create_src_mask(src)
        trg_mask = create_trg_mask(trg)

        pred = model(src, trg, src_mask, trg_mask)
        
        pred_dim = pred.shape[-1]
        pred = pred.contiguous().view(-1, pred_dim)

        loss = criterion(pred, trg.contiguous().view(-1))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)




def valid_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    batch_bleu = []

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            src, trg = batch[0].to(device), batch[1].to(device)
            
            src_mask = create_src_mask(src)
            trg_mask = create_trg_mask(trg)

            pred = model(src, trg, src_mask, trg_mask)
            
            pred_dim = pred.shape[-1]
            pred = pred.contiguous().view(-1, pred_dim)

            loss = criterion(pred, trg.contiguous().view(-1))

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
import torch
import torch.nn as nn
from tools import get_kappa

def train_model(model, device, lr, epochs, train_dataloader, valid_dataloader = None, logger = None):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr) 
    criterion = nn.MSELoss()
    best_loss = 1e15
    for ep in range(epochs):
        
        # train
        model.train()
        total = 0
        sum_loss = 0
        for x, lengths, scores, feat in train_dataloader:
            x = x.to(device)
            lengths = lengths.to(device)
            scores = scores.to(device)
            feat = feat.to(device)
            optimizer.zero_grad()
            pred_scores = model(x, lengths, feat)
            loss = criterion(scores, pred_scores)
            loss.backward()
            optimizer.step()
            total += scores.shape[0]
            sum_loss += loss.item() * scores.shape[0]
        
        # log train loss
        if logger is not None:
            logger.log(train_loss = sum_loss / total)
            
        # validate
        if valid_dataloader is not None:
            valid_loss, valid_w_kappa, valid_g_kappa, valid_i_kappa = evaluate_model(model, device, valid_dataloader, criterion)
        
        # log valid metrics
        if valid_dataloader is not None and logger is not None:
            logger.log(valid_loss = valid_loss, valid_weighted_kappa = valid_w_kappa, valid_global_kappa = valid_g_kappa, valid_individual_kappa = valid_i_kappa)
        
        # save best weights
        if valid_loss < best_loss:
            best_loss = valid_loss
            logger.checkpoint_weights(model)
        
        # display
        if (ep + 1) % 5 == 0:
            valid_string = f", (valid) loss {valid_loss: .3f}, weighted kappa {valid_w_kappa: .3f}, global kappa {valid_g_kappa: .3f}, individual kappa {list(valid_i_kappa.values())}" if valid_dataloader is not None else ""
            print(f"Ep[{ep + 1}/{epochs}] (train) loss {sum_loss / total: .3f}{valid_string}")

def evaluate_model(model, device, dataloader, criterion = nn.MSELoss()):
    model.eval()
    total = 0
    sum_loss = 0
    all_pred_scores = torch.zeros(len(dataloader.dataset))
    with torch.no_grad():
        for x, lengths, scores, feat in dataloader:
            x = x.to(device)
            lengths = lengths.to(device)
            scores = scores.to(device)
            feat = feat.to(device)
            pred_scores = model(x, lengths, feat)
            all_pred_scores[total: total + scores.shape[0]] = pred_scores.cpu()
            loss = criterion(scores, pred_scores)
            total += scores.shape[0]
            sum_loss += loss.item() * scores.shape[0]
        scores = dataloader.dataset.recover(dataloader.dataset.get_scores())
        pred_scores = dataloader.dataset.recover(all_pred_scores.numpy(), round_to_known = True)
        valid_w_kappa, valid_g_kappa, valid_i_kappa = get_kappa(scores, pred_scores, dataloader.dataset.get_sets())
    return sum_loss / total, valid_w_kappa, valid_g_kappa, valid_i_kappa
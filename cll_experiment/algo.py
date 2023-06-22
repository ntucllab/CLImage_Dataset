import torch
import torch.nn.functional as F
import numpy as np

def ga_loss(outputs, labels, class_prior, T, num_classes):
    device = labels.device
    if torch.det(T) != 0:
        Tinv = torch.inverse(T)
    else:
        Tinv = torch.pinverse(T)
    batch_size = outputs.shape[0]
    outputs = -F.log_softmax(outputs, dim=1)
    loss_mat = torch.zeros([num_classes, num_classes], device=device)
    for k in range(num_classes):
        mask = k == labels
        indexes = torch.arange(batch_size).to(device)
        indexes = torch.masked_select(indexes, mask)
        if indexes.shape[0] > 0:
            outputs_k = outputs[indexes]
            # outputs_k = torch.gather(outputs, 0, indexes.view(-1, 1).repeat(1,num_classes))
            loss_mat[k] = class_prior[k] * outputs_k.mean(0)
    loss_vec = torch.zeros(num_classes, device=device)
    for k in range(num_classes):
        loss_vec[k] = torch.inner(Tinv[k], loss_mat[k])
    return loss_vec

def l_mae(y, output):
    return 2 - 2 * F.softmax(output, dim=1)[:, y]

def l_cce(y, output):
    return -F.log_softmax(output, dim=1)[:, y]

def l_wmae(w):
    def real_l_wmae(y, output):
        return w[y] * l_mae(y, output)
    return real_l_wmae

def l_gce(y, output, q=0.7):
    return (1-F.softmax(output, dim=1)[:, y].pow(q)) / q

def l_sl(y, output, alpha=0.1, beta=1.0, A=-4):
    def l_rce(y, output, A):
        return -A * F.softmax(output, dim=1).sum(dim=1) - F.softmax(output, dim=1)[:, y]
    return alpha * l_cce(y, output) + beta * l_rce(y, output, A)

def robust_ga_loss(outputs, labels, class_prior, T, num_classes, algo_name):
    device = labels.device
    if torch.det(T) != 0:
        Tinv = torch.inverse(T)
    else:
        Tinv = torch.pinverse(T)
    
    if algo_name == 'rob-mae':
        loss_func = l_mae
    elif algo_name == 'rob-cce':
        loss_func = l_cce
    elif algo_name == 'rob-wmae':
        loss_func = l_wmae(Tinv.sum(dim=0).squeeze())
    elif algo_name == 'rob-gce':
        loss_func = l_gce
    elif algo_name == 'rob-sl':
        loss_func = l_sl
    else:
        raise NotImplementedError
        
    loss_vec = torch.zeros(num_classes, device=device)
    for k in range(num_classes):
        for j in range(num_classes):
            mask = j == labels
            indexes = torch.arange(outputs.shape[0]).to(device)
            indexes = torch.masked_select(indexes, mask)
            if indexes.shape[0] > 0:
                loss_vec[k] += class_prior[j] * Tinv[j][k] * loss_func(k, outputs[indexes]).mean()
    return loss_vec
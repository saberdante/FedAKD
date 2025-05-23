import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
__all__ = ["akd_loss"]


class akd_loss(nn.Module):
    def __init__(self, num_classes=10, tau=1, beta=1, gamma=1):
        super(akd_loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets, dg_logits, major_labels, data_map_value, data_distribution_map):
        targets = targets.long()
        ce_loss = self.CE(logits, targets)
        akd_loss = lmd_criterion(logits, dg_logits, targets, self.tau, major_labels, self.num_classes, data_map_value, self.beta, self.gamma)
        loss = ce_loss + akd_loss 
        return loss

def lmd_criterion(logits_student, logits_teacher, target, T, major_labels, num_classes, data_map_value, beta, gamma,):
    bs = logits_student.size(0)
    gt_mask = _get_gt_mask(logits_student, target)
    label_mask = torch.zeros_like(logits_student).scatter_(1, major_labels.repeat(bs, 1), 1).bool()
    other_mask = ~label_mask
    label_mask2 = label_mask & ~gt_mask
    other_mask2 = other_mask & ~gt_mask 

    
    logits_teacher = logits_teacher - torch.log(data_map_value + 1e-9).to(logits_teacher.dtype)
    
    pred_student = F.softmax(logits_student / T - 1000 * gt_mask, dim=1)
    pred_teacher = F.softmax(logits_teacher / T - 1000 * gt_mask, dim=1)
    
    pred_s3 = cat_mask(pred_student, label_mask2, other_mask2)
    pred_t3 = cat_mask(pred_teacher, label_mask2, other_mask2)  

    pred_t1, pred_t2 = cat_mask2(pred_teacher, label_mask2, other_mask2)
    pred_s1, pred_s2 = cat_mask2(pred_student, label_mask2, other_mask2)
    
    log_pred_s1, log_pred_s2 = torch.log(pred_s1 + 1e-9), torch.log(pred_s2 + 1e-9)
    log_pred_s3 = torch.log(pred_s3 + 1e-9)
    
    
    kl_part1 = nn.KLDivLoss(reduction="batchmean")(log_pred_s1, pred_t1)
    kl_part2 = nn.KLDivLoss(reduction="batchmean")(log_pred_s2, pred_t2)
    kl_part3 = nn.KLDivLoss(reduction="batchmean")(log_pred_s3, pred_t3)
    
    loss = (beta * kl_part1 + kl_part3 + gamma * kl_part2) * (T ** 2)
    
    return loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    # t1 = (t * mask1)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def cat_mask2(t, mask1, mask2):
    t1 = (t * mask1)
    t2 = (t * mask2)
    return t1, t2
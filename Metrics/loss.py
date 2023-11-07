import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=0.001):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        batch_num = targets.size(0)
        class_num = targets.size(1)

        if class_num == 1:
           probs = torch.sigmoid(logits)
           m1 = probs.view(batch_num, -1)
           m2 = targets.view(batch_num, -1)
           intersection = m1 * m2
           score = (
                   2.0
                   * (intersection.sum(1) + self.smooth)
                   / (m1.sum(1) + m2.sum(1) + self.smooth)
           )
           score = 1 - score.sum()/batch_num
        else:
           probs = torch.softmax(logits, dim=1)
           m1 = probs.view(batch_num, class_num, -1)
           m2 = targets.view(batch_num, class_num, -1)
           intersection = m1 * m2
           score = (
                   2.0
                   * (intersection.sum(2) + self.smooth)
                   / (m1.sum(2) + m2.sum(2) + self.smooth)
           )
           score = (1 - score.sum(0))/batch_num
           score = (score[0] + score[1]) / 2

        return score

# binary_class can choice nn.BCELoss()
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits, targets):
        batch_num = targets.size(0)
        class_num = targets.size(1)

        m1 = logits.view(batch_num, class_num, -1)
        m2 = targets.view(batch_num, class_num, -1)

        m11 = torch.nn.functional.log_softmax(m1, dim=1)
        loss = -torch.sum(m11 * m2)/ batch_num

        return loss

class wBCE_loss(nn.Module):
    def __init__(self):
        super(wBCE_loss, self).__init__()

    def forward(self, logits, targets):
        batch_num = targets.size(0)
        weight = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        weight_bce = (weight * bce).view(batch_num, -1).sum(1) / weight.view(batch_num, -1).sum(1)
        weight_bce = weight_bce.sum() / batch_num

        return weight_bce

class wIOU_loss(nn.Module):
    def __init__(self, smooth=0.001):
        super(wIOU_loss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        batch_num = targets.size(0)
        weight = 1 + 5 * torch.abs(F.avg_pool2d(targets, kernel_size=31, stride=1, padding=15) - targets)

        logits = torch.sigmoid(logits)
        inter = logits * targets
        union = logits + targets
        iou = 1 - (inter + self.smooth) / (union - inter + self.smooth)
        weight_iou = (weight * iou).view(batch_num, -1).sum(1) / weight.view(batch_num, -1).sum(1)
        weight_iou = weight_iou.sum() / batch_num

        return weight_iou

class tversky_loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2):
        super(tversky_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, targets):
        batch_num = targets.size(0)
        logits = torch.sigmoid(logits)
        logits = logits.view(batch_num, -1)
        targets = targets.view(batch_num, -1)

        TP = (logits * targets).sum(1)
        FP = ((1 - targets) * logits).sum(1)
        FN = ((1- logits) * targets).sum(1)

        Tversky = (TP + 1) /(TP + self.alpha * FP + self.beta * FN + 1)
        Tversky = Tversky.sum() / batch_num
        return (1 - Tversky) ** self.gamma
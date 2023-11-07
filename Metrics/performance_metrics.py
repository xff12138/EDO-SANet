import torch
import torch.nn.functional as F

class DiceScore(torch.nn.Module):
    def __init__(self, smooth=0.001):
        super(DiceScore, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        batch_num = targets.size(0)
        class_num = targets.size(1)

        if class_num == 1:
           probs = torch.sigmoid(logits)
           m1 = probs.view(batch_num, -1) > 0.5
           m2 = targets.view(batch_num, -1) > 0.5
           intersection = m1 * m2
           score = (
                   2.0
                   * (intersection.sum(1) + self.smooth)
                   / (m1.sum(1) + m2.sum(1) + self.smooth)
           )
           score = score.sum() / batch_num
        else:
           probs = torch.softmax(logits, dim=1)
           m1 = probs.view(batch_num, class_num, -1)
           m1 = torch.argmax(m1, dim=1)
           m1 = F.one_hot(m1, num_classes=class_num)
           m1 = m1.transpose(1, 2)

           m2 = targets.view(batch_num, class_num, -1)
           m2 = torch.argmax(m2, dim=1)
           m2 = F.one_hot(m2, num_classes=class_num)
           m2 = m2.transpose(1, 2)
           intersection = m1 * m2
           score = (
                   2.0
                   * (intersection.sum(2) + self.smooth)
                   / (m1.sum(2) + m2.sum(2) + self.smooth)
           )
           score = score.sum(0) / batch_num
           score = (score[0] + score[1]) / 2

        return score






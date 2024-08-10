import torch
import numpy as np
import torch.nn as nn


class tct_Loss(nn.Module):
    def __init__(self, margin=0.7):
        super(tct_Loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs1, inputs2, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat1 = inputs1.chunk(label_num * 2, 0)
        feat2 = inputs2.chunk(label_num * 2, 0)
 
        center1 = []
        center2 = []
 
        for i in range(label_num * 2):
            center1.append(torch.mean(feat1[i], dim=0, keepdim=True))
            center2.append(torch.mean(feat2[i], dim=0, keepdim=True))
          
        c1 = torch.cat(center1)
        c2 = torch.cat(center2)
     
        n = c1.size(0)

        dist1 = pdist_torch(c1, c1)
        dist2 = pdist_torch(c2, c2)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap1, dist_an1 = [], []
   
        for i in range(n):
            dist_ap1.append(dist1[i][mask[i]].max().unsqueeze(0))
            dist_an1.append(dist2[i][mask[i] == 0].min().unsqueeze(0))

      
        dist_ap1 = torch.cat(dist_ap1)
        dist_an1 = torch.cat(dist_an1)

  

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an1)
        loss = self.ranking_loss(dist_an1, dist_ap1, y)
    

        return loss


def pdist_torch(inputs1, inputs2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    n = inputs1.size(0)

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs1, inputs1.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


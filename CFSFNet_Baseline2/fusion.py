import torch.nn as nn
import torch
import math


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.inchannel = 2048
        reduc_ratio = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.i = 0

        self.weight = nn.Sequential(
            nn.Conv2d(self.inchannel, self.inchannel // reduc_ratio, kernel_size= 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inchannel // reduc_ratio, self.inchannel, kernel_size= 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, bn):#x[s,c]
        # b, c, h, w = x[0].shape

        x0 = self.avg_pool(x[0])
        x1 = self.avg_pool(x[1])
        x0 = self.weight(x0)  # n,c,1,1
        x1 = self.weight(x1)
        x_w0 = x0.sqrt() / (x0.sqrt() + x1.sqrt())
        x_w1 = x1.sqrt() / (x0.sqrt() + x1.sqrt())

        x_fu = x_w0 * x[0] + x_w1 * x[1]

        p = 1e+6
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()  # c 2048
        bn1 = (bn1 / bn1.sum())
        bn2 = (bn2 / bn2.sum())
        bn1 = torch.round(bn1*p)
        bn2 = torch.round(bn2*p)

        x_fu[:, bn2 >= bn1] = x[1][:, bn2 >= bn1]
        out = x_fu

        #
        # if self.i % 2048 == 0:
        #     c = 0
        #     d = 0
        #     e = 0
        #     # print(x_w0)
        #     for s in range(2048):
        #         if(bn2[s] > bn1[s]):
        #             c = c + 1
        #         if(bn2[s]==bn1[s]):
        #             d = d+1
        #         if(bn2[s] < bn1[s]):
        #             e = e+1
        #     print(c,d,e)
        # self.i = self.i + 1

        return out




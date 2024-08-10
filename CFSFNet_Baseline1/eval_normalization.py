import torch.nn as nn


class BN(nn.Module):
    def __init__(self, num_channels, num_features):#2048 2
        super(BN, self).__init__()
        for i in range(num_features):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_channels, affine=True, track_running_stats=True))

    def forward(self, feature):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(feature)]

class Get_BN(nn.Module):
    def __init__(self, num_channels, num_features):
        super(Get_BN, self).__init__()
        self.list = []
        self.bn = BN(num_channels, num_features)

    def forward(self, features):
        bn_features = self.bn(features)
        for module in self.bn.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.list.append(module)
        return bn_features, self.list

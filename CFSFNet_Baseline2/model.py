from torchvision import models
import torch, math
import torch.nn as nn
from torch.nn import init
from functools import partial
from resnet import resnet50
from fusion import Fusion
from eval_normalization import *
import torch.nn.functional as F


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
        m.bias.requires_grad_(False)




class visible_module(nn.Module):
    def __init__(self, non_local='on', arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)

        self.visible = model_v
        layers = [3, 4, 6, 3]
        non_layers = [0, 2, 3, 0]
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.visible.layer1)):
            x = self.visible.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        x_v = x.clone()
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.visible.layer2)):
            x = self.visible.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3

        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.visible.layer3)):
            x = self.visible.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
     # Layer 4

        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.visible.layer4)):
            x = self.visible.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x_v, x


class common_module(nn.Module):
    def __init__(self, non_local='on', arch='resnet50'):
        super(common_module, self).__init__()

        model_c = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)

        self.common = model_c
        layers = [3, 4, 6, 3]
        non_layers = [0, 2, 3, 0]
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x):

        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.common.layer2)):
            x = self.common.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3

        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.common.layer3)):
            x = self.common.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4

        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.common.layer4)):
            x = self.common.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x


class thermal_module(nn.Module):
    def __init__(self, non_local='on', arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        self.thermal = model_t
        layers = [3, 4, 6, 3]
        non_layers = [0, 2, 3, 0]
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.thermal.layer1)):
            x = self.thermal.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        x_t = x.clone()
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.thermal.layer2)):
            x = self.thermal.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3

        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.thermal.layer3)):
            x = self.thermal.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4

        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.thermal.layer4)):
            x = self.thermal.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x_t, x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)

        x = self.base.layer2(x)

        x = self.base.layer3(x)

        x = self.base.layer4(x)

        return x


class embed_net(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.common_module = common_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.fusion = Fusion()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dim = 2048
        self.bn = Get_BN(2048, 2)

        self.bottleneck = nn.BatchNorm1d(self.dim)
        self.classifier = nn.Linear(self.dim, class_num, bias=False)
        self.bottleneck.apply(weights_init)
        self.classifier.apply(weights_init)
        self.ignored_params = nn.ModuleList([self.bottleneck, self.classifier])
        # self.np, self.nh = 6, 8


    def forward(self, input1, input2, modal=0):
        if modal == 0:

            x_v1, x_sv = self.visible_module(input1)
            x_t1, x_st = self.thermal_module(input2)
            input3 = torch.cat((x_v1, x_t1), 0)
            x_c = self.common_module(input3)
            x_cv, x_ct = torch.chunk(x_c, 2)

            x_v = [x_sv, x_cv]
            x_t = [x_st, x_ct]

            x_v_bn, list1 = self.bn(x_v)
            x_v = self.fusion(x_v, list1)
            x_v_bn = self.fusion(x_v_bn, list1)

            x_t_bn, list2 = self.bn(x_t)
            x_t = self.fusion(x_t, list2)
            x_t_bn = self.fusion(x_t_bn, list2)

            x = torch.cat((x_v_bn, x_t_bn), 0)


        elif modal == 1:
            x_v1, x_sv = self.visible_module(input1)
            x_cv = self.common_module(x_v1)
            x = [x_sv, x_cv]

            x_bn, list1 = self.bn(x)
            x = self.fusion(x_bn, list1)


        elif modal == 2:
            x_t1, x_st = self.thermal_module(input2)
            x_ct = self.common_module(x_t1)
            x = [x_st, x_ct]

            x_bn, list2 = self.bn(x)
            x = self.fusion(x_bn, list2)


        # B, C, H, W = x_bn.shape
        # pp = x_bn.view(B, self.nh, self.dim // self.nh, self.np, H // self.np, W)
        # pp = pp.mean(-1).mean(-1).permute(0, 1, 3, 2).contiguous()
        # pp = pp.view(B, self.nh * self.np, self.dim // self.nh)

        pool = self.avgpool(x).squeeze()
        feat = self.bottleneck(pool)
        if self.training:
            y = self.classifier(feat)
            feat_com = []
            for i in (x_sv, x_cv, x_st, x_ct, x_v, x_t):
                feat_com.append(i)
            # b, c, h, w = feat_com[0].shape
            # x_pool = []
            for i in range(len(feat_com)):
                feat_com[i] = self.avgpool(feat_com[i]).squeeze()

            return pool, y, feat_com
        else:
            return pool, feat

#
# class resnet(nn.Module):
#     def __init__(self, class_num):
#         super(resnet, self).__init__()
#         model = models.resnet50(pretrained=True)
#         for mo in model.layer4[0].modules():
#             if isinstance(mo, nn.Conv2d):
#                 mo.stride = (1, 1)
#         self.layer = nn.Sequential(*list(model.children())[:-2])
#         self.dim = 2048
#         self.np, self.nh = 6, 8
#         # --------------------------------------------------------------------------
#         # self.mam = MAM(2048)
#         # num_patches = 18 * 9
#         # depth = 1
#         # self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, dim))])
#         # trunc_normal_(self.cls_token[0], std=.02)
#
#         # self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, num_patches + 1, dim))])
#         # trunc_normal_(self.pos_embed[0], std=.02)
#
#         # self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, num_patches, dim))])
#         # trunc_normal_(self.pos_embed[0], std=.02)
#
#         # dpr = [x.item() for x in torch.linspace(0, 0.1, depth)] # dpr = [x.item() for x in torch.linspace(0, 0.1, 1)]
#         # self.blocks = nn.ModuleList([
#         #     Block(
#         #         dim=2048, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#         #         drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6))
#         #     for i in range(depth)])
#         # self.norm = partial(nn.LayerNorm, eps=1e-6)(dim)
#         # --------------------------------------------------------------------------
#         self.bottleneck = nn.BatchNorm1d(self.dim)
#         self.bottleneck.apply(weights_init)
#         self.classifier = nn.Linear(self.dim, class_num, bias=False)
#         self.classifier.apply(weights_init)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.ignored_params = nn.ModuleList([self.bottleneck, self.classifier])
#
#
#
#
#     def forward(self, x, camids=None, pids=None):
#         x = self.layer(x)
#         B, C, H, W = x.shape
#         pp = x.view(B, self.nh, self.dim // self.nh, self.np, H // self.np, W)
#         pp = pp.mean(-1).mean(-1).permute(0, 1, 3, 2).contiguous()
#         pp = pp.view(B, self.nh * self.np, self.dim // self.nh)
#
#         pool = self.avgpool(x).squeeze()
#         feat = self.bottleneck(pool)
#         if self.training:
#             y = self.classifier(feat)
#             return pool, y, pp
#         else:
#             return pool, feat

#
# def _no_grad_trunc_normal_(tensor, mean, std, a, b):
#     def norm_cdf(x):
#         return (1. + math.erf(x / math.sqrt(2.))) / 2.
#
#     if (mean < a - 2 * std) or (mean > b + 2 * std):
#         print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
#               "The distribution of values may be incorrect.", )
#
#     with torch.no_grad():
#         l = norm_cdf((a - mean) / std)
#         u = norm_cdf((b - mean) / std)
#         tensor.uniform_(2 * l - 1, 2 * u - 1)
#         tensor.erfinv_()
#         tensor.mul_(std * math.sqrt(2.))
#         tensor.add_(mean)
#         tensor.clamp_(min=a, max=b)
#         return tensor
#
#
# def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
#     return _no_grad_trunc_normal_(tensor, mean, std, a, b)


if __name__ == '__main__':
    pass






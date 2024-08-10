import io
import os
import requests
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import pdb
from model_vis import embed_net
from model_base import embed_net_base
import argparse
import torch.backends.cudnn as cudnn
from gradcam import GradCAM, show_cam_on_image, center_crop_img
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')

parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')

parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='on', type=str, help='performing PCB, on or off')
parser.add_argument('--w_center', default=1.0, type=float, help='the weight for center loss')

parser.add_argument('--local_feat_dim', default=256, type=int,
                    help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=9, type=int,
                    help='num of local strips in PCB')
args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# input image

model_path = ''
img_path = ''

def main():

    for i in range(2):
        if model_path == 'model/fusion,sysu_com_b8_p6_lr_0.1_tbc0.0_kl0.0_dds0.0_margin0.7_epoch_78.t':
            model = embed_net(395, no_local='on', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb,
                              local_feat_dim=args.local_feat_dim,
                              num_strips=args.num_strips)

            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['net'], strict=False)
            target_layers = [model.exchange]  # 定义目标层

        else:
            model = embed_net_base(395, no_local='on', gm_pool='on', arch=args.arch, share_net=args.share_net,
                                   pcb=args.pcb,
                                   local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
            target_layers = [model.base_resnet.base.layer4[-1]]

        model = model.eval().to(device)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # load image

        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

        img = Image.open(img_path).convert('RGB')

        t = transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)), ])
        img = t(img)
        img = np.array(img, dtype=np.uint8)

        # [C, H, W]
        img_tensor = data_transform(img)

        # expand batch dimension
        # [C, H, W] -> [N, C, H, W]
        input_tensor = torch.unsqueeze(img_tensor, dim=0)

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
        target_category = 395
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                          grayscale_cam,
                                          use_rgb=True)
        plt.imshow(visualization)
        plt.show()


if __name__ == '__main__':
    main()
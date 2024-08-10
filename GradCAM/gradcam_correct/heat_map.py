import sys
import os.path as osp
from datetime import datetime
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torch
import cv2
# from model_mine import embed_net_base
from model import embed_net
import numpy as np
import matplotlib.pyplot as plt
import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#input img file
input_folder="input_vispic"
output_folder="out_vispic"

# input_folder="input_infpic"
# output_folder="out_infpic"

weight=[0.9,0.9,1.1]
current_datetime=datetime.now()
output_folder=output_folder+'/'+current_datetime.strftime("%Y-%m-%d-%H;%M")
# sys.stdout = Logger(osp.join(output_folder, "output.log"))
#

model_dir= ""

model = embed_net(395, no_local='on', gm_pool='on', arch='resnet50', share_net=2, pcb='on',
                              local_feat_dim=256,
                              num_strips=9)

checkpoint = torch.load(model_dir, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['net'], strict=False)

transform = transforms.Compose([
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# model_features=nn.Sequential(*list(model.children())[:-2])
# print(model_features)
model.eval()
os.makedirs(output_folder,exist_ok=True)


def Batch_creat_heatmap(model, input_folder, output_folder, filename, weight, feature_lay_need):
    blue_weight=weight[0]
    green_wight=weight[1]
    red_wight=weight[2]
    image_path = input_folder + '/' + filename
    output_path = output_folder + '/' + filename


    image = Image.open(image_path)
    image=image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        # feature,  feature_attentionWeight,yl4 = model(input_tensor)
        feature ,feature_attentionWeight = model(input_tensor)

    # feature_lay=feature_lay_need
    # attention_map=feature[feature_lay]
    # attention_map = attention_map.squeeze(0)  # 去掉批次维度
    # attention_map = torch.mean(attention_map, dim=0)  # 计算每个通道的平均值作为注意力热图

    feature_lay = feature_lay_need
    attention_map = feature[feature_lay]
    attention_map = attention_map.squeeze(0)  # 去掉批次维度

    feature_attentionWeight_list=feature_attentionWeight[feature_lay]
    feature_AW_MAX_index=torch.argmax(feature_attentionWeight_list)
    attention_map_channel=attention_map[feature_AW_MAX_index.item(),:,:]
    attention_map=attention_map_channel

    heatmap_np = attention_map.detach().numpy()
    heatmap_np = (heatmap_np - np.min(heatmap_np)) / (np.max(heatmap_np) - np.min(heatmap_np))  # 归一化到0~1之间
    heatmap_np = np.uint8(255 * heatmap_np)

    heatmap_color = cv2.applyColorMap(heatmap_np, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, image.size)

    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # 将 PIL 图像转换为 numpy 数组并转换颜色通道顺序
    overlayed_image = cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.5, 0)

    overlayed_image = overlayed_image.astype(np.float64)
    overlayed_image[:,:,0]*=blue_weight
    overlayed_image[:,:,1]*=green_wight
    overlayed_image[:,:,2]*=red_wight

    cv2.imwrite(output_path, overlayed_image)
    print("图像{}热力图已生成并保存,热力通道参数为红色通道{}，绿色通道{}，蓝色通道{}".format(filename,red_wight,green_wight,blue_weight))

count=0
feature_lay_need=0

for file_name in os.listdir(input_folder):
    Batch_creat_heatmap(model, input_folder, output_folder, file_name,weight,feature_lay_need)
    count+=1

print("*"*50+"\n文件夹中共有{}个文件，成功生成并保存了{}个注意力热图,保存在{}".format(len(os.listdir(input_folder)),count,output_folder))


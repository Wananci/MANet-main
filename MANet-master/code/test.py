# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import sys
import cv2
import time
sys.path.append('./models')
import numpy as np
import os, argparse
from manet import *
# from samnet import *
# from mobilenetonly2 import *
from _data import test_dataset

parser = argparse.ArgumentParser()
type = 'real'  # real or craft
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='../dataset/NSRD/test/test/' + type + '/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path
model_version = 'manet'

# load the model
model = HybridAttionNetwork('test')
model.load_state_dict(torch.load('../save/checkpoints/'+model_version+'.pth'))
model.cuda()
model.eval()

save_path = '../save/result/'+model_version +'/'+ type + '/img/'
edge_save_path = '../save/result/'+model_version+'/' + type + '/edge/'
# s1_save_path = '../save/result/'+model_version+'/'+type+'/s1/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(edge_save_path):
    os.makedirs(edge_save_path)
# if not os.path.exists(s1_save_path):
#     os.makedirs(s1_save_path)
image_root = dataset_path + 'images/'
gt_root = dataset_path + 'annotations/'
test_loader = test_dataset(image_root, gt_root, opt.testsize)
mae_sum = 0
mae_sum1 = 0
time_t = 0.0
for i in range(test_loader.size):
    image, gt, name, image_for_post = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    time_start = time.time()
    epoch = 0
    out1, _, _,_ = model(image)
    time_end = time.time()
    time_t = time_t + time_end - time_start
    res = F.upsample(out1, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    cv2.imwrite(save_path + name, res * 255)
    # o2 = F.upsample(out2, size=gt.shape, mode='bilinear', align_corners=False)
    # o3 = F.upsample(out3, size=gt.shape, mode='bilinear', align_corners=False)
    # o4 = F.upsample(out4, size=gt.shape, mode='bilinear', align_corners=False)
    # edge1 = F.upsample(ed, size=gt.shape, mode='bilinear', align_corners=False)
    # o2 = o2.sigmoid().data.cpu().numpy().squeeze()
    # o3 = o3.sigmoid().data.cpu().numpy().squeeze()
    # o4 = o4.sigmoid().data.cpu().numpy().squeeze()
    # edge1 = edge1.sigmoid().data.cpu().numpy().squeeze()
    # o2 = (o2 - o2.min()) / (o2.max() - o2.min() + 1e-8)
    # o3 = (o3 - o3.min()) / (o3.max() - o3.min() + 1e-8)
    # o4 = (o4 - o4.min()) / (o4.max() - o4.min() + 1e-8)
    # edge1 = (edge1 - edge1.min()) / (edge1.max() - edge1.min() + 1e-8)
    # cv2.imwrite('../save/result/'+model_version +'/'+ type + '/img2/' + name, o2 * 255)
    # cv2.imwrite('../save/result/'+model_version +'/'+ type + '/img3/' + name, o3 * 255)
    # cv2.imwrite('../save/result/'+model_version +'/'+ type + '/img4/' + name, o4 * 255)
    # cv2.imwrite(edge_save_path + name, edge1 * 255)
    mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
    # mae_sum1 += np.sum(np.abs(s1 - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

mae = mae_sum / test_loader.size
# mae1 = mae_sum1 / test_loader.size
print(mae)
fps = test_loader.size / time_t
print('FPS is %f' % (fps))



import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import visual_utils

from visual_utils import default_loader, default_flist_reader, ImageFilelist, compute_per_class_acc, save_correct_imgs, \
    add_paths, prepare_attri_label, get_group, add_glasso, add_dim_glasso
from CAM_Utils import calculate_atten_IoU
import sys
from PIL import Image
import numpy as np
#import h5py
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
import scipy.io as sio
import pickle
import json
from statistics import mean

def calculate_average_IoU(whole_IoU, IoU_thr=0.3):
    img_num = len(whole_IoU)
    body_parts = whole_IoU[0].keys()
    body_avg_IoU = {}
    for body_part in body_parts:
        body_avg_IoU[body_part] = []
        body_IoU = []
        for im_id in range(img_num):
            if whole_IoU[im_id][body_part][0] != -1:
                if_one = []
                for item in whole_IoU[im_id][body_part]:
                    if_one.append(1 if item > IoU_thr else 0)
                body_IoU.append(mean(if_one))
        body_avg_IoU[body_part].append(mean(body_IoU))
    num = 0
    sum = 0
    for part in body_avg_IoU:
        if part != 'tail':
            sum += body_avg_IoU[part][0]
            num += 1
    print(sum/num *100)
    return body_avg_IoU, sum/num *100


# def IoU_attention(input, impath, save_att_idx, maps, pre_attri, IoU_scale, save_att=False):
#     """
#     In this function, we can test the IoU of attribute attention maps, and can save attention maps to visualize the activation
#     :param input: input image
#     :param impath: input image paths
#     :param save_att_idx: list of len batch_size, indicate is save attribute for that image
#     :param maps: the attention maps with shape
#     :param pre_attri: the predicted attribute value
#     :param IoU_scale: the bounding box size.
#     :param save_att: The direction to save attention maps
#     :return: the IoU of each body part in this batch of images
#     """
#     sub_group_dic = json.load(open('./save_files/attri_groups_8_layer.json'))
#     # sub_group_dic is a dict with 8 keys, each key corresponds to one body part
#     # In each body part key, there is a dictionary that saves the attribute index for each attributes,
#     # such as 15 colors for has_bill_color
#     # {"head": {"has_bill_color": [attribute indexes for 15 colors in has_bill_color],
#     #           "has_eye_color": [attribute indexes for 15 colors in has_eye_color],]},
#     #   "back": {"has_upperparts_color":... },
#     #   "belly": {...},
#     # ...
#     # }
#     vis_groups = json.load(open('./save_files/attri_groups_8.json'))
#
#     target_groups = [{} for _ in range(pre_attri.size(0))]  # calculate the target groups for each image
#     for part in vis_groups.keys():
#         sub_group = sub_group_dic[part]
#         keys = list(sub_group.keys())
#         sub_activate_id = []
#         # sub_activate_id is the attention id for each part in each image. The size is img_num * sub_group_num
#         for k in keys:
#             sub_activate_id.append(torch.argmax(pre_attri[:, sub_group[k]], dim=1, keepdim=True))
#         sub_activate_id = torch.cat(sub_activate_id, dim=1).cpu().tolist()  # (batch_size, sub_group_dim)
#         for img_id, argdims in enumerate(sub_activate_id):
#             target_groups[img_id][part] = [sub_group[keys[i]][argdim] for i, argdim in enumerate(argdims)]
#
#         KP_root = './save_KPs/'
#         batch_IoU = calculate_atten_IoU(input, impath, save_att_idx, maps, target_groups, KP_root,
#                                         save_att=save_att, scale=IoU_scale)
#         return batch_IoU

# def test_with_IoU(model, testloader, attribute, IoU_thr, IoU_scale, required_num=2, save_att=False):
#     """
#     In this function, we can test the IoU of attribute attention maps, and can save attention maps to visualize the activation
#     :param model: loaded model
#     :param testloader:
#     :param attribute: train attributes (model input, no effect to activation maps here)
#     :param IoU_scale: the bounding box size.
#     :param IoU_thr: if IoU(attention, ground_truth) > IoU_thr, the attention map is correctly localized.
#     :param required_num: the number of images in each categories to be saved
#     :return:
#     """
#     print('Calculating the IoU of attention maps, saving attention map to:', save_att)
#     whole_IoU = []
#     # print("vis_groups:", vis_groups)
#     with torch.no_grad():
#         count = dict()
#         for i, (input, target, impath) in tqdm(enumerate(testloader)):
#
#             # choose the highest predicted attribute in each subgroup to evaluate the IoU
#             # i.e. if red is the highest predicted attribute in has_bill_color,
#             # then we evaluate the IoU between attribute attention map for has_bill_color::red and the bounding box for bill.
#
#             save_att_idx = []
#             labels = target.data.tolist()
#             for idx in range(len(labels)):
#                 label = labels[idx]
#                 if label in count:
#                     count[label] = count[label] + 1
#                 else:
#                     count[label] = 1
#                 if count[label] <= required_num:
#                     save_att_idx.append(1)
#                 else:
#                     save_att_idx.append(0)
#
#             input = input.cuda()
#
#             output, pre_attri, attention, _ = model(input, attribute)
#             maps = attention['layer4'].cpu().numpy()
#             pre_attri = pre_attri['layer4']
#
#             # pre_attri.shape : 64, 312
#             # maps.shape : 64, 312, 7, 7 {batch_size * attribute_numbers * feature_map_size}
#
#             batch_IoU = IoU_attention(input, impath, save_att_idx, maps, pre_attri, IoU_scale, save_att=False)
#
#             whole_IoU += batch_IoU
#
#             # break
#     body_avg_IoU, mean_IoU = calculate_average_IoU(whole_IoU, IoU_thr=IoU_thr)
#     return body_avg_IoU, mean_IoU

def test_with_IoU(model, testloader, attribute, IoU_thr, IoU_scale, required_num=2, save_att=False):
    """
    In this function, we can test the IoU of attribute attention maps, and can save attention maps to visualize the activation
    :param model: loaded model
    :param testloader:
    :param attribute: train attributes (model input, no effect to activation maps here)
    :param vis_groups: the groups to be shown
    :param vis_layer: the layers to be shown
    :param vis_root: save path to activation maps
    :param required_num: the number of images in each categories to be saved
    :return:
    """
    print('Calculating the IoU of attention maps, saving attention map to:', save_att)
    sub_group_dic = json.load(open('./save_files/attri_groups_8_layer.json'))
    '''
    sub_group_dic is a dict with 8 keys, each key corresponds to one body part
    In each body part key, there is a dictionary that saves the attribute index for each attributes,
    such as 15 colors for has_bill_color
    {"head": {"has_bill_color": [attribute indexes for 15 colors in has_bill_color],
              "has_eye_color": [attribute indexes for 15 colors in has_eye_color],]},
      "back": {"has_upperparts_color":... },
      "belly": {...},
    ...
    }
    '''

    vis_groups = json.load(open('./save_files/attri_groups_8.json'))

    whole_IoU = []
    # print("vis_groups:", vis_groups)
    with torch.no_grad():
        count = dict()
        # --------------------------------------
        # Here: replace you own dataloader and model,
        # make sure you can get (input, target, impath) and maps, pre_attri with the same size and format

        for i, (input, target, impath) in tqdm(enumerate(testloader)):


            input = input.cuda()

            _, pre_attri, attention, _ = model(input, attribute)

            maps = attention['layer4'].cpu().numpy()
            pre_attri = pre_attri['layer4']

            # pre_attri.shape : 64, 312
            # maps.shape : 64, 312, 7, 7 {batch_size * attribute_numbers * feature_map_size}
            # --------------------------------------

            target_groups = [{} for _ in range(pre_attri.size(0))]  # calculate the target groups for each image
            # target_groups is a list of size image_num
            # each item is a dict, including the attention index for each subgroup
            for part in vis_groups.keys():
                sub_group = sub_group_dic[part]
                keys = list(sub_group.keys())
                sub_activate_id = []
                # sub_activate_id is the attention id for each part in each image. The size is img_num * sub_group_num
                for k in keys:
                    sub_activate_id.append(torch.argmax(pre_attri[:, sub_group[k]], dim=1, keepdim=True))
                sub_activate_id = torch.cat(sub_activate_id, dim=1).cpu().tolist()  # (batch_size, sub_group_dim)
                for img_id, argdims in enumerate(sub_activate_id):
                    target_groups[img_id][part] = [sub_group[keys[i]][argdim] for i, argdim in enumerate(argdims)]

            save_att_idx = []
            labels = target.data.tolist()
            for idx in range(len(labels)):
                label = labels[idx]
                if label in count:
                    count[label] = count[label] + 1
                else:
                    count[label] = 1
                if count[label] <= required_num:
                    save_att_idx.append(1)
                else:
                    save_att_idx.append(0)

            KP_root = './save_KPs/'
            scale = IoU_scale
            batch_IoU = calculate_atten_IoU(input, impath, save_att_idx, maps, target_groups, KP_root,
                                            save_att=save_att, scale=scale)
            whole_IoU += batch_IoU

            # break
    body_avg_IoU, mean_IoU = calculate_average_IoU(whole_IoU, IoU_thr=IoU_thr)
    return body_avg_IoU, mean_IoU

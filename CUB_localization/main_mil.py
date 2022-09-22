from __future__ import print_function
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
from model_MIL import Net
from CAM_Utils import calculate_atten_IoU
from IoU import test_with_IoU, calculate_average_IoU
import sys
from PIL import Image
import numpy as np
import h5py
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

cudnn.benchmark = True

CC_HOME = os.environ.get('CC_HOME')
CC_DATA = os.environ.get('CC_DATA')
CC_HOME = './'
CC_DATA = './data/'


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default = None, type=str, help='Path to the model state file')
parser.add_argument('--dataset', default='CUB', help='FLO, CUB')
parser.add_argument('--root', default=CC_HOME, help='path to project')

parser.add_argument('--image_root', default=CC_DATA + 'images/', type=str, metavar='PATH',
                    help='path to image root')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--ol', action='store_true', default=False,
                    help='original learning, use unseen dataset when training classifier')
parser.add_argument('--preprocessing', action='store_true', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
# parser.add_argument('--lr', type=float, default=1e-6, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=1e-6, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--gpu_id', default=None)  # GPU id
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', default='cub', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--ALE', action='store_true', default=False, help='use ALE as classifier')
parser.add_argument('--imagelist', default=CC_HOME + '/ZSL_REG/data/CUB/cub_imagelist.txt', type=str, metavar='PATH',
                    help='path to imagelist (default: none)')
parser.add_argument('--onlycorrect', action='store_true', default=False,
                    help="only process the correctly classified images")
parser.add_argument('--resnet_path', default='../pretrained_models/resnet101_cub.pth.tar',   # resnet101_cub.pth.tar resnet101-5d3b4d8f.pth
                    help="path to pretrain resnet classifier")
parser.add_argument('--finetune', action='store_true', default=False, help='use ALE as classifier')
parser.add_argument('--train_id', type=int, default=1001)
parser.add_argument('--pretrained', default=None, help="path to pretrain classifier (to continue training)")
parser.add_argument('--checkpointroot', default=CC_HOME + '/ZSL_REG/checkpoint', help='path to checkpoint')
parser.add_argument('--image_type', default='test_unseen_loc', type=str, metavar='PATH',
                    help='image_type to visualize, usually test_unseen_small_loc, test_unseen_loc, test_seen_loc')
parser.add_argument('--pretrain_epoch', type=int, default=5)
parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='learning rate to pretrain model')

# ---------IoU Settings-------------
parser.add_argument('--save_att', type=str, default=False, help='The save direction of attributes attention')
parser.add_argument('--IoU_scale', type=int, default=4)  # The size of IoU_scale
parser.add_argument('--IoU_thr', type=float, default=0.1)  # The threshold of IoU
parser.add_argument('--save_img_num', type=int, default=3, help='How many images in each class should be saved.')  # The threshold of IoU
# ----------------------------------

global opt
opt = parser.parse_args()
opt.dataroot = opt.root + 'data'
print(opt)

# define random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# improve the efficiency
# check CUDA
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def main():
    # -----------load data-------------
    data = visual_utils.DATA_LOADER(opt)
    if opt.image_type == 'test_unseen_small_loc':
        test_loc = data.test_unseen_small_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_unseen_loc':
        test_loc = data.test_unseen_loc
        test_classes = data.unseenclasses
    elif opt.image_type == 'test_seen_loc':
        test_loc = data.test_seen_loc
        test_classes = data.seenclasses
    else:
        try:
            sys.exit(0)
        except:
            print("choose the image_type in ImageFileList")

    # We need to prepare the attribute labels
    class_attribute = data.attribute
    attribute_zsl = prepare_attri_label(class_attribute, data.unseenclasses).cuda()
    attribute_seen = prepare_attri_label(class_attribute, data.seenclasses).cuda()
    attribute_gzsl = torch.transpose(class_attribute, 1, 0).cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset_test = ImageFilelist(opt, data_inf=data,
                                  transform=transforms.Compose([
                                      transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      normalize,]),
                                  dataset=opt.dataset,
                                  image_type=opt.image_type)

    print("# of test samples: ", len(dataset_test))
    testloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opt.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    print('Create Model...')
    print("Use cuda:", opt.cuda)
    # -----------load model-------------
    model = Net(models.resnet50())

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.cuda()
        attribute_zsl = attribute_zsl.cuda()
        attribute_seen = attribute_seen.cuda()
    else:
        device = torch.device('cpu')
    if opt.model_name is not None:
        model.load_state_dict(torch.load(opt.model_name,map_location=device),strict=False)
   
    # -----------calculate IoU-------------
    # if you only want to test the IoU, set the save_att = False
    # save_att = False
    # if you want to save the attention map and bounding box, set following:
    save_att = opt.save_att

    body_avg_IoU, mean_IoU = test_with_IoU(model, testloader, attribute_zsl, IoU_thr=opt.IoU_thr, IoU_scale=opt.IoU_scale, save_att=save_att, required_num=opt.save_img_num)
    print('the Body part IoU is:/n', mean_IoU, body_avg_IoU)

if __name__ == '__main__':
    main()


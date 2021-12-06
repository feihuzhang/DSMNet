from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10
#from GCNet.modules.GCNet import L1Loss
import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from models.Pool2x2 import GANet
from models.DSMNet2x2 import DSMNet
from dataloader.data import get_test_set
from pylab import *
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--dataset', type=str, default='kitti2015_train.list', help='dataset?')
parser.add_argument('--data_path', type=str, required=True, help="data root")
parser.add_argument('--test_list', type=str, required=True, help="training list")
parser.add_argument('--save_path', type=str, default='./result/', help="location to save result")

opt = parser.parse_args()


print(opt)

cuda = opt.cuda
#cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

#torch.manual_seed(opt.seed)
#if cuda:
#    torch.cuda.manual_seed(opt.seed)
#print('===> Loading datasets')


print('===> Building model')
model = DSMNet(opt.max_disp)
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#quit()

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage.cuda())
        msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(opt.resume))
        print(msg)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def test_transform(temp_data, crop_height, crop_width):
    _, h, w=np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3,crop_height,crop_width],'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w

def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
#    opt.crop_height = int(height/48.)*48
#    opt.crop_width = int(width/48.)*48
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    if len(size)>2:
        r = left[:, :, 0]
        g = left[:, :, 1]
        b = left[:, :, 2]
    else:
        r = left[:, :]
        g = left[:, :]
        b = left[:, :]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    if len(size)>2:
        r = right[:, :, 0]
        g = right[:, :, 1]
        b = right[:, :, 2]	
    else:
        r = right[:, :]
        g = right[:, :]
        b = right[:, :]	
    #r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data

def test(leftname, rightname, savename):
  #  count=0
    
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    
    input1 = Variable(input1, requires_grad = False)
    input2 = Variable(input2, requires_grad = False)

    model.eval()
#    model.train()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()
    with torch.no_grad():

        print(input1.shape, input2.shape)
        prediction = model(input1, input2)
        prediction = prediction.detach().cpu().numpy()
        print(prediction.shape)
        if height <= opt.crop_height and width <= opt.crop_width:
            prediction = prediction[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
        else:
            prediction = prediction[0, :, :]
        return prediction
    skimage.io.imsave(savename, (prediction * 256).astype('uint16'))

   
if __name__ == "__main__":
    file_path = opt.data_path
    file_list = opt.test_list
    f = open(file_list, 'r')
    filelist = f.readlines()
    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for index in range(0,len(filelist)):
        current_file = filelist[index]
        if opt.dataset == 'kitti2015':
            leftname = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[:-1]
        if opt.dataset == 'kitti':
            leftname = file_path + 'colored_0/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'colored_1/' + current_file[0: len(current_file) - 1]
            savename = opt.save_path + current_file[:-1]
        if opt.dataset == 'middlebury':
            current_file = current_file.strip().split(' ')
            leftname = file_path + 'rgb/' + current_file[0] #current_file[0: len(current_file) - 1]
            rightname = file_path + 'rgb/' + current_file[1] # right_file[0: len(current_file) - 1]
            savename = opt.save_path + str(index) + '.png'
        if opt.dataset=='cityscapes':
            leftname = file_path + 'left/' + current_file[0: len(current_file) - 1]
            rightname = file_path + 'right/' + current_file[0: len(current_file) - 16] + 'rightImg8bit.png'
            savename = opt.save_path + 'cityscape' + str(index) + '.png'
        if opt.dataset == 'eth3d':
            leftname = file_path + current_file[0: len(current_file) - 1]
            rightname = file_path + current_file[0: len(current_file) - 6] + '1.png'
            savename = file_path + current_file[0: len(current_file) - 8] +'eth3d.png'
            savename = opt.save_path + 'eth3d' + str(index) + '-2.png'
        if opt.dataset == 'sceneflow':
            leftname = file_path  + current_file[0: len(current_file) - 1]
            rightname = file_path + current_file[0: len(current_file) - 14] + 'right/' + current_file[len(current_file) - 9:len(current_file) - 1]
            savename = opt.save_path + current_file[0: len(current_file) - 1]
            savename = opt.save_path + 'sceneflow' + str(index) + '.png'
        prediction = test(leftname, rightname, savename)
        skimage.io.imsave(savename, (prediction * 256).astype('uint16'))
        

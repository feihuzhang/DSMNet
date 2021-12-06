import numpy as np
import random
import math
from PIL import Image

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter
import torch.nn.functional as F


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow, shift=3):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        #min_scale = np.maximum(
        min_scale = np.minimum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))
        #min_scale = self.min_scale

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)


        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
        h, w = img1.shape[:2]
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        if h > crop_height and w <= crop_width:
            temp_img = np.zeros([h+shift, crop_width + shift, 3], 'float32') 
            temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = img1
            img1 = temp_img
            temp_img = np.zeros([h+shift, crop_width + shift, 3], 'float32')
            temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = img2
            img2 = temp_img
            temp_img = np.zeros([h+shift, crop_width + shift, 2], 'float32') + 1e4
            temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = flow
            flow = temp_img
            #temp_img = np.zeros([h+shift, crop_width + shift], 'float32')
            #temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = valid
            #valid = temp_img
        if h < crop_height and w > crop_width:
            temp_img = np.zeros([crop_height + shift, w + shift, 3], 'float32') 
            temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = img1
            img1 = temp_img
            temp_img = np.zeros([crop_height+shift, w + shift, 3], 'float32')
            temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = img2
            img2 = temp_img
            temp_img = np.zeros([crop_height+shift, w + shift, 2], 'float32') + 1e4
            temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = flow
            flow = tem_img
            #temp_img = np.zeros([crop_height+shift, w + shift], 'float32') 
            #temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = valid
            #valid = tem_img
   
        if h <= crop_height and w <= crop_width:
            temp_img = np.zeros([crop_height + shift, crop_width + shift, 3], 'float32')
            temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = img1
            img1 = temp_img
            temp_img = np.zeros([crop_height + shift, crop_width + shift, 3], 'float32')
            temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = img2
            img2 = temp_img
            temp_img = np.zeros([crop_height + shift, crop_width + shift, 2], 'float32') + 1e4
            temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = flow
            flow = temp_img
            #temp_img = np.zeros([crop_height + shift, crop_width + shift], 'float32')
            #temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = valid
            #valid = temp_img

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
        shift_x = 0
        if shift > 0:
            shift_x = random.randint(-shift, shift)
            shift_y = random.randint(-shift, shift)
            if shift_x + x0 < 0 or shift_x + x0 + self.crop_size[1] > img1.shape[1]:
                shift_x = 0
            if shift_y + y0 < 0 or shift_y + y0 + self.crop_size[0] > img1.shape[0]:
                shift_y = 0
        
        img1 = img1[y0:y0+self.crop_size[0], x0+shift_x:x0+shift_x+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0+shift_x:x0+shift_x+self.crop_size[1]]
        flow[:,:,0] += shift_x

        return img1, img2, flow
    def normalize(self, img):
        img = np.float32(img)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        img[:,:,0] = r
        img[:,:,1] = g
        img[:,:,2] = b

        img[:, :, 0] = (r - np.mean(r[:])) / (np.std(r[:]) + 1e-6)
        img[:, :, 1] = (g - np.mean(g[:])) / (np.std(g[:]) + 1e-6)
        img[:, :, 2] = (b - np.mean(b[:])) / (np.std(b[:]) + 1e-6)
        return img

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
#        img1 = self.normalize(img1)
#        img2 = self.normalize(img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3/3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid, shift=3):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        #min_scale = np.maximum(
        min_scale = np.minimum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))
        #min_scale = self.min_scale

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        if self.do_flip:
            if np.random.rand() < 0.5: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]
        h, w = img1.shape[:2]
        crop_height, crop_width = self.crop_size[0], self.crop_size[1]
        if h >= crop_height and w < crop_width:
            temp_img = np.zeros([h+shift, crop_width + shift, 3], 'float32') 
            temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = img1
            img1 = temp_img
            temp_img = np.zeros([h+shift, crop_width + shift, 3], 'float32')
            temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = img2
            img2 = temp_img
            temp_img = np.zeros([h+shift, crop_width + shift, 2], 'float32') + 1e4
            temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = flow
            flow = temp_img
            temp_img = np.zeros([h+shift, crop_width + shift], 'float32')
            temp_img[h + shift - h: h + shift, crop_width + shift - w: crop_width + shift] = valid
            valid = temp_img
        elif h < crop_height and w >= crop_width:
            temp_img = np.zeros([crop_height + shift, w + shift, 3], 'float32') 
            temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = img1
            img1 = temp_img
            temp_img = np.zeros([crop_height+shift, w + shift, 3], 'float32')
            temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = img2
            img2 = temp_img
            temp_img = np.zeros([crop_height+shift, w + shift, 2], 'float32') + 1e4
            temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = flow
            flow = tem_img
            temp_img = np.zeros([crop_height+shift, w + shift], 'float32') 
            temp_img[crop_height + shift - h: crop_height + shift, w + shift - w: w + shift] = valid
            valid = tem_img
   
        elif h <= crop_height and w <= crop_width:
            temp_img = np.zeros([crop_height + shift, crop_width + shift, 3], 'float32')
            temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = img1
            img1 = temp_img
            temp_img = np.zeros([crop_height + shift, crop_width + shift, 3], 'float32')
            temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = img2
            img2 = temp_img
            temp_img = np.zeros([crop_height + shift, crop_width + shift, 2], 'float32') + 1e4
            temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = flow
            flow = temp_img
            temp_img = np.zeros([crop_height + shift, crop_width + shift], 'float32')
            temp_img[crop_height + shift - h: crop_height + shift, crop_width + shift - w: crop_width + shift] = valid
            valid = temp_img

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        shift_x = 0
        if shift > 0:
            shift_x = random.randint(-shift, shift)
            shift_y = random.randint(-shift, shift)
            if shift_x + x0 < 0 or shift_x + x0 + self.crop_size[1] > img1.shape[1]:
                shift_x = 0
            if shift_y + y0 < 0 or shift_y + y0 + self.crop_size[0] > img1.shape[0]:
                shift_y = 0
        
        img1 = img1[y0:y0+self.crop_size[0], x0+shift_x:x0+shift_x+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0+shift_x:x0+shift_x+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0+shift_x:x0+shift_x+self.crop_size[1]]
        flow[:,:,0] += shift_x
        return img1, img2, flow, valid

    def normalize(self, img):
        img = np.float32(img)
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        img[:,:,0] = r
        img[:,:,1] = g
        img[:,:,2] = b

        img[:, :, 0] = (r - np.mean(r[:])) / (np.std(r[:]) + 1e-6)
        img[:, :, 1] = (g - np.mean(g[:])) / (np.std(g[:]) + 1e-6)
        img[:, :, 2] = (b - np.mean(b[:])) / (np.std(b[:]) + 1e-6)
        return img


    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
    
        #img1 = self.normalize(img1)
        #img2 = self.normalize(img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid

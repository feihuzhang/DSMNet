# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from dataloader.utils import frame_utils
from dataloader.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, vkitti=False, carla=False):
        self.augmentor = None
        self.sparse = sparse
        self.vkitti = vkitti
        self.carla = carla
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            if self.vkitti:
                #flow, valid = frame_utils.readFlowVKITTI(self.flow_list[index])
                #flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
                flow, valid = frame_utils.readDispVKITTI(self.flow_list[index])
            elif self.carla:
                flow, valid = frame_utils.readDispCarla(self.flow_list[index], self.extra_info[index])
            else:
                #flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
                flow, valid = frame_utils.readDispKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        assert (img1 is not None) and (img2 is not None) and (flow is not None)

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        #flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow[:,:,0]).abs().float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            #valid = (flow[0].abs() < 2000) & (flow[1].abs() < 2000)
            valid = (flow.abs() < 2000)
        flow[valid<1] = 20000

        return img1, img2, flow
        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='/export/work/feihu/flow/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='/export/work/feihu/flow/SceneFlow', dstype='frames_cleanpass'):
        super(FlyingThings3D, self).__init__(aug_params)

        #limage_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
        #limage_dirs = sorted([osp.join(f, "left") for f in image_dirs])
        #rimage_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
        #rimage_dirs = sorted([osp.join(f, "right") for f in image_dirs])
        #flow_dirs = sorted(glob(osp.join(root, 'disparity/TRAIN/*/*')))
        #flow_dirs = sorted([osp.join(f, "left") for f in flow_dirs])
        limage_dirs = []
        rimage_dirs = []
        flow_dirs = []

        for subset in ["family_x2", "funnyworld_camera2_augmented0_x2", "lonetree_difftex2_x2", "flower_storm_augmented0_x2", "funnyworld_camera2_augmented1_x2", "lonetree_difftex_x2", "treeflight_augmented0_x2", "a_rain_of_stones_x2", "flower_storm_augmented1_x2", "funnyworld_camera2_x2", "lonetree_winter_x2", "treeflight_augmented1_x2", "eating_camera2_x2", "flower_storm_x2", "funnyworld_x2", "lonetree_x2", "treeflight_x2", "eating_naked_camera2_x2", "funnyworld_augmented0_x2", "lonetree_augmented0_x2", "eating_x2", "funnyworld_augmented1_x2", "lonetree_augmented1_x2", "top_view_x2"]:
            image_dirs = sorted(glob(osp.join(root, dstype, subset)))
            limage_dirs += sorted([osp.join(f, "left") for f in image_dirs])
            image_dirs = sorted(glob(osp.join(root, dstype, subset)))
            rimage_dirs += sorted([osp.join(f, "right") for f in image_dirs])

            fdirs = sorted(glob(osp.join(root, 'disparity/' + subset)))
            flow_dirs += sorted([osp.join(f, "left") for f in fdirs])
        for subset in ["15mm_focallength", "35mm_focallength", "TRAIN"]:
            image_dirs = sorted(glob(osp.join(root, dstype, subset+'/*/*')))
            limage_dirs += sorted([osp.join(f, "left") for f in image_dirs])
            image_dirs = sorted(glob(osp.join(root, dstype, subset+'/*/*')))
            rimage_dirs += sorted([osp.join(f, "right") for f in image_dirs])

            fdirs = sorted(glob(osp.join(root, 'disparity/' + subset +'/*/*')))
            flow_dirs += sorted([osp.join(f, "left") for f in fdirs])
        #print(len(limage_dirs))
        #print(len(flow_dirs))
        #print(len(rimage_dirs))

        for ldir, rdir, fdir in zip(limage_dirs, rimage_dirs, flow_dirs):
            limages = sorted(glob(osp.join(ldir, '*.png')) )
            rimages = sorted(glob(osp.join(rdir, '*.png')) )
            flows = sorted(glob(osp.join(fdir, '*.pfm')) )
            for i in range(len(flows)):
                self.image_list += [ [limages[i], rimages[i]] ]
                self.flow_list += [ flows[i] ]
      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/export/work/feihu/kitti2015'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_3/*_10.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'disp_occ_0/*_10.png')))
class KITTI2012(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/export/work/feihu/kitti2012'):
        super(KITTI2012, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'colored_0/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'colored_1/*_10.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'disp_occ/*_10.png')))


class CarlaStereo(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/export/work/feihu/Carla', file_list='carla.list'):
        super(CarlaStereo, self).__init__(aug_params, sparse=True, carla=True)
        if split == 'testing':
            self.is_test = True

        filename = osp.join(root, file_list)
        f = open(filename)
        files = f.readlines()
        for i in range(len(files)):
            cur_file = files[i]
            cur_file = cur_file.split()
            img1 = osp.join(root, 'rgb/' + cur_file[0])
            img2 = osp.join(root, 'rgb/' + cur_file[1])
        #flow = osp.join(root, 'depth/' + cur_file[0] + ' ' + cur_file[2] + ' ' cur_file[3])
            flow = osp.join(root, 'depth/' + cur_file[0])

            self.extra_info += [ [cur_file[2], cur_file[3]] ]
            self.image_list += [ [img1, img2] ]

            if split == 'training':
                self.flow_list += [flow]

class VKITTI(FlowDataset):
    def __init__(self, aug_params=None, root='/export/work/feihu/vkitti'):
        super(VKITTI, self).__init__(aug_params, sparse=True, vkitti=True)

        #for cam in ['Camera_0', 'Camera_1']:
            #for direction in ['forwardFlow', 'backwardFlow']:
        #    for direction in ['forwardFlow']:
        for dstype in ['15-deg-left','15-deg-right','30-deg-left','30-deg-right','clone','fog','morning','overcast','rain', 'sunset']:
        #for dstype in ['15-deg-left','15-deg-right','30-deg-left','30-deg-right','clone','morning','overcast', 'sunset']:
            image_dirs = sorted(glob(osp.join(root, 'Scene*/'+ dstype +'/frames/rgb')))
            left_dirs = sorted([osp.join(f, "Camera_0") for f in image_dirs])
            right_dirs = sorted([osp.join(f, "Camera_1") for f in image_dirs])

            flow_dirs = sorted(glob(osp.join(root, 'correct/Scene*/'+ dstype +'/frames')))
            flow_dirs = sorted([osp.join(f, "depth", "Camera_0") for f in flow_dirs])

            for ldir, rdir, fdir in zip(left_dirs, right_dirs, flow_dirs):
                limages = sorted(glob(osp.join(ldir, '*.jpg')) )
                rimages = sorted(glob(osp.join(rdir, '*.jpg')) )
                flows = sorted(glob(osp.join(fdir, '*.png')) )
                for i in range(len(flows)):
                        self.image_list += [ [limages[i], rimages[i]] ]
                        self.flow_list += [ flows[i] ]


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1
class Cityscapes(FlowDataset):
    def __init__(self, aug_params=None, root='/export/work/feihu/cityscapes'):
        super(Cityscapes, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        self.is_test=True
        #while 1:
#            flows = sorted(glob(os.path.join(root, 'lindau/%06d_*.png' % seq_ix)))
        #images = sorted(glob(os.path.join(root, 'part2/*.png')))
        images = sorted(glob(os.path.join(root, '*.png')))


        for i in range(len(images)-1):
            frame_id = images[i].split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.flow_list += [images[i]]
            self.image_list += [ [images[i], images[i+1]] ]

def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """
    crop_size = [args.crop_height, args.crop_width]
    if args.stage == 'synthetic':

        aug_params = {'crop_size': crop_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': False}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        aug_params = {'crop_size': crop_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        carla_dataset = CarlaStereo(aug_params)
        vkitti_dataset = VKITTI(aug_params)
        kitti_dataset = KITTI(aug_params)
        kitti2012_dataset = KITTI2012(aug_params)
#        print(len(vkitti_dataset), len(carla_dataset), len(kitti_dataset), len(kitti2012_dataset), len(clean_dataset), len(final_dataset))
        train_dataset = clean_dataset + final_dataset + vkitti_dataset + carla_dataset

    
    elif args.stage == 'things':
        aug_params = {'crop_size': crop_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': False}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training', root="/export/work/feihu/kitti2015")
    elif args.stage == 'kitti2012':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI2012(aug_params, split='training', root="/export/work/feihu/kitti2012")

    print('Training with %d image pairs' % len(train_dataset))
    return train_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=4, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader


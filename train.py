from __future__ import print_function
import argparse
from math import log10

from libs.GANet.modules.GANet import MyLoss2
import sys
import shutil
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.DSMNet import DSMNet
from models.DSMNet2x2 import DSMNet
import torch.nn.functional as F
from dataloader.data import get_training_set, get_test_set
import torch.multiprocessing as mp
import numpy as np
from dataloader.datasets import fetch_dataloader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GC-Net Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--manual_seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train2.list', help="training list")
#parser.add_argument('--training_list', type=str, default='./lists/debug.list', help="training list")
parser.add_argument('--data_path', type=str, default='/export/work/feihu/flow/SceneFlow/', help="data root")
parser.add_argument('--val_list', type=str, default='./lists/kitti2015_train.list', help="validation list")
parser.add_argument('--save_path', type=str, default='./checkpoints/', help="location to save models")
parser.add_argument('--gpu',  default='0,1,2,3,4,5,6,7', type=str, help="gpu idxs")
parser.add_argument('--workers', type=int, default=16, help="workers")
parser.add_argument('--world_size', type=int, default=1, help="world_size")
parser.add_argument('--rank', type=int, default=0, help="rank")
parser.add_argument('--dist_backend', type=str, default="nccl", help="dist_backend")
parser.add_argument('--dist_url', type=str, default="tcp://127.0.0.1:6789", help="dist_url")
parser.add_argument('--distributed', type=int, default=0, help="distribute")
parser.add_argument('--sync_bn', type=int, default=0, help="sync bn")
parser.add_argument('--multiprocessing_distributed', type=int, default=0, help="multiprocess")
parser.add_argument('--freeze_bn', type=int, default=0, help="freeze bn")
parser.add_argument('--stage', type=str, default='synthetic', help="training stage: 1) things 2) synthetic 3) real 4) mixed.")


#print(args)

#cuda = args.cuda
#cuda = True
#if cuda and not torch.cuda.is_available():
#    raise Exception("No GPU found, please run without --cuda")

#torch.manual_seed(args.seed)
#if cuda:
#    torch.cuda.manual_seed(args.seed)

#print('===> Loading datasets')

#print('===> Building model')
def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.gpu = (args.gpu).split(',')
   # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpu.split(','))
    #args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    if args.manual_seed is not None:
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = True
        cudnn.deterministic = True
    args.ngpus_per_node = len(args.gpu)
    if len(args.gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        main_worker(args.gpu, args.ngpus_per_node, args)
    else:
        args.sync_bn = True
        args.distributed = True
        args.multiprocessing_distributed = True
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
        #print(args)
        #quit()
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)
def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    model = DSMNet()
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        args.testBatchSize = int(args.testBatchSize / args.ngpus_per_node)
        args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()


    criterion = MyLoss2(thresh=3, alpha=2)
    optimizer=optim.Adam(model.parameters(), lr=args.lr,betas=(0.9,0.999))
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            msg=model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                print("=> loaded checkpoint '{}'".format(args.resume))
                print(msg)
                sys.stdout.flush()
        else:
            if main_process():
                print("=> no checkpoint found at '{}'".format(args.resume))

    #train_set = get_training_set(args.data_path, args.training_list, [args.crop_height, args.crop_width], args.left_right, args.kitti, args.kitti2015, args.shift)
    train_set = fetch_dataloader(args)
    #test_set = get_test_set(args.data_path, args.val_list, [384,1248], args.left_right, args.kitti, args.kitti2015)
    test_set = get_test_set('/export/work/feihu/kitti2015/training/', args.val_list, [384,1280], args.left_right, False, True)
    #training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=True, drop_last=True)
    #testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.testBatchSize, shuffle=False)
    sys.stdout.flush()
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    else:
        train_sampler = None
        val_sampler = None
    training_data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    testing_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.testBatchSize, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    error = 100
    for epoch in range(1, args.nEpochs + 1):
#        if args.kitti or args.kitti2015:
        adjust_learning_rate(optimizer, epoch)
        if args.distributed:
            train_sampler.set_epoch(epoch-1)

        #loss = val(testing_data_loader, model)
        #quit()
        train(training_data_loader, model, optimizer, epoch)
        is_best = False
        loss = val(testing_data_loader, model)
        if loss < error:
            error = loss
            is_best = True
        if main_process():
            save_checkpoint(args.save_path, epoch,{
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best)

    save_checkpoint(args.save_path, args.nEpochs,{
            'epoch': args.nEpochs,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best)




def train(training_data_loader, model, optimizer, epoch):
    epoch_loss = 0
    epoch_error0 = 0
    epoch_error1 = 0
    epoch_error2 = 0
    valid_iteration = 0
    model.train()
    if args.freeze_bn:
        model.module.freeze_bn()
        if main_process():
            print("freezing bn...")
            sys.stdout.flush()
    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), Variable(batch[2], requires_grad=False)
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        target=torch.squeeze(target,1)
        mask = target < args.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid > 0:

            optimizer.zero_grad()
            disp0, disp1, disp2=model(input1,input2)
#            disp0, disp1, disp2 = chp.checkpoint(model, input1, input2)

            #if args.kitti or args.kitti2015:
            #    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  criterion(disp2[mask], target[mask])
#                loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
            #else:
            loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') +  F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
#            loss = 0.2 * criterion(disp0[mask], target[mask]) + 0.6 * criterion(disp1[mask], target[mask]) + criterion(disp2[mask], target[mask])
            loss.backward()
            optimizer.step()
            error0 = torch.mean(torch.abs(disp0[mask] - target[mask])) 
            error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
            error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
            valid_iteration += 1
            loss_value = loss.detach()

            if args.multiprocessing_distributed:
                count = target.new_tensor([1], dtype=torch.long)
                dist.all_reduce(loss_value), dist.all_reduce(error0), dist.all_reduce(error1), dist.all_reduce(error2), dist.all_reduce(count)
                n = count.item()
                loss_value, error0, error1, error2 = loss_value / n, error0 / n, error1 / n, error2 / n
                epoch_loss += loss_value.item()
                epoch_error0 += error0.item()
                epoch_error1 += error1.item()
                epoch_error2 += error2.item()      

            if main_process():
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} {:.4f} {:.4f})".format(epoch, iteration, len(training_data_loader), loss_value.item(), error0.item(), error1.item(), error2.item()))
            sys.stdout.flush()

    if main_process():
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Error: ({:.4f} {:.4f} {:.4f})".format(epoch, epoch_loss / valid_iteration,epoch_error0/valid_iteration,epoch_error1/valid_iteration,epoch_error2/valid_iteration))

def val(testing_data_loader, model):
    epoch_error = 0
    epoch_error_rate = 0
    valid_iteration = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
        input1 = input1.cuda(non_blocking=True)
        input2 = input2.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target=torch.squeeze(target,1)
        mask = target < args.max_disp
        mask.detach_()
        valid=target[mask].size()[0]
        if valid>0:
            with torch.no_grad():
                disp = model(input1,input2)
                error = torch.mean(torch.abs(disp[mask] - target[mask])) 
                error_rate = (torch.abs(disp[mask] - target[mask]) > 3.0).sum() * 1.0 /mask.sum()
                valid_iteration += 1
            if args.multiprocessing_distributed:
                count = target.new_tensor([1], dtype=torch.long)
                dist.all_reduce(error)
                dist.all_reduce(error_rate)
                dist.all_reduce(count)
                n = count.item()
                error /= n
                error_rate /= n
                epoch_error += error.item()
                epoch_error_rate += error_rate.item()

            if main_process():
                print("===> Test({}/{}): Error: ({:.4f} {:.4f})".format(iteration, len(testing_data_loader), error.item(), error_rate.item()))
            sys.stdout.flush()

    if main_process():
        print("===> Test: Avg. Error: ({:.4f} {:.4f})".format(epoch_error/valid_iteration, epoch_error_rate/valid_iteration))

    return epoch_error/valid_iteration

def save_checkpoint(save_path, epoch,state, is_best):
    filename = save_path + "_epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + '_best_epoch.pth')
    print("Checkpoint saved to {}".format(filename))

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 100:
       lr = args.lr
    else:
       lr = args.lr*0.1
    if main_process():
        print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()

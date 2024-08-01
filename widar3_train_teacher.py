from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np

from kd_losses import *

from lr_scheduler import LR_Scheduler
from network import FeatureExtracter2d, ActRecognizer, ConvNet
import dataset_widar3

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_root', type=str, default='./result', help='models and logs are saved here')
parser.add_argument('--domain_name', type=str)
parser.add_argument('--train_domain', type=int)
parser.add_argument('--test_domain', type=int)

# net choice
parser.add_argument('--name', type=str, help='model name of teacher', default='Convnet')    # Convnet / resnet18

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loaders')
parser.add_argument('--opt', type=str, default='Adam', help='DL optimizer')
parser.add_argument('--adjust_lr', type=str, default='step', help='the method to adjust learning rate')
parser.add_argument('--warmup_epoch', type=int, default=0, help='the number of warmup epochs')
parser.add_argument('--step_gamma', type=float, default=0, help='decay weight for step policy')
parser.add_argument('--step_size', type=int, default=0, help='decay step for step policy')
parser.add_argument('--T_mult', type=int, default=1, help='a factor for CosineAnnealingWarmRestarts')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--num_class', type=int, default=6, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)

# hyperparameter
parser.add_argument('--feat_dim', type=int, default=512)
parser.add_argument('--softplus_beta', type=float, default=1.0)


# others
parser.add_argument('--seed', type=int, default=2, help='random seed')

# # net and dataset choosen
# parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
# parser.add_argument('--net_name', type=str, required=True, help='name of basenet')  # resnet20/resnet110


args, unparsed = parser.parse_known_args()
print(f"Domain NAME: {args.domain_name}")

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_best__checkpoint, save_current_checkpoint
from utils import create_exp_dir, count_parameters_in_MB
from utils import adjust_lr, get_scheduler, get_model, get_optimizer

create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

board_path = os.path.join(args.save_root, 'board')
create_exp_dir(board_path)
writer = SummaryWriter(board_path)


def main():
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

        # torch.cuda.set_device(1)
        logging.info('------------ GPU Inffor -------------------')
        logging.info(torch.cuda.get_device_name())


    logging.info('----------- Network Initialization --------------')
    net = get_model(args.name, args)
    if args.cuda:
        net = net.cuda()
    logging.info('%s', net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info('-----------------------------------------------')

    # save initial parameters
    logging.info('Saving initial parameters......')
    save_path = os.path.join(args.save_root, 'initial.pth.tar')
    torch.save({
        'epoch': 0,
        'net': net.state_dict(),
        'prec@1': 0.0,
        'prec@5': 0.0,
    }, save_path)

    # define loss functions
    if args.cuda:
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = get_optimizer(net, args)
    lr_schduler = get_scheduler(optimizer, args)

    train_loader, test_loader =  dataset_widar3.get_train_and_valid_loader(args.train_domain, args)

    best_top1 = 0
    best_top3 = 0
    best_epoch = 0
    for epoch in range(1, args.epochs+1):

        # train one epoch
        epoch_start_time = time.time()
        train(train_loader, net, optimizer, criterion, epoch)

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top3 = test(test_loader, net, criterion)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        # update lr
        logging.info('Epoch: {}  lr: {:.6f}'.format(epoch, optimizer.param_groups[0]['lr']))
        # adjust_lr(optimizer, epoch, args, logging)
        lr_schduler.step()

        # save model
        is_best = False
        if test_top1 > best_top1:
            best_top1 = test_top1
            best_top3 = test_top3
            is_best = True
            best_epoch = epoch
        logging.info('Saving models......')
        save_best__checkpoint(epoch, {
            'epoch': epoch,
            'net': net.state_dict(),
            'prec@1': test_top1,
            'prec@3': test_top3,
        }, is_best, args.save_root)
        logging.info(f'best epoch is {best_epoch}, prec@1: {round(best_top1, 2)}, prec@3: {round(best_top3, 2)}')
        logging.info('')

    # save current model
    save_current_checkpoint({
        'epoch' : epoch,
        'net' : net.state_dict(),
        'prec@1' : test_top1,
        'prec@3' : test_top3,
    }, args.save_root)


def train(train_loader, net, optimizer, criterion, epoch):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top3       = AverageMeter()

    net.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        x = x.float()
        target = target.long()

        if args.cuda:
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        out = net(x)
        loss = criterion(out, target)

        prec1, prec3 = accuracy(out, target, topk=(1,3))
        losses.update(loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        top3.update(prec3.item(), x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or (epoch == 1 and i < args.print_freq):
            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'BatchTime:{batch_time.val:.4f} '
                       'DataTime:{data_time.val:.4f}  '
                       'loss:{losses.val:.4f}({losses.avg:.4f})  '
                       'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                       'prec@3:{top3.val:.2f}({top3.avg:.2f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                       losses=losses, top1=top1, top3=top3))
            logging.info(log_str)


def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1   = AverageMeter()
    top3   = AverageMeter()

    net.eval()

    end = time.time()
    for i, (x, target) in enumerate(test_loader, start=1):

        x = x.float()
        target = target.long()

        if args.cuda:
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            out = net(x)
            loss = criterion(out, target)

        prec1, prec3 = accuracy(out, target, topk=(1,3))
        losses.update(loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        top3.update(prec3.item(), x.size(0))

    f_l = [losses.avg, top1.avg, top3.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@3: {:.2f}'.format(*f_l))

    return top1.avg, top3.avg



if __name__ == '__main__':
    main()
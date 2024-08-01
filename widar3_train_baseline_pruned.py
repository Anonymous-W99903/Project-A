"""
train baseline on N-1 domain and test on the N th domain
save the model which has best performance on domain N
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import sys
import time
import logging
import argparse
import numpy as np



#################### FOR DEBUG #############################
preread = True
# preread = False
#################### END DEBUG #############################

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_root', type=str, default='./result', help='models and logs are saved here')
parser.add_argument('--domain_name', type=str)
# parser.add_argument('--train_domain', type=int)
parser.add_argument('--train_domain', nargs='+', help='train domains', required=True)
parser.add_argument('--test_domain', type=int)
parser.add_argument('--pruned_model_path', type=str, help="path to the pruned student model", required=True)
# net choice
parser.add_argument('--name', type=str, default='Convnet', help='model name of baseline')    # Convnet / resnet18

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loaders')
parser.add_argument('--opt', type=str, default='Adam', help='DL optimizer')
parser.add_argument('--adjust_lr', type=str, default='step', help='the method to adjust learning rate')
parser.add_argument('--warmup_epoch', type=int, default=0, help='the number of warmup epochs')
parser.add_argument('--step_gamma', type=float, default=0, help='decay weight for step policy')
parser.add_argument('--step_size', type=int, default=0, help='decay step for step policy')
parser.add_argument('--T_mult', type=int, default=1, help='a factor for CosineAnnealingWarmRestarts')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--num_class', type=int, default=6, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)

# hyperparameter
parser.add_argument('--feat_dim', type=int, default=512)
parser.add_argument('--softplus_beta', type=float, default=1.0)


# others
parser.add_argument('--seed', type=int, default=0, help='random seed')

# # net and dataset choosen
# parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
# parser.add_argument('--net_name', type=str, required=True, help='name of basenet')  # resnet20/resnet110

args, unparsed = parser.parse_known_args()
if isinstance(args.train_domain, list):
    args.train_domain = list(map(int, args.train_domain))

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
from kd_losses import *

from lr_scheduler import LR_Scheduler
from network import FeatureExtracter2d, ActRecognizer, ConvNet
import dataset_widar3
import torch_pruning as tp

if os.path.exists(args.save_root):
	shutil.rmtree(args.save_root)
create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

board_path = os.path.join(args.save_root, 'board')
create_exp_dir(board_path)
writer = SummaryWriter(board_path)


# Teacher model path (please make key-value pair consistent)
class DOMAIN :
    LOC1 = 1
    LOC2 = 2
    LOC3 = 3
    LOC4 = 4
    LOC5 = 5

DOMAIN_LIST = [1,2,3,4,5]

# TEACHER_MODEL_PATH = {
#     DOMAIN.LOC1 : "/home/yueyang/code/results/widar3/loc/T1_5conv_14_softplus",
#     DOMAIN.LOC2 : "/home/yueyang/code/results/widar3/loc/T2_5conv_14_softplus",
#     DOMAIN.LOC3 : "/home/yueyang/code/results/widar3/loc/T3_5conv_14_softplus",
#     DOMAIN.LOC4 : "/home/yueyang/code/results/widar3/loc/T4_5conv_14_softplus",
#     DOMAIN.LOC5 : "/home/yueyang/code/results/widar3/loc/T5_5conv_14_softplus",
# }

fun1 = lambda x : f'results/widar3/loc/baseline{x}/model_best.pth.tar'
BASELINE_MODEL_PATH = {x : fun1(x) for x in DOMAIN_LIST}

INT_DOMAIN_NAME_DICT = {
    1 : 'loc1',
    2 : 'loc2',
    3 : 'loc3',
    4 : 'loc4',
    5 : 'loc5',
}

def reset_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            module.reset_parameters()


def main():
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        # torch.cuda.set_device(args.gpu_id)
        # torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

        logging.info('------------ GPU Inffor -------------------')
        logging.info(torch.cuda.get_device_name())


    logging.info('----------- Network Initialization --------------')
    net = get_model(args.name, args)
    # load pruned model
    pruned_checkpoint = torch.load(args.pruned_model_path)
    load_pretrained_model(net, pruned_checkpoint['net'])
    
    net = net.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    example_inputs = torch.randn(64, 6, 121, 121).to(device)

    DG = tp.DependencyGraph().build_dependency(net, example_inputs=example_inputs)

    def prune_conv(DG, conv, channels_to_prune):
        pruning_group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=channels_to_prune)
        pruning_group.prune()

    def prune_linear(DG, linear, channels_to_prune):
        pruning_group = DG.get_pruning_group(linear, tp.prune_linear_out_channels, idxs=channels_to_prune)
        pruning_group.prune()

    modules = list(net.named_modules())
    for i, (name, module) in enumerate(modules):
        if isinstance(module, nn.Linear) and i == len(modules) - 1:
            continue
        if isinstance(module, nn.Conv2d):
            weight_mask = module.weight != 0
            channels_to_prune = torch.where(torch.sum(weight_mask, dim=(1, 2, 3)) == 0)[0].tolist()
            if channels_to_prune:
                prune_conv(DG, module, channels_to_prune)
        elif isinstance(module, nn.Linear):
            weight_mask = module.weight != 0
            channels_to_prune = torch.where(torch.sum(weight_mask, dim=1) == 0)[0].tolist()
            if channels_to_prune:
                prune_linear(DG, module, channels_to_prune)

    # Reset parameters of the adjusted model
    reset_parameters(net)
    
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

    # define dataloader
    train_loader, valid_loader = dataset_widar3.get_train_and_valid_loader(args.train_domain, args,
                                    domain_name=args.domain_name, preread=preread, return_domain_label=False)
    test_dataset = dataset_widar3.DomainSet(args.data_dir, args.domain_name, args.test_domain, preread=preread,
                                            isTest=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    best_top1 = 0
    best_top3 = 0
    best_epoch = 0
    for epoch in range(1, args.epochs+1):

        # train one epoch
        epoch_start_time = time.time()
        train_loss, train_prec = train(train_loader, net, optimizer, criterion, epoch, lr_schduler)
        v_loss, v_prec = valid(valid_loader, net, criterion)

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top3, test_loss = test(test_loader, net, criterion)

        # record results of the epoch(avg of iterations)
        writer.add_scalars('total_loss', {'train' : train_loss.avg, 'valid' : v_loss.avg}, epoch)
        writer.add_scalars(f'prec', {'train' : train_prec.avg, 'valid' : v_prec.avg}, epoch)
        writer.add_scalar('test/prec_1', test_top1, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        # update lr
        # logging.info('Epoch: {}  lr: {:.6f}'.format(epoch, optimizer.param_groups[0]['lr']))
        # adjust_lr(optimizer, epoch, args, logging)
        # lr_schduler.step()

        # save model
        # is_best = False
        # if test_top1 > best_top1:
        #     best_top1 = test_top1
        #     best_top3 = test_top3
        #     is_best = True
        #     best_epoch = epoch
        # logging.info('Saving models......')
        # save_best__checkpoint(epoch, {
        #     'epoch': epoch,
        #     'net': net.state_dict(),
        #     'prec@1': test_top1,
        #     'prec@3': test_top3,
        # }, is_best, args.save_root)
        # logging.info(f'best epoch is {best_epoch}, prec@1: {round(best_top1, 2)}, prec@3: {round(best_top3, 2)}')
        # logging.info('')

        # save current model
        save_current_checkpoint({
            'epoch' : epoch,
            'net' : net.state_dict(),
            'prec@1' : test_top1,
            'prec@3' : test_top3,
        }, args.save_root)


def train(train_loader, net, optimizer, criterion, epoch, scheduler):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    top1       = AverageMeter()
    top3       = AverageMeter()

    iters = len(train_loader)

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
        # update lr
        scheduler.step(epoch - 1 + i / iters)

        batch_time.update(time.time() - end)
        end = time.time()

    log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
               'BatchTime:{batch_time:.4f} '
               'DataTime:{data_time:.4f}  '
               'lr: {lr:.6f} loss:{losses:.4f}  '
               'prec@1:{top1:.2f}  '
               'prec@3:{top3:.2f}'.format(
               epoch, i, len(train_loader), batch_time=batch_time.avg, data_time=data_time.avg,
                lr=optimizer.param_groups[0]['lr'], losses=losses.avg, top1=top1.avg, top3=top3.avg))
    logging.info(log_str)

    return losses, top1

def valid(valid_loader, net, criterion):
    losses = AverageMeter()
    top1   = AverageMeter()
    top3   = AverageMeter()

    net.eval()

    end = time.time()
    for i, (x, target) in enumerate(valid_loader, start=1):

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

    log_str = ('validing on {0:04} samples...'.format(len(valid_loader.dataset)))
    logging.info(log_str)
    f_l = [losses.avg, top1.avg, top3.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@3: {:.2f}'.format(*f_l))

    return losses, top1


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

    log_str = ('testing on {0:04} samples...'.format(len(test_loader.dataset)))
    logging.info(log_str)
    f_l = [losses.avg, top1.avg, top3.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@3: {:.2f}'.format(*f_l))

    return top1.avg, top3.avg, losses.avg



if __name__ == '__main__':
    main()

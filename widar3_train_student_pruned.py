"""
use N-1 Teachers to train a student.
Optional: baseline Teacher

TOTAL LOSS = lamda_1 * KD_LOSS_1 + ... + lamda_n * KD_loss_n + lamda_baseline * KD_baseline_loss

NOTE: In this version, when loading train data, we sample a batch in every domain seperately.
"""

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

#################### FOR DEBUG #############################
preread = True
# preread = False
#################### END DEBUG #############################

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--data_dir', type=str, default="/dataset/widar3_dfs")
parser.add_argument('--save_root', type=str, default='./result', help='models and logs are saved here')
parser.add_argument('--domain_name', type=str)
# parser.add_argument('--train_domain', type=int)
parser.add_argument('--train_domain', nargs='+', help='train domains', required=True)
parser.add_argument('--test_domain', type=int)
parser.add_argument('--baseline_path', type=str)
parser.add_argument('--t_dir', type=str, help="the directory where teacher models exist")
parser.add_argument('--pruned_model_path', type=str, help="path to the pruned student model", required=True)

# net choice
parser.add_argument('--t_name', type=str, help='model name of teacher', default='Convnet')    # Convnet / resnet18
parser.add_argument('--base_name', type=str, help='model name of baseline', default='Convnet')    # Convnet / resnet18
parser.add_argument('--s_name', type=str, help='model name of student', default='Convnet')    # Convnet / resnet18/ SConvnet


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
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--num_class', type=int, default=6, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)

# hyperparameter
parser.add_argument('--feat_dim', type=int, default=512)
parser.add_argument('--softplus_beta', type=float, default=1.0)
parser.add_argument('--kd_mode', default='st', type=str, help='mode of kd, which can be:'
                                                              'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                              'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--lambda_cls', type=float, default=1.0, help='trade-off parameter for classification loss')
# parser.add_argument('--lambda_kd', type=float, default=1.0, help='trade-off parameter for kd loss')
parser.add_argument('--lambda_kd', nargs='+', help='trade-off parameter list for teachers', required=True)
parser.add_argument('--lambda_kd_baseline', type=float, default=0, help='trade-off parameter for baseline kd loss')
parser.add_argument('--T', type=float, default=4.0, help='temperature for ST')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')


# # net and dataset choosen
# parser.add_argument('--data_name', type=str, required=True, help='name of dataset') # cifar10/cifar100
# parser.add_argument('--net_name', type=str, required=True, help='name of basenet')  # resnet20/resnet110


args, unparsed = parser.parse_known_args()
if isinstance(args.train_domain, list) :
    args.train_domain = list(map(int, args.train_domain))
if isinstance(args.lambda_kd, list) :
    args.lambda_kd = list(map(float, args.lambda_kd))
#normalize weight
s = args.lambda_kd_baseline + args.lambda_cls
s_t = 0
for d in args.train_domain:
	s_t += args.lambda_kd[d-1]
args.lambda_cls /= s + 4
args.lambda_kd_baseline /= s + 4
args.lambda_kd = [x/(s+ s_t - 4) for x in args.lambda_kd]

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
import torch_pruning as tp

create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'), mode='w')
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

TEACHER_MODEL_PATH = {x : f"{args.t_dir}/T{x}/model_best.pth.tar" for x in DOMAIN_LIST}

INT_DOMAIN_NAME_DICT = {
    1 : 'loc1',
    2 : 'loc2',
    3 : 'loc3',
    4 : 'loc4',
    5 : 'loc5',
}

DOMAIN_TEACHER_MODEL_PAIR = {}
baseline = None


# def reset_parameters(model):
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             module.reset_parameters()

# def reset_parameters(model):
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
#             module.reset_parameters()
def reset_parameters(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)



def main() :
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    if args.cuda :
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

        logging.info('------------ GPU Inffor -------------------')
        logging.info(torch.cuda.get_device_name())

    global DOMAIN_TEACHER_MODEL_PAIR, baseline

    # load teacher model
    for domain in args.train_domain :
        path = TEACHER_MODEL_PATH[domain]
        tnet = get_model(args.t_name, args)
        checkpoint = torch.load(path,map_location='cuda:0')
        load_pretrained_model(tnet, checkpoint['net'])
        if args.cuda :
            tnet = tnet.cuda()
        tnet.eval()
        for param in tnet.parameters() :
            param.requires_grad = False
        logging.info(f'Load {path}, prec@1: {checkpoint["prec@1"]:.2}, prec@3: {checkpoint["prec@3"]:.2}')

        DOMAIN_TEACHER_MODEL_PAIR[domain] = tnet

    # load baseline (used as teacher, too)
    baseline = get_model(args.base_name, args)
    checkpoint = torch.load(args.baseline_path,map_location='cuda:0')
    load_pretrained_model(baseline, checkpoint['net'])
    if args.cuda :
        baseline = baseline.cuda()
    baseline.eval()
    for param in baseline.parameters() :
        param.requires_grad = False
    logging.info(
        f'Load baseline from {args.baseline_path}, prec@1: {checkpoint["prec@1"]:.2}, prec@3: {checkpoint["prec@3"]:.2}')

    logging.info('----------- Network Initialization --------------')
    snet = get_model(args.s_name, args)

    # Load pruned model and adjust structure
    pruned_checkpoint = torch.load(args.pruned_model_path)
    load_pretrained_model(snet, pruned_checkpoint['net'])
    snet = snet.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snet.to(device)

    example_inputs = torch.randn(64, 6, 121, 121).to(device)

    DG = tp.DependencyGraph().build_dependency(snet, example_inputs=example_inputs)

    def prune_conv(DG, conv, channels_to_prune):
        pruning_group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=channels_to_prune)
        pruning_group.prune()

    def prune_linear(DG, linear, channels_to_prune):
        pruning_group = DG.get_pruning_group(linear, tp.prune_linear_out_channels, idxs=channels_to_prune)
        pruning_group.prune()

    modules = list(snet.named_modules())
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
    reset_parameters(snet)

    if args.cuda :
            snet = snet.cuda()
    logging.info('%s', snet)
    logging.info("param size = %fMB", count_parameters_in_MB(snet))
    logging.info('-----------------------------------------------')

    # save initial parameters
    logging.info('Saving initial parameters......')
    save_path = os.path.join(args.save_root, 'initial.pth.tar')
    torch.save({
        'epoch' : 0,
        'net' : snet.state_dict(),
        'prec@1' : 0.0,
        'prec@5' : 0.0,
    }, save_path)

    # define loss functions
    tnet = None
    if args.kd_mode == 'logits' :
        criterionKD = Logits()
    elif args.kd_mode == 'st' :
        criterionKD = SoftTarget(args.T)
    elif args.kd_mode == 'at' :
        criterionKD = AT(args.p)
    elif args.kd_mode == 'fitnet' :
        criterionKD = Hint()
    elif args.kd_mode == 'nst' :
        criterionKD = NST()
    elif args.kd_mode == 'pkt' :
        criterionKD = PKTCosSim()
    elif args.kd_mode == 'fsp' :
        criterionKD = FSP()
    elif args.kd_mode == 'rkd' :
        criterionKD = RKD(args.w_dist, args.w_angle)
    elif args.kd_mode == 'ab' :
        criterionKD = AB(args.m)
    elif args.kd_mode == 'sp' :
        criterionKD = SP()
    elif args.kd_mode == 'sobolev' :
        criterionKD = Sobolev()
    elif args.kd_mode == 'cc' :
        criterionKD = CC(args.gamma, args.P_order)
    elif args.kd_mode == 'lwm' :
        criterionKD = LwM()
    elif args.kd_mode == 'irg' :
        criterionKD = IRG(args.w_irg_vert, args.w_irg_edge, args.w_irg_tran)
    elif args.kd_mode == 'vid' :
        s_channels = snet.module.get_channel_num()[1 :4]
        t_channels = tnet.module.get_channel_num()[1 :4]
        criterionKD = []
        for s_c, t_c in zip(s_channels, t_channels) :
            criterionKD.append(VID(s_c, int(args.sf * t_c), t_c, args.init_var))
        criterionKD = [c.cuda() for c in criterionKD] if args.cuda else criterionKD
        criterionKD = [None] + criterionKD  # None is a placeholder
    elif args.kd_mode == 'ofd' :
        s_channels = snet.module.get_channel_num()[1 :4]
        t_channels = tnet.module.get_channel_num()[1 :4]
        criterionKD = []
        for s_c, t_c in zip(s_channels, t_channels) :
            criterionKD.append(OFD(s_c, t_c).cuda() if args.cuda else OFD(s_c, t_c))
        criterionKD = [None] + criterionKD  # None is a placeholder
    elif args.kd_mode == 'afd' :
        s_channels = snet.module.get_channel_num()[1 :4]
        t_channels = tnet.module.get_channel_num()[1 :4]
        criterionKD = []
        for t_c in t_channels :
            criterionKD.append(AFD(t_c, args.att_f).cuda() if args.cuda else AFD(t_c, args.att_f))
        criterionKD = [None] + criterionKD  # None is a placeholder
    else :
        raise Exception('Invalid kd mode...')

    if args.cuda :
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else :
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer
    optimizer = get_optimizer(snet, args)
    lr_schduler = get_scheduler(optimizer, args)

    # define data loader
    train_loader_list = []
    for d in args.train_domain :
        train_dataset = dataset_widar3.DomainSet(args.data_dir, args.domain_name, d, preread=preread,
                                                 return_domain_label=True)
        train_loader_list.append(
            torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        )

    test_dataset = dataset_widar3.DomainSet(args.data_dir, args.domain_name, args.test_domain, preread=preread,
                                            isTest=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # warp nets and criterions for train and test
    nets = {'snet' : snet, 'tnet' : DOMAIN_TEACHER_MODEL_PAIR, 'baseline' : baseline}
    criterions = {'criterionCls' : criterionCls, 'criterionKD' : criterionKD}

    best_top1 = 0
    best_top3 = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1) :

        # train one epoch
        epoch_start_time = time.time()
        train_prec = train(train_loader_list, nets, optimizer, criterions, epoch, lr_schduler)

        # evaluate on testing set
        logging.info('Testing the models......')
        test_top1, test_top3, test_loss = test(test_loader, snet, criterions['criterionCls'])

        epoch_duration = time.time() - epoch_start_time
        logging.info('Epoch time: {}s'.format(int(epoch_duration)))

        writer.add_scalar('test/prec_1', test_top1, epoch)
        writer.add_scalar('test/loss', test_loss, epoch)

        # save model
        is_best = False
        if test_top1 > best_top1 :
            best_top1 = test_top1
            best_top3 = test_top3
            is_best = True
            best_epoch = epoch
            logging.info('Saving models......')
            save_best__checkpoint(epoch, {
                'epoch' : epoch,
                'net' : snet.state_dict(),
                'prec@1' : test_top1,
                'prec@3' : test_top3,
            }, is_best, args.save_root)
        logging.info(f'best epoch is {best_epoch}, prec@1: {round(best_top1, 2)}, prec@3: {round(best_top3, 2)}')
        logging.info('')

    # save current model
    save_current_checkpoint({
            'epoch' : epoch,
            'net' : snet.state_dict(),
            'prec@1' : test_top1,
            'prec@3' : test_top3,
        }, args.save_root)

def train(train_loader_list, nets, optimizer, criterions, epoch, scheduler) :
    batch_time = AverageMeter()
    data_time = AverageMeter()
    cls_losses = {d : AverageMeter() for d in args.train_domain}
    kd_losses = {d : AverageMeter() for d in args.train_domain}
    kd_baseline_losses = {d : AverageMeter() for d in args.train_domain}
    top1 = {d : AverageMeter() for d in args.train_domain}
    top3 = {d : AverageMeter() for d in args.train_domain}

    iters = len(train_loader_list[0])

    snet = nets['snet']
    snet.train()
    DOMAIN_TEACHER_MODEL_PAIR = nets['tnet']
    baseline = nets['baseline']

    criterionCls = criterions['criterionCls']
    criterionKD = criterions['criterionKD']

    end = time.time()
    for i, data in enumerate(zip(*train_loader_list), start=1) :
        data_time.update(time.time() - end)

        for x, target, domain in data :
            x = x.float()
            target = target.long()
            domain = int(domain[0])

            if args.cuda :
                x = x.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            out_s = snet(x)
            with torch.no_grad():
                out_b = baseline(x)
            cls_loss = criterionCls(out_s, target) * args.lambda_cls

            if args.lambda_kd_baseline != 0 :
                kd_baseline_loss = criterionKD(out_s, out_b.detach()) * args.lambda_kd_baseline
            else :
                kd_baseline_loss = torch.Tensor([0.0]).cuda()
            
            loss = cls_loss  + kd_baseline_loss

            domain += 1
            if domain in args.train_domain:
                tnet = DOMAIN_TEACHER_MODEL_PAIR[domain]
                with torch.no_grad():
                    out_t = tnet(x)
                if args.lambda_kd[domain - 1] != 0 :
                    if args.kd_mode in ['logits', 'st'] :
                        kd_loss = criterionKD(out_s, out_t.detach()) * args.lambda_kd[domain - 1]
                    else :
                        raise Exception('Invalid kd mode...')
                else:
                    kd_loss = torch.Tensor([0.0]).cuda()
                loss += kd_loss

            prec1, prec3 = accuracy(out_s, target, topk=(1, 3))
            cls_losses[domain].update(cls_loss.item(), x.size(0))
            kd_losses[domain].update(kd_loss.item(), x.size(0))
            kd_baseline_losses[domain].update(kd_baseline_loss.item(), x.size(0))
            top1[domain].update(prec1.item(), x.size(0))
            top3[domain].update(prec3.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        scheduler.step(epoch - 1 + i/iters)

        if i % args.print_freq == 0 or (epoch == 1 and i < args.print_freq) :

            log_str = ('Epoch[{0}]:[{1:03}/{2:03}] '
                       'BatchTime:{batch_time.val:.4f} '
                       'DataTime:{data_time.val:.4f}  '
                        'lr: {lr:.6f}'
                .format(
                epoch, i, len(train_loader_list[0]), batch_time=batch_time, data_time=data_time, lr=optimizer.param_groups[0]['lr']))
            logging.info(log_str)

            for domain in args.train_domain :
                log_str = ('on domain {domain} '
                           'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                           'KD:{kd_losses.val:.4f}({kd_losses.avg:.4f})  '
                           'KD_baseline:{kd_baseline_losses.val:.4f}({kd_baseline_losses.avg:.4f})  '
                           'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                           'prec@3:{top3.val:.2f}({top3.avg:.2f})'
                    .format(
                    domain=domain, cls_losses=cls_losses[domain], kd_baseline_losses=kd_baseline_losses[domain],
                    kd_losses=kd_losses[domain], top1=top1[domain], top3=top3[domain]
                )
                )
                logging.info(log_str)


def test(test_loader, net, criterion) :
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net.eval()

    end = time.time()
    for i, (x, target) in enumerate(test_loader, start=1) :

        x = x.float()
        target = target.long()

        if args.cuda :
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad() :
            out = net(x)
            loss = criterion(out, target)

        prec1, prec3 = accuracy(out, target, topk=(1, 3))
        losses.update(loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        top3.update(prec3.item(), x.size(0))

    f_l = [losses.avg, top1.avg, top3.avg]
    logging.info('Loss: {:.4f}, Prec@1: {:.2f}, Prec@3: {:.2f}'.format(*f_l))

    return top1.avg, top3.avg, losses.avg


if __name__ == '__main__' :
    main()


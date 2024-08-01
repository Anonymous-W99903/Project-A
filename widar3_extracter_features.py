from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import argparse
import numpy as np
import ot 
from kd_losses import *
import matplotlib.pyplot as plt
from lr_scheduler import LR_Scheduler
from network import FeatureExtracter2d, ActRecognizer, ConvNet
import dataset_widar3

parser = argparse.ArgumentParser(description='train base net')

# various path
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_root', type=str, default='./result', help='models and logs are saved here')
parser.add_argument('--domain_name', type=str)
# parser.add_argument('--train_domain', type=int)
# parser.add_argument('--test_domain', type=int)
parser.add_argument('--model_path', type=str, help='path to pretrained model')
parser.add_argument('--domain_cmp', type=str, help='domains to compare, e.g., "1,2,3,4"')



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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

board_path = os.path.join(args.save_root, 'board')
create_exp_dir(board_path)
writer = SummaryWriter(board_path)


def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = model.extract_features(inputs)
            features.append(outputs.cpu())
    return torch.cat(features, dim=0)

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

    # Load pretrained model
    if args.model_path:
        logging.info(f'Loading pretrained model from {args.model_path}')
        checkpoint = torch.load(args.model_path)
        net.load_state_dict(checkpoint['net'])

    domains = [int(d) for d in args.domain_cmp.split(',')]
    feature_sets = []
    for domain in domains:
        loader,_ = dataset_widar3.get_train_and_valid_loader(domain, args)
        features = extract_features(net, loader, torch.device('cuda' if args.cuda else 'cpu'))
        print(f"Shape of features for domain {domain}: {features.shape}")
        feature_sets.append(features)

        num_domains = len(feature_sets)
    emd_matrix = np.zeros((num_domains, num_domains))

    for i in range(num_domains):
        for j in range(i, num_domains):
            if i == j:
                emd_matrix[i, j] = 0
            else:
                feat_i = feature_sets[i].numpy()
                feat_j = feature_sets[j].numpy()
                dist_matrix = ot.dist(feat_i, feat_j)

                a = np.ones((feat_i.shape[0],)) / feat_i.shape[0]
                b = np.ones((feat_j.shape[0],)) / feat_j.shape[0]

                emd = ot.emd2(a, b, dist_matrix)
                emd_matrix[i, j] = emd
                emd_matrix[j, i] = emd

    print("EMD matrix between domains:")
    print(emd_matrix)

if __name__ == '__main__':
    main()
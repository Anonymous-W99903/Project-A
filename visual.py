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
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

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
parser.add_argument('--data_dir', type=str, default="/home/yueyang/dataset/widar3_dfs")
parser.add_argument('--save_root', type=str, default='./result', help='models and logs are saved here')
parser.add_argument('--domain_name', type=str)
# parser.add_argument('--train_domain', type=int)
parser.add_argument('--train_domain', nargs='+', help='train domains', required=True)
parser.add_argument('--test_domain', type=int)
parser.add_argument('--baseline_path', type=str)
parser.add_argument('--t_dir', type=str, help="the directory where teacher models exist")


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
for d in args.train_domain:
	s += args.lambda_kd[d-1]
args.lambda_cls /= s
args.lambda_kd_baseline /= s
args.lambda_kd = [x/s for x in args.lambda_kd]

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

# fun1 = lambda x : f'results/widar3/loc/baseline{x}/model_best.pth.tar'
# BASELINE_MODEL_PATH = {x : fun1(x) for x in DOMAIN_LIST}

INT_DOMAIN_NAME_DICT = {
    1 : 'loc1',
    2 : 'loc2',
    3 : 'loc3',
    4 : 'loc4',
    5 : 'loc5',
}

DOMAIN_TEACHER_MODEL_PAIR = {}
baseline = None

def extract_features(model, dataloader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x, target in dataloader:
            x = x.float().cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            out = model(x)
            features.append(out.cpu().numpy())
            labels.append(target.cpu().numpy())
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

def visualize_features(features, labels, save_path, domain):
    lda = LDA(n_components=2)
    reduced_features = lda.fit_transform(features, labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f'Feature Visualization using LDA - Domain {domain}')
    plt.savefig(save_path)
    plt.close()

def visualize_combined_features(features_dict, labels, save_path):
    lda = LDA(n_components=2)
    plt.figure(figsize=(15, 10))
    
    for domain, features in features_dict.items():
        reduced_features = lda.fit_transform(features, labels)
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.7, label=f'Domain {domain}')
    
    plt.legend()
    plt.title('Combined Feature Visualization using LDA')
    plt.savefig(save_path)
    plt.close()

def main():
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

        logging.info('------------ GPU Inffor -------------------')
        logging.info(torch.cuda.get_device_name())

    global DOMAIN_TEACHER_MODEL_PAIR, baseline

    # load teacher model
    for domain in DOMAIN_LIST:
        path = TEACHER_MODEL_PATH[domain]
        tnet = get_model(args.t_name, args)
        checkpoint = torch.load(path, map_location='cuda:0')
        load_pretrained_model(tnet, checkpoint['net'])
        if args.cuda:
            tnet = tnet.cuda()
        tnet.eval()
        for param in tnet.parameters():
            param.requires_grad = False
        logging.info(f'Load {path}, prec@1: {checkpoint["prec@1"]:.2}, prec@3: {checkpoint["prec@3"]:.2}')

        DOMAIN_TEACHER_MODEL_PAIR[domain] = tnet

    baseline = get_model(args.base_name, args)
    checkpoint = torch.load(args.baseline_path, map_location='cuda:0')
    load_pretrained_model(baseline, checkpoint['net'])
    if args.cuda:
        baseline = baseline.cuda()
    baseline.eval()
    for param in baseline.parameters():
        param.requires_grad = False
    logging.info(f'Load baseline from {args.baseline_path}, prec@1: {checkpoint["prec@1"]:.2}, prec@3: {checkpoint["prec@3"]:.2}')


     # Load student model
    student_path = "/home/wzy-21/lab/AI_wireless/widar3/val_location/v2_2_T2_S_r1/model_best.pth.tar"
    student = get_model(args.s_name, args)
    checkpoint = torch.load(student_path, map_location='cuda:0')
    load_pretrained_model(student, checkpoint['net'])
    if args.cuda:
        student = student.cuda()
    student.eval()
    for param in student.parameters():
        param.requires_grad = False
    logging.info(f'Load student from {student_path}, prec@1: {checkpoint["prec@1"]:.2}, prec@3: {checkpoint["prec@3"]:.2}')


    # Define test data loader
    test_dataset = dataset_widar3.DomainSet(args.data_dir, args.domain_name, args.test_domain, preread=preread, isTest=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    features_dict = {}
    for domain, tnet in DOMAIN_TEACHER_MODEL_PAIR.items():
        features, labels = extract_features(tnet, test_loader)
        features_dict[domain] = features
        
        # Visualize and save each domain separately
        save_path = os.path.join(args.save_root, f'lda_visualization_domain_{domain}.png')
        visualize_features(features, labels, save_path, domain)

    # Extract features for baseline teacher
    baseline_features, labels = extract_features(baseline, test_loader)
    features_dict['Baseline'] = baseline_features
    
    # Visualize baseline teacher separately
    baseline_save_path = os.path.join(args.save_root, 'lda_visualization_baseline.png')
    visualize_features(baseline_features, labels, baseline_save_path, 'Feature Visualization using LDA - Baseline Teacher')

    # Extract features for student model
    student_features, labels = extract_features(student, test_loader)
    features_dict['Student'] = student_features
    
    # Visualize student model separately
    student_save_path = os.path.join(args.save_root, 'lda_visualization_student.png')
    visualize_features(student_features, labels, student_save_path, 'Feature Visualization using LDA - Student Model')


    # # Visualize combined features from all domains and baseline
    # combined_save_path = os.path.join(args.save_root, 'lda_visualization_combined.png')
    # visualize_combined_features(features_dict, labels, combined_save_path)

if __name__ == '__main__':
    main()
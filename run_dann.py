"""
An simple implementation for DANN.

TOTAL LOSS = cls_loss + alpha * entropy_loss - beta * domain_loss

Note:
1.cls_loss is only for labeled data.
2.ent_loss is only for unlabeled data.
3.minus operator is replaced by GradReverse.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import logging
import shutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, accuracy, transform_time
from utils import load_pretrained_model, save_best__checkpoint
from utils import create_exp_dir, count_parameters_in_MB, save_current_checkpoint
from utils import adjust_lr, get_scheduler, get_model, get_optimizer
from kd_losses import *

import dataset_widar3

#################### FOR DEBUG #############################
preread = True
# preread = False
#################### END DEBUG #############################

parser = argparse.ArgumentParser(description="train base net")

# various path
parser.add_argument("--data_dir", type=str, default="/home/data/widar3_dfs")
parser.add_argument(
    "--save_root", type=str, default="./result", help="models and logs are saved here"
)
parser.add_argument("--domain_name", type=str)
parser.add_argument("--train_domain", nargs="+", help="train domains", required=True)
parser.add_argument("--test_domain", type=int)

# net choice
parser.add_argument("--name", type=str, help="model name", default="DANN")


# training hyper parameters
parser.add_argument(
    "--print_freq",
    type=int,
    default=10,
    help="frequency of showing training results on console",
)
parser.add_argument(
    "--epochs", type=int, default=50, help="number of total epochs to run"
)
parser.add_argument("--batch_size", type=int, default=16, help="The size of batch")
parser.add_argument("--num_workers", type=int, default=0, help="number of data loaders")
parser.add_argument("--opt", type=str, default="Adam", help="DL optimizer")
parser.add_argument(
    "--adjust_lr", type=str, default="step", help="the method to adjust learning rate"
)
parser.add_argument(
    "--warmup_epoch", type=int, default=0, help="the number of warmup epochs"
)
parser.add_argument(
    "--step_gamma", type=float, default=0, help="decay weight for step policy"
)
parser.add_argument(
    "--step_size", type=int, default=0, help="decay step for step policy"
)
parser.add_argument(
    "--T_mult", type=int, default=1, help="a factor for CosineAnnealingWarmRestarts"
)
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay")
parser.add_argument("--cuda", type=int, default=1)
parser.add_argument("--gpu_id", type=int, default=0)

# hyperparameter
parser.add_argument("--num_class", type=int, default=6, help="number of classes")
parser.add_argument("--feat_dim", type=int, default=512)
parser.add_argument("--softplus_beta", type=float, default=1.0)
parser.add_argument(
    "--kd_mode",
    default="st",
    type=str,
    help="mode of kd, which can be:"
    "logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/"
    "sp/sobolev/cc/lwm/irg/vid/ofd/afd",
)
parser.add_argument("--alpha", type=float, default=1, help="weight of entropy loss")
parser.add_argument(
    "--beta", type=float, default=1, help="weight of domain classification loss"
)

# others
import random
parser.add_argument("--seed", type=int, default=random.randint(0,100000), help="random seed")

args, unparsed = parser.parse_known_args()
args.gamma = 10.0  # for generating a constant for reversing grad
if isinstance(args.train_domain, list):
    args.train_domain = list(map(int, args.train_domain))
args.num_domain = (
    len(args.train_domain) + 1
    if isinstance(args.test_domain, int)
    else len(args.train_domain) + len(args.test_domain)
)

# normalize weight
s = 1 + args.alpha + args.beta
args.alpha /= s
args.beta /= s

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if os.path.exists(args.save_root):
    shutil.rmtree(args.save_root)
create_exp_dir(args.save_root)

log_format = "%(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

board_path = os.path.join(args.save_root, "board")
create_exp_dir(board_path)
writer = SummaryWriter(board_path)


def main():
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

        logging.info("------------ GPU Inffor -------------------")
        logging.info(torch.cuda.get_device_name())

    logging.info("----------- Network Initialization --------------")
    net = get_model(args.name, args)
    if args.cuda:
        net = net.cuda()
    logging.info("%s", net)
    logging.info("param size = %fMB", count_parameters_in_MB(net))
    logging.info("-----------------------------------------------")

    # save initial parameters
    logging.info("Saving initial parameters......")
    save_path = os.path.join(args.save_root, "initial.pth.tar")
    torch.save(
        {
            "epoch": 0,
            "net": net.state_dict(),
            "prec@1": 0.0,
            "prec@5": 0.0,
        },
        save_path,
    )

    # set up loss function
    criterionCls = F.cross_entropy

    def criterionEnt(x):
        pred = torch.softmax(x, dim=-1)
        return (-pred * torch.log(pred)).sum(dim=-1).mean(dim=-1)

    criterionDom = F.cross_entropy

    # initialize optimizer
    optimizer = get_optimizer(net, args)
    lr_schduler = get_scheduler(optimizer, args)

    # define data loader: train loader, validation loader, test loader
    s_train_loader, s_valid_loader = dataset_widar3.get_train_and_valid_loader(
        args.train_domain,
        args,
        domain_name=args.domain_name,
        preread=preread,
        return_domain_label=True,
    )
    t_train_loader, t_valid_loader = dataset_widar3.get_train_and_valid_loader(
        args.test_domain,
        args,
        domain_name=args.domain_name,
        preread=preread,
        return_domain_label=True,
    )

    test_dataset = dataset_widar3.DomainSet(
        args.data_dir, args.domain_name, args.test_domain, preread=preread, isTest=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # warp nets and criterions for train and test
    criterions = {
        "criterionCls": criterionCls,
        "criterionEnt": criterionEnt,
        "criterionDom": criterionDom,
    }

    best_top1 = 0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):

        # train and validate one epoch
        epoch_start_time = time.time()
        train_res = train(
            s_train_loader,
            t_train_loader,
            net,
            optimizer,
            criterions,
            epoch,
            lr_schduler,
        )
        with torch.no_grad():
            valid_res = valid(s_valid_loader, t_valid_loader, net, criterions, epoch)
            test_top1, test_top3, test_loss = test(
                test_loader, net, criterions["criterionCls"]
            )

        # record results of the epoch(avg of iterations)
        writer.add_scalars(
            "total_loss", {"train": train_res[0].avg, "valid": valid_res[0].avg}, epoch
        )
        writer.add_scalars(
            f"source/prec",
            {"train": train_res[1].avg, "valid": valid_res[1].avg},
            epoch,
        )
        writer.add_scalars(
            f"target/prec",
            {"train": train_res[2].avg, "valid": valid_res[2].avg},
            epoch,
        )
        writer.add_scalars(
            f"source/cls_loss",
            {"train": train_res[3].avg, "valid": valid_res[3].avg},
            epoch,
        )
        writer.add_scalars(
            f"target/ent_loss",
            {"train": train_res[4].avg, "valid": valid_res[4].avg},
            epoch,
        )
        writer.add_scalars(
            f"dom_loss",
            {"train": train_res[5].avg, "valid": valid_res[5].avg},
            epoch,
        )
        writer.add_scalar("test/prec_1", test_top1, epoch)
        writer.add_scalar("test/prec_3", test_top3, epoch)
        writer.add_scalar("test/loss", test_loss, epoch)

        epoch_duration = time.time() - epoch_start_time
        logging.info("Epoch time: {}s".format(int(epoch_duration)))

    # save current model
    if epoch % 10 == 0:
        save_current_checkpoint(
            {
                "epoch": epoch,
                "net": net.state_dict(),
                "prec@1": test_top1,
                "prec@3": test_top3,
            },
            args.save_root,
        )


def train(s_train_loader, t_train_loader, net, optimizer, criterions, epoch, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    ent_losses = AverageMeter()
    dom_losses = AverageMeter()
    s_top1 = AverageMeter()
    t_top1 = AverageMeter()

    num_samples = min(len(s_train_loader.dataset), len(t_train_loader.dataset))
    iters = min(len(s_train_loader), len(t_train_loader))

    criterionCls = criterions["criterionCls"]
    criterionEnt = criterions["criterionEnt"]
    criterionDom = criterions["criterionDom"]

    end = time.time()

    net.train()
    for i, (s_data, t_data) in enumerate(zip(s_train_loader, t_train_loader), start=1):
        data_time.update(time.time() - end)

        # generate constant for GradReverse
        p = (epoch - 1 + i / iters) / args.epochs
        assert p >= 0 and p <= 1.0
        constant = 2.0 / (1.0 + np.exp(-args.gamma * p)) - 1

        source_input = s_data[0].float()
        source_labels = s_data[1].long()
        source_domain = s_data[2].long()
        target_input = t_data[0].float()
        target_domain = t_data[2].long()
        if args.cuda:
            source_input = source_input.cuda(non_blocking=True)
            source_labels = source_labels.cuda(non_blocking=True)
            source_domain = source_domain.cuda(non_blocking=True)
            target_input = target_input.cuda(non_blocking=True)
            target_domain = target_domain.cuda(non_blocking=True)

        src_cls, src_dom = net(source_input, constant)
        tgt_cls, tgt_dom = net(target_input, constant)

        cls_loss = criterionCls(src_cls, source_labels)
        ent_loss = criterionEnt(tgt_cls) * args.alpha
        dom_loss = (
            criterionDom(src_dom, source_domain)
            + criterionDom(tgt_dom, target_domain) * args.beta
        )
        loss = cls_loss + ent_loss + dom_loss

        s_prec1, _ = accuracy(src_cls, source_labels, topk=(1, 3))
        t_prec1, _ = accuracy(
            tgt_cls, t_data[1].long().cuda(non_blocking=True), topk=(1, 3)
        )

        # record results of the iteration
        cls_losses.update(cls_loss.item(), source_input.size(0))
        ent_losses.update(ent_loss.item(), source_input.size(0))
        dom_losses.update(dom_loss.item(), source_input.size(0))
        losses.update(loss.item(), source_input.size(0))
        s_top1.update(s_prec1.item(), source_input.size(0))
        t_top1.update(t_prec1.item(), source_input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update lr
        scheduler.step(epoch - 1 + i / iters)

        batch_time.update(time.time() - end)
        end = time.time()

        log_str = (
            "Epoch[{0}]:[samples:{1:03}] "
            "BatchTime:{batch_time.val:.4f} "
            "DataTime:{data_time.val:.4f}  "
            "lr: {lr:.6f} loss: {loss:.6f}".format(
                epoch,
                num_samples,
                batch_time=batch_time,
                data_time=data_time,
                lr=optimizer.param_groups[0]["lr"],
                loss=losses.avg,
            )
        )
    logging.info(log_str)

    return [losses, s_top1, t_top1, cls_losses, ent_losses, dom_losses]


def valid(s_valid_loader, t_valid_loader, net, criterions, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    cls_losses = AverageMeter()
    ent_losses = AverageMeter()
    dom_losses = AverageMeter()
    s_top1 = AverageMeter()
    t_top1 = AverageMeter()

    num = min(len(s_valid_loader.dataset), len(t_valid_loader.dataset))
    iters = min(len(s_valid_loader), len(t_valid_loader))

    criterionCls = criterions["criterionCls"]
    criterionEnt = criterions["criterionEnt"]
    criterionDom = criterions["criterionDom"]

    end = time.time()

    net.eval()
    for i, (s_data, t_data) in enumerate(zip(s_valid_loader, t_valid_loader), start=1):
        data_time.update(time.time() - end)

        # generate constant for GradReverse
        p = (epoch - 1 + i / iters) / args.epochs
        assert p >= 0 and p <= 1.0
        constant = 2.0 / (1.0 + np.exp(-args.gamma * p)) - 1

        source_input = s_data[0].float()
        source_labels = s_data[1].long()
        source_domain = s_data[2].long()
        target_input = t_data[0].float()
        target_domain = t_data[2].long()
        if args.cuda:
            source_input = source_input.cuda(non_blocking=True)
            source_labels = source_labels.cuda(non_blocking=True)
            source_domain = source_domain.cuda(non_blocking=True)
            target_input = target_input.cuda(non_blocking=True)
            target_domain = target_domain.cuda(non_blocking=True)

        src_cls, src_dom = net(source_input, constant)
        tgt_cls, tgt_dom = net(target_input, constant)

        cls_loss = criterionCls(src_cls, source_labels)
        ent_loss = criterionEnt(tgt_cls) * args.alpha
        dom_loss = (
            criterionDom(src_dom, source_domain)
            + criterionDom(tgt_dom, target_domain) * args.beta
        )
        loss = cls_loss + ent_loss + dom_loss

        s_prec1, _ = accuracy(src_cls, source_labels, topk=(1, 3))
        t_prec1, _ = accuracy(
            tgt_cls, t_data[1].long().cuda(non_blocking=True), topk=(1, 3)
        )

        # record results of the iteration
        cls_losses.update(cls_loss.item(), source_input.size(0))
        ent_losses.update(ent_loss.item(), source_input.size(0))
        dom_losses.update(dom_loss.item(), source_input.size(0))
        losses.update(loss.item(), source_input.size(0))
        s_top1.update(s_prec1.item(), source_input.size(0))
        t_top1.update(t_prec1.item(), source_input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    log_str = (
        "validating:[samples:{0:03}] "
        "BatchTime:{batch_time.val:.4f} "
        "DataTime:{data_time.val:.4f}  "
        "loss: {loss:.6f}".format(
            num, batch_time=batch_time, data_time=data_time, loss=losses.avg
        )
    )
    logging.info(log_str)

    return [losses, s_top1, t_top1, cls_losses, ent_losses, dom_losses]


def test(test_loader, net, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    net.eval()

    for i, (x, target) in enumerate(test_loader, start=1):

        x = x.float()
        target = target.long()

        if args.cuda:
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        with torch.no_grad():
            cls_pred, _ = net(x, 1.0)
            loss = criterion(cls_pred, target)

        prec1, prec3 = accuracy(cls_pred, target, topk=(1, 3))
        losses.update(loss.item(), x.size(0))
        top1.update(prec1.item(), x.size(0))
        top3.update(prec3.item(), x.size(0))

    f_l = [losses.avg, top1.avg, top3.avg]
    logging.info("Loss: {:.4f}, Prec@1: {:.2f}, Prec@3: {:.2f}".format(*f_l))

    return top1.avg, top3.avg, losses.avg


if __name__ == "__main__":
    main()

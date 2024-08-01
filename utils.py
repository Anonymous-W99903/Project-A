from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import shutil
import numpy as np
import torch
from torch.optim import lr_scheduler
import math
from network import ConvNet, resnet18, DANN, Simplified_ConvNet, PrunedConvNet
from torchvision import models
from Nadam import Nadam

class AverageMeter(object) :
	def __init__(self) :
		self.reset()

	def reset(self) :
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1) :
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def count_parameters_in_MB(model) :
	return sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def create_exp_dir(path) :
	if not os.path.exists(path) :
		os.makedirs(path)
	print('Experiment dir : {}'.format(path))


def load_pretrained_model(model, pretrained_dict) :
	model_dict = model.state_dict()
	# 1. filter out unnecessary keys
	pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(pretrained_dict)
	# 3. load the new state dict
	model.load_state_dict(model_dict)


def transform_time(s) :
	m, s = divmod(int(s), 60)
	h, m = divmod(m, 60)
	return h, m, s


def save_current_checkpoint(state, save_root):
	save_path = os.path.join(save_root, f'checkpoint_current.pth.tar')
	torch.save(state, save_path)

def save_best__checkpoint(epoch, state, is_best, save_root) :
	# for Linux only??
	if is_best :
		best_save_path = os.path.join(save_root, f'model_best.pth.tar')
		torch.save(state, best_save_path)
		# shutil.copyfile(save_path, best_save_path)


def accuracy(output, target, topk=(1,)) :
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk :
		correct_k = correct[:k].contiguous().view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def get_optimizer(net, args):
	optimizer = None
	if args.opt == 'SGD':
		optimizer = torch.optim.SGD(net.parameters(),
		                            lr=args.lr,
		                            weight_decay=args.weight_decay)
	elif args.opt == 'momentum':
		optimizer = torch.optim.SGD(net.parameters(),
		                            lr=args.lr,
		                            momentum=args.momentum,
		                            weight_decay=args.weight_decay)

	elif args.opt == 'NAG':
		optimizer = torch.optim.SGD(net.parameters(),
		                            lr=args.lr,
		                            momentum=args.momentum,
		                            weight_decay=args.weight_decay,
		                            nesterov=True)
	elif args.opt == 'RMSProp':
		optimizer = torch.optim.RMSprop(net.parameters(),
		                                lr=args.lr,
		                                # alpha=0.9,
		                                weight_decay=args.weight_decay)

	elif args.opt == 'Adam':
		optimizer = torch.optim.Adam(net.parameters(),
		                            lr=args.lr,
		                            weight_decay=args.weight_decay)
	elif args.opt == 'Nadam':
		optimizer = Nadam(net.parameters(),
		                            lr=args.lr,
		                            weight_decay=args.weight_decay)

	else:
		raise NotImplementedError(f'{args.opt} is not implemented.')
	return optimizer


def adjust_lr(optimizer, epoch, args, logging) :
	if args.adjust_lr == 'step' :
		EPOCHS = args.epochs
		scale = 0.1
		lr_list = [args.lr] * (EPOCHS // 2)
		lr_list += [args.lr * scale] * (EPOCHS // 4)
		lr_list += [args.lr * scale * scale] * (EPOCHS - EPOCHS // 2 - EPOCHS // 4)

		lr = lr_list[epoch - 1]
		for param_group in optimizer.param_groups :
			param_group['lr'] = lr
	elif args.adjust_lr == '' :
		pass


def get_scheduler(optimizer, opt) :

	# def linear_rule(epoch) :
	# 	lr_l = 1.0 - epoch / opt.epochs
	# 	return lr_l
	def warmup_rule(epoch) :
		if epoch < opt.warmup_epoch :
			lr_l = epoch / opt.warmup_epoch
		else:
			T = epoch - opt.warmup_epoch
			total_epoch = opt.epochs - opt.warmup_epoch

			if opt.adjust_lr == 'cosine':
				lr_l = 0.5 * (1 + math.cos(T / total_epoch * math.pi))

			elif opt.adjust_lr == 'step':
				gamma = opt.step_gamma
				step_size = opt.step_size
				lr_l = gamma ** (T//step_size)

			elif opt.adjust_lr == 'linear':
				lr_l = 1.0 - T / total_epoch
			elif opt.adjust_lr == 'const':
				lr_l = 1.0
		return lr_l

	if opt.adjust_lr == 'linear' :
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_rule)
	elif opt.adjust_lr == 'step' :
		# scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_rule)
	# elif opt.adjust_lr == 'multistep' :
	# 	epochs = opt.epochs
	# 	milestones = [epochs // 2, epochs * 3 // 4]
	# 	scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
	elif opt.adjust_lr == 'plateau':
		scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
	elif opt.adjust_lr == 'cosine':
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_rule)
	elif opt.adjust_lr == 'cosine2':
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.step_size,eta_min=1e-5)
	elif opt.adjust_lr == 'cosine3':
		scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.step_size, T_mult=opt.T_mult, eta_min=1e-5)
	elif opt.adjust_lr == 'const':
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_rule)
	else:
		raise NotImplementedError('learning rate policy [%s] is not implemented', opt.adjust_lr)
	return scheduler

def convert_confsuion_matrix(cm):
	"""
	convert confusion matrix into percentage form
	"""
	cm = np.array(cm)
	new_cm = cm / cm.sum(axis=1) * 100
	return np.around(new_cm, 1)


def get_model(name, args):
	if name == 'Convnet':
		return ConvNet(6, vars(args))
	if name == 'resnet18':
		return resnet18(6, vars(args))
	if name == 'DANN':
		return DANN(6, vars(args))
	if name == 'SConvnet':
		return Simplified_ConvNet(6, vars(args))
	if name == 'PrunedConvNet':
		return PrunedConvNet(6, vars(args))

	raise NotImplementedError(f'model {name} is not implemented')
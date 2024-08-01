import argparse
import logging
import os
import shutil
import sys
import torch.nn.functional as F
import torch
import numpy as np
import dataset_widar3
import utils

########################## define settings ############################
preread = False

parser = argparse.ArgumentParser(description='train base net')

parser.add_argument('--data_dir', type=str, default="/home/yueyang/dataset/widar3_dfs")
parser.add_argument('--save_root', type=str,  help='models and logs are saved here')
parser.add_argument('--vote_mode', type=str,  help='max/avg')
parser.add_argument('--domain_name', type=str)
parser.add_argument('--test_domain', type=int)

# net choice
parser.add_argument('--name', type=str, help='model name', default='Convnet')    # Convnet / resnet18

# training hyper parameters
parser.add_argument('--batch_size', type=int, default=16, help='The size of batch')
parser.add_argument('--num_workers', type=int, default=0, help='number of data loaders')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--feat_dim', type=int, default=512)
parser.add_argument('--num_class', type=int, default=6, help='number of classes')
parser.add_argument('--softplus_beta', type=float, default=1.0)
parser.add_argument('--kd_mode', default='st', type=str, help='mode of kd, which can be:'
                                                              'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                              'sp/sobolev/cc/lwm/irg/vid/ofd/afd')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args, unparsed = parser.parse_known_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if os.path.exists(args.save_root):
	shutil.rmtree(args.save_root)
utils.create_exp_dir(args.save_root)

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.save_root, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

get_path = lambda x : f"results/widar3/ori/{x}/checkpoint_current.pth.tar"
# MODEL_PATHS = [
# 	get_path("TS_ori1_v4.3.1.1"),
# 	get_path("TS_ori1_v4.3.1.2"),
# 	get_path("TS_ori1_v4.3.1.3"),
# 	get_path("TS_ori1_v4.3.2.1"),
# 	get_path("TS_ori1_v4.3.2.2"),
# 	get_path("TS_ori1_v4.3.4.2"),
# 	get_path("TS_ori1_v4.3.4.3"),
# ]

# MODEL_PATHS = [
# 	get_path("TS_ori1_v4.3.4.1"),
# 	get_path("TS_ori1_v4.3.4.2"),
# 	get_path("TS_ori1_v4.3.4.3"),
# 	get_path("TS_ori1_v4.3.3.1"),
# 	get_path("TS_ori1_v4.3.2.3"),
# ]
#
MODEL_PATHS = [
	get_path("baseline1"),
	get_path("TS_ori1_v4.3.4.3"),
	get_path("TS_ori1_v4.3.3.1"),
	get_path("TS_ori1_v4.3.3.2"),
	get_path("TS_ori1_v4.3.3.3"),

]

# MODEL_PATHS = [
# 	get_path("TS_ori5_v4.3.1.1"),
# 	get_path("TS_ori5_v4.3.1.2"),
# 	get_path("TS_ori5_v4.3.1.3"),
# 	get_path("TS_ori5_v4.3.1.4"),
# 	get_path("TS_ori5_v4.3.2.1"),
# ]

# load models
USING_MODELS = []
for i, path in enumerate(MODEL_PATHS, start=1):
	net = utils.get_model(args.name, args)
	checkpoint = torch.load(path, map_location='cuda:0')
	utils.load_pretrained_model(net, checkpoint['net'])
	if args.cuda :
		net = net.cuda()
	net.eval()
	for param in net.parameters() :
		param.requires_grad = False
	logging.info(f'Load model{i}: {path}, prec@1: {checkpoint["prec@1"]:.3}, prec@3: {checkpoint["prec@3"]:.3}')
	USING_MODELS.append(net)


########################## run ########################################
def test(dataloader):
	true_samples = 0
	for x, y in dataloader:
		if args.cuda:
			x = x.cuda()
			y = y.cuda()
		final_probs = []
		final_labels = []
		for model in USING_MODELS:
			logits = model(x)
			prob = F.softmax(logits, dim=1).cpu().numpy()
			final_probs.append(prob)
			one_hot = np.zeros(prob.shape)
			one_hot[np.arange(prob.shape[0]), np.argmax(prob, axis=-1)] = 1
			final_labels.append(one_hot)
		final_probs = np.array(final_probs)
		final_labels = np.array(final_labels)
		# choose vote mode
		if args.vote_mode == 'max' :
			s = np.sum(final_labels, axis=0)
			y_hat = np.argmax(s, axis=-1)
		elif args.vote_mode == 'avg' :
			avg_prob = np.sum(final_probs, axis=0) / final_probs.shape[0]
			y_hat = np.argmax(avg_prob, axis=1)
		else :
			raise NotImplementedError("Unknown vote mode: %s" % args.vote_mode)
		y = y.cpu().numpy()
		true_num = np.sum(y_hat==y)
		true_samples += true_num
	return true_samples

def main():
	# define test set
	test_dataset = dataset_widar3.DomainSet(args.data_dir, args.domain_name, args.test_domain, preread=preread,
	                                        isTest=True)
	test_loader = torch.utils.data.DataLoader(test_dataset,
	                                          batch_size=args.batch_size, shuffle=True,
	                                          num_workers=args.num_workers, pin_memory=True, drop_last=False)

	true_num = test(test_loader)
	total_num = len(test_loader.dataset)
	logging.info('voting mode: %s'%args.vote_mode)
	logging.info('voting results:\n'
	             'testing on %d samples, correct samples is %d, prec1 is %.1f'%
	                (total_num, true_num, true_num/total_num*100))

main()
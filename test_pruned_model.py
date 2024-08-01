import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
from thop import profile
import argparse
from utils import load_pretrained_model, get_model
import torch_pruning as tp

def measure_model_performance(model, input_size):
    model.eval()
    input_data = torch.randn(input_size).cuda()

    print("Model Structure:")
    print(model)

    start_time = time.time()
    with torch.no_grad():
        _ = model(input_data)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.6f} seconds")

    flops, params = profile(model, inputs=(input_data,))
    print(f"FLOPs: {flops:,}")
    print(f"Params: {params:,}")

def check_zero_elements(model):
    zero_elements = {}
    for name, param in model.named_parameters():
        zero_count = torch.sum(param == 0).item()
        total_elements = param.numel()
        if zero_count > 0:
            zero_elements[name] = (zero_count, total_elements)
    return zero_elements

parser = argparse.ArgumentParser(description='Test pruned model')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
parser.add_argument('--model_name', type=str, default='Convnet', help='Model name: Convnet / resnet18 / etc.')

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
parser.add_argument('--is_pruned', type=int, default=0)

# hyperparameter
parser.add_argument('--feat_dim', type=int, default=512)
parser.add_argument('--softplus_beta', type=float, default=1.0)
parser.add_argument('--kd_mode', default='st', type=str, help='mode of kd, which can be:'
                                                              'logits/st/at/fitnet/nst/pkt/fsp/rkd/ab/'
                                                              'sp/sobolev/cc/lwm/irg/vid/ofd/afd')

# others
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()

model = get_model(args.model_name, args)
checkpoint = torch.load(args.model_path)
load_pretrained_model(model, checkpoint['net'])
model = model.cuda()
print('Before:')
print(model)
for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            weight_mask = module.weight != 0
            num_pruned = torch.sum(~weight_mask).item()
            num_total = module.weight.numel()
            print(f"Module: {name} - Pruned {num_pruned} / {num_total} weights")
print('After:')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

example_inputs = torch.randn(64, 6, 121, 121).to(device)

DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

def prune_conv(DG, conv, channels_to_prune):
    pruning_group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=channels_to_prune)
    pruning_group.prune()

def prune_linear(DG, linear, channels_to_prune):
    pruning_group = DG.get_pruning_group(linear, tp.prune_linear_out_channels, idxs=channels_to_prune)
    pruning_group.prune()

modules = list(model.named_modules())
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
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        weight_mask = module.weight != 0
        num_pruned = torch.sum(~weight_mask).item()
        num_total = module.weight.numel()
        print(f"Module: {name} - Pruned {num_pruned} / {num_total} weights")

# print(model)

measure_model_performance(model, (64, 6, 121, 121))

import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.construct import model_construct
from torch_geometric.datasets import Planetoid,Flickr,Amazon

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer_cite', 
                    help='Dataset',
                    choices=['cora_cite','citeseer_cite','house'])
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='MLP', help='model',
                    choices=['HyperGCN','HGNN','HNHN','AllDeepSets','AllSetTransformer','MLP'])
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--sample_num', type=int, default=32,
                    help='Number of samples in sage.')

args = parser.parse_known_args()[0]
args.cuda =  torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from utils import load_hypergraph_data, get_split_hypergraph,normalize_features

data = load_hypergraph_data(args.dataset, root_dir='./data/')
data.H = data.H.to(device)
data.x = data.x.to(device)
data.y = data.y.to(device)
data.x = normalize_features(data.x)
data.train_mask = data.train_mask.to(device)
data.val_mask = data.val_mask.to(device)
data.test_mask = data.test_mask.to(device)

data, idx_train, idx_val, idx_clean_test, idx_atk = get_split_hypergraph(args, data, device)

import sage_modified as sage

# 构建和训练模型
pre_train = model_construct(args,args.model,data,device).to(device)
pre_train.fit(data.x, data.y, idx_train, idx_val, train_iters=args.epochs,verbose=False)

x, y = data.x.cpu().numpy(), data.y
num_classes = torch.max(y) + 1
y = torch.nn.functional.one_hot(y, num_classes=num_classes).cpu().numpy()
# 创建特征名称列表
feature_names = [str(i) for i in range(0, data.x.shape[1])]

# 特征选择（SAGE模块）
model = pre_train
imputer = sage.MarginalImputer(model, x[:args.sample_num])
estimator = sage.PermutationEstimator(imputer, 'mse')
sage_values = estimator(x, y)

val, std = sage_values.save_num()

directory = f'save_selected_feature/{args.dataset}'

if not os.path.exists(directory):
    os.makedirs(directory)

np.save(f'{directory}/val_{args.sample_num}.npy', val)
np.save(f'{directory}/std_{args.sample_num}.npy', std)

figure = sage_values.plot(feature_names,return_fig=True)

save_dir = f'{args.dataset}/saved_plot.png'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

figure.savefig(save_dir, dpi=600, bbox_inches='tight')

plt.show()



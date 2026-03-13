import argparse
import numpy as np
import torch
import torch.nn.functional as F

from help_funcs import reconstruct_prune_unrelated_hyperedge
from help_funcs import prune_unrelated_hyperedge
from select_sample import select
from utils import load_hypergraph_data, get_split_hypergraph, subgraph_hypergraph, normalize_features, build_uniform_hyperedge_weights

# ---- Timestamped logging setup ----
import os, sys, atexit, datetime, builtins

def setup_ts_logger(log_dir='logs', filename_prefix='run'):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(log_dir, f'{filename_prefix}_{ts}.log')
    fh = open(path, 'a', encoding='utf-8', buffering=1)

    def ts_print(*args, **kwargs):
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        msg = sep.join(str(a) for a in args)
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{now}] {msg}{end}'
        sys.__stdout__.write(line)
        fh.write(line)
    builtins.print = ts_print

    class _StderrWrapper:
        def write(self, s):
            if not s:
                return
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            line = f'[{now}] {s}' if s.rstrip() else s
            sys.__stderr__.write(line)
            fh.write(line)
        def flush(self):
            sys.__stderr__.flush(); fh.flush()
    sys.stderr = _StderrWrapper()

    atexit.register(lambda: fh.close())
    return path

LOG_PATH = setup_ts_logger()
# ---- End logging setup ----

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='HyperSAGE', help='model',
                    choices=['HyperGCN','HGNN','HNHN','AllDeepSets','AllSetTransformer','HyperSAGE'])
parser.add_argument('--dataset', type=str, default='cora_cite', 
                    help='Dataset',
                    choices=['cora_cite','citeseer_cite','house','pubmed_cite','cora_coauth','dblp_copub'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8,
                    help='Number of hidden units.')
parser.add_argument('--hgnn_layers', type=int, default=2,
                    help='Number of dhg layers (HGNN/HNHN).')
parser.add_argument('--alldeep_layers', type=int, default=2,
                    help='Number of AllDeepSets propagation layers.')
parser.add_argument('--alldeep_hidden', type=int, default=None,
                    help='Hidden dimension for AllDeepSets MLP blocks (defaults to --hidden).')
parser.add_argument('--alldeep_mlp_layers', type=int, default=2,
                    help='Number of layers inside AllDeepSets HalfNLHconv MLPs.')
parser.add_argument('--alldeep_classifier_hidden', type=int, default=None,
                    help='Hidden dimension for AllDeepSets classifier (defaults to AllDeepSets hidden).')
parser.add_argument('--alldeep_classifier_layers', type=int, default=1,
                    help='Number of layers in the AllDeepSets classifier MLP.')
parser.add_argument('--alldeep_heads', type=int, default=1,
                    help='Number of attention heads for AllDeepSets (if attention is enabled).')
parser.add_argument('--alldeep_aggr', type=str, default='add',
                    choices=['add','mean','max'],
                    help='Aggregation function used in AllDeepSets.')
parser.add_argument('--alldeep_norm', type=str, default='bn',
                    choices=['bn','ln','None'],
                    help='Normalization type within AllDeepSets MLP blocks.')
parser.add_argument('--alldeep_input_norm', action='store_true',
                    help='Apply input normalization in AllDeepSets MLP blocks.')
parser.add_argument('--alldeep_gpr', action='store_true',
                    help='Enable the GPR variant for AllDeepSets.')
parser.add_argument('--allset_layers', type=int, default=2,
                    help='Number of AllSetTransformer propagation layers.')
parser.add_argument('--allset_hidden', type=int, default=None,
                    help='Hidden dim for AllSetTransformer (defaults to --hidden).')
parser.add_argument('--allset_mlp_layers', type=int, default=2,
                    help='MLP depth inside AllSetTransformer blocks.')
parser.add_argument('--allset_classifier_hidden', type=int, default=None,
                    help='Hidden dim for AllSetTransformer classifier (defaults to allset_hidden).')
parser.add_argument('--allset_classifier_layers', type=int, default=1,
                    help='Number of classifier MLP layers for AllSetTransformer.')
parser.add_argument('--allset_heads', type=int, default=4,
                    help='Number of attention heads for AllSetTransformer.')
parser.add_argument('--allset_aggr', type=str, default='mean',
                    choices=['add','mean','max'],
                    help='Aggregation in AllSetTransformer.')
parser.add_argument('--allset_norm', type=str, default='bn',
                    choices=['bn','ln','None'],
                    help='Normalization type in AllSetTransformer MLPs.')
parser.add_argument('--allset_input_norm', action='store_true',
                    help='Apply input norm in AllSetTransformer MLPs.')
parser.add_argument('--hypersage_layers', type=int, default=2,
                    help='Number of HyperSAGE propagation layers.')
parser.add_argument('--hypersage_power', type=float, default=2.0,
                    help='Power parameter for HyperSAGE aggregation.')
parser.add_argument('--hypersage_num_sample', type=int, default=16,
                    help='Neighborhood sample size for HyperSAGE.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=200, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')


parser.add_argument('--shadow_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trojan_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=10,
                    help="number of poisoning nodes relative to the full graph")
parser.add_argument('--defense_mode', type=str, default="prune",
                    choices=['prune', 'none','reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.1,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=0.1,
                    help="Weight of optimize similarity loss")
parser.add_argument('--cos_inv_loss_weight', type=float, default=0.1,
                    help="Weight of optimize cosine invariance loss")
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--test_model', type=str, default='HyperSAGE',
                    choices=['HyperGCN','HGNN','HNHN','AllDeepSets','AllSetTransformer','HyperSAGE'],
                    help='Model used to attack')
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")

parser.add_argument('--alpha', type=float, default=0.02,
                    help="Ratio of feature dimensions to perturb")
parser.add_argument('--alpha_int', type=int, default=30,
                    help="Number of feature dimensions to perturb")
parser.add_argument('--outter_size', type=int, default=512,
                    help="Number of outter samples")

parser.add_argument('--rec_epochs', type=int,  default=100,
                    help='Number of epochs to train benign and backdoor model.')
args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

data = load_hypergraph_data(args.dataset, root_dir='./data/')
data.H = data.H.to(device)
data.x = data.X.to(device)
data.y = data.Y.to(device)
data.x = normalize_features(data.x)
data.train_mask = data.train_mask.to(device)
data.val_mask = data.val_mask.to(device)
data.test_mask = data.test_mask.to(device)

data, idx_train, idx_val, idx_clean_test, idx_atk = get_split_hypergraph(args, data, device)

train_H, selected_hyperedges = subgraph_hypergraph(torch.bitwise_not(data.test_mask), data.H, relabel_nodes=False)
edge_mask = torch.isin(data.H[1], selected_hyperedges)
mask_H = data.H[:, torch.bitwise_not(edge_mask)].to(device)
train_H = train_H.to(device)

from models.backdoor_GCN_cos import Backdoor
from models.construct import model_construct

# 选择未标记节点
unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("Attach Nodes:{}".format(size))
assert size>0, 'The number of selected trigger nodes must be larger than 0!'
# 选择投毒节点
idx_attach = select(data,args,idx_train,idx_val,device).to(device)

print("idx_attach: {}".format(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print(unlabeled_idx)

# 初始化后门模型并训练
model = Backdoor(args,device)
model.fit(data.x, train_H, None, data.y, idx_train,idx_attach, unlabeled_idx)
# 获取被投毒的数据
poison_x, poison_H, poison_edge_weights, poison_labels = model.get_poisoned()

poison_x = poison_x.to(device)
poison_H = poison_H.to(device)
poison_labels = poison_labels.to(device)
poison_edge_weights = build_uniform_hyperedge_weights(poison_H, device)

# 防御机制选择：剪枝或重构
if args.defense_mode == 'prune':
    poison_H, _ = prune_unrelated_hyperedge(args, poison_H, poison_edge_weights, poison_x, device, large_graph=False)
    poison_H = poison_H.to(device)
elif args.defense_mode == 'reconstruct':
    poison_H, _ = reconstruct_prune_unrelated_hyperedge(args, poison_H, poison_edge_weights, poison_x, data.x, data.H, device, idx_attach, large_graph=True)
    poison_H = poison_H.to(device)
poison_edge_weights = build_uniform_hyperedge_weights(poison_H, device)
bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
print(f"Precent of left attach nodes: {len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach):.3f}")

# 初始化评估指标
total_overall_asr = 0
total_overall_ca = 0
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=1)
overall_asr = 0
overall_ca = 0

# 评估模型性能
for seed in seeds:
    args.seed = seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 构建并训练测试模型
    test_model = model_construct(args,args.test_model,data,device).to(device) 
    test_model.fit(poison_x, poison_H, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=False)

    # 使用训练好的模型在带有后门触发器的数据上进行推理
    output = test_model(poison_x,poison_H,poison_edge_weights)
    # 计算目标类别在投毒节点上的分类成功率
    train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
    print("target class rate on Vs: {:.4f}".format(train_attach_rate))

    # 将投毒边和掩码边合并
    induct_H = torch.cat([poison_H, mask_H], dim=1)
    induct_edge_weights = build_uniform_hyperedge_weights(induct_H, device)
    # 使用合并后的边进行测试
    clean_acc = test_model.test(poison_x,induct_H,induct_edge_weights,data.y,idx_clean_test)
    print("accuracy on clean test nodes: {:.4f}".format(clean_acc))
    
    # 进行攻击测试
    induct_x, induct_H, induct_edge_weights = model.inject_trigger(idx_atk, poison_x, induct_H, induct_edge_weights, device)
    induct_x = induct_x.clone().detach().to(device)
    induct_H = induct_H.clone().detach().to(device)
    induct_edge_weights = induct_edge_weights.clone().detach().to(device) if induct_edge_weights is not None else None
    if args.defense_mode == 'prune':
        induct_H, induct_edge_weights = prune_unrelated_hyperedge(args, induct_H, induct_edge_weights, induct_x, device)
        induct_H = induct_H.to(device)
        if induct_edge_weights is not None:
            induct_edge_weights = induct_edge_weights.to(device)
    # 使用注入触发器后的数据进行推理
    output = test_model(induct_x, induct_H, induct_edge_weights)
    # 计算攻击成功率，即投毒节点被正确分类为目标类别
    train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    print("ASR: {:.4f}".format(train_attach_rate))
    asr = train_attach_rate
    # 选出原本不属于目标类别的攻击节点
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    # 计算这些误分类节点的攻击成功率
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
    induct_x, induct_H, induct_edge_weights = induct_x.cpu(), induct_H.cpu(), induct_edge_weights.cpu()
    output = output.cpu()
    overall_asr += asr
    overall_ca += clean_acc
    test_model = test_model.cpu()

# 计算最终结果
overall_asr = overall_asr/len(seeds)
overall_ca = overall_ca/len(seeds)
print("Overall ASR: {:.4f} ({} model, Seed: {})".format(overall_asr, args.test_model, args.seed))
print("Overall Clean Accuracy: {:.4f}".format(overall_ca))
total_overall_asr += overall_asr
total_overall_ca += overall_ca
test_model.to(torch.device('cpu'))
torch.cuda.empty_cache()



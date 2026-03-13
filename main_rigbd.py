import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli

from help_funcs import reconstruct_prune_unrelated_hyperedge
from help_funcs import prune_unrelated_hyperedge
from select_sample import select
from utils import (
    load_hypergraph_data,
    get_split_hypergraph,
    subgraph_hypergraph,
    normalize_features,
    build_uniform_hyperedge_weights,
)


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--model', type=str, default='HyperGCN', help='model',
                    choices=['HyperGCN','HGNN','HNHN','AllDeepSets'])
parser.add_argument('--dataset', type=str, default='cora_cite', 
                    help='Dataset',
                    choices=['cora_cite','citeseer_cite','house','pubmed_cite','cora_coauth','dblp_copub'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')


parser.add_argument('--shadow_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trojan_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=40,
                    help="number of poisoning nodes relative to the full graph")
parser.add_argument('--defense_mode', type=str, default="prune",
                    choices=['prune', 'none','reconstruct'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--homo_loss_weight', type=float, default=0.1,
                    help="Weight of optimize similarity loss")
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--test_model', type=str, default='HyperGCN',
                    choices=['HyperGCN','HGNN','HNHN','AllDeepSets'],
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

# Load hypergraph data (follow main.py style)
data = load_hypergraph_data(args.dataset, root_dir='./data/')
data.H = data.H.to(device)
data.x = data.X.to(device)
data.y = data.Y.to(device)
data.x = normalize_features(data.x)
data.train_mask = data.train_mask.to(device)
data.val_mask = data.val_mask.to(device)
data.test_mask = data.test_mask.to(device)

# Split into train/val/test/attack sets
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split_hypergraph(args, data, device)

# Build train hypergraph and the held-out part
train_H, selected_hyperedges = subgraph_hypergraph(torch.bitwise_not(data.test_mask), data.H, relabel_nodes=False)
edge_mask = torch.isin(data.H[1], selected_hyperedges)
mask_H = data.H[:, torch.bitwise_not(edge_mask)].to(device)
train_H = train_H.to(device)

from models.backdoor_GCN_cos import Backdoor
from models.construct import model_construct

# unlabeled nodes
unlabeled_idx = (torch.bitwise_not(data.test_mask) & torch.bitwise_not(data.train_mask)).nonzero().flatten()
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("Attach Nodes:{}".format(size))
assert size>0, 'The number of selected trigger nodes must be larger than 0!'

# select attach nodes
idx_attach = select(data, args, idx_train, idx_val, device).to(device)

print("idx_attach: {}".format(idx_attach))
unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
print(unlabeled_idx)

# train backdoor generator
model = Backdoor(args, device)
model.fit(data.x, train_H, None, data.y, idx_train, idx_attach, unlabeled_idx)

# get poisoned graph
poison_x, poison_H, poison_edge_weights, poison_labels = model.get_poisoned()

poison_x = poison_x.to(device)
poison_H = poison_H.to(device)
poison_labels = poison_labels.to(device)
poison_edge_weights = build_uniform_hyperedge_weights(poison_H, device)

# optional defense
if args.defense_mode == 'prune':
    poison_H, _ = prune_unrelated_hyperedge(args, poison_H, None, poison_x, device, large_graph=False)
    poison_H = poison_H.to(device)
elif args.defense_mode == 'reconstruct':
    poison_H, _ = reconstruct_prune_unrelated_hyperedge(args, poison_H, None, poison_x, data.x, data.H, device, idx_attach, large_graph=True)
    poison_H = poison_H.to(device)
poison_edge_weights = build_uniform_hyperedge_weights(poison_H, device)
bkd_tn_nodes = torch.cat([idx_train, idx_attach]).to(device)
print(f"Precent of left attach nodes: {len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist()))/len(idx_attach):.3f}")

known_nodes = torch.cat([idx_train, idx_attach]).to(device)
predictions = []

#### train a backdoored model on poisoned graph #### 
test_model = model_construct(args, args.test_model, data, device).to(device)
test_model.fit(poison_x, poison_H, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=args.epochs, verbose=False)
test_model.eval()

clean_acc = test_model.test(poison_x, poison_H, poison_edge_weights, poison_labels, idx_attach)
output_clean = test_model(poison_x, poison_H, poison_edge_weights)
ori_predict = torch.exp(output_clean[known_nodes])

# merge with held-out hyperedges
induct_H = torch.cat([poison_H, mask_H], dim=1)
induct_edge_weights = build_uniform_hyperedge_weights(induct_H, device)

# inject triggers into merged hypergraph
induct_x, induct_H, induct_edge_weights = model.inject_trigger(idx_atk, poison_x, induct_H, induct_edge_weights, device)
induct_x, induct_H, induct_edge_weights = induct_x.clone().detach(), induct_H.clone().detach(), induct_edge_weights.clone().detach()

# evaluate ASR and CA
output = test_model(induct_x, induct_H, induct_edge_weights)
train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
print("ASR: {:.4f}".format(train_attach_rate))
asr = train_attach_rate
ca = test_model.test(poison_x, induct_H, induct_edge_weights, data.y, idx_clean_test)
print("CA: {:.4f}".format(ca))

###### formal test ########
test_model = model_construct(args, args.test_model, data, device, add_selfloop=False).to(device) 
test_model.fit(poison_x, poison_H, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val, train_iters=args.epochs, verbose=False)
test_model.eval()
clean_acc = test_model.test(poison_x, poison_H, poison_edge_weights, poison_labels, idx_attach)
output_clean = test_model(poison_x, poison_H, poison_edge_weights)
ori_predict = torch.exp(output_clean[known_nodes])
print("accuracy on poisoned target nodes: {:.4f}".format(clean_acc))

# noise sampling on hyperedges
drop_ratio = 0.5

def sample_noise_all(H, edge_weight, device):
    H = H.to(device)
    if edge_weight is None:
        edge_weight = build_uniform_hyperedge_weights(H, device)
    else:
        edge_weight = edge_weight.to(device)

    if H.numel() == 0:
        return H, edge_weight

    he_ids = torch.unique(H[1])
    keep_mask_he = Bernoulli(1 - drop_ratio).sample(he_ids.size()).bool().to(device)
    kept_he = he_ids[keep_mask_he]
    mask_pairs = torch.isin(H[1], kept_he)
    noisy_H = H[:, mask_pairs]
    noisy_edge_weight = build_uniform_hyperedge_weights(noisy_H, device)

    # Restore isolated nodes by adding back original incident pairs
    num_nodes = int(H[0].max().item()) + 1 if H[0].numel() else 0
    node_degrees = torch.zeros(num_nodes, device=device)
    if noisy_H.numel() > 0:
        node_degrees.index_add_(0, noisy_H[0], torch.ones(noisy_H.size(1), device=device))
    isolated_nodes = node_degrees == 0
    if isolated_nodes.any() and H.numel() > 0:
        restore_mask = isolated_nodes[H[0]]
        restore_pairs = H[:, restore_mask]
        if restore_pairs.numel() > 0:
            noisy_H = torch.cat([noisy_H, restore_pairs], dim=1)
            noisy_edge_weight = build_uniform_hyperedge_weights(noisy_H, device)

    return noisy_H, noisy_edge_weight

predictions = [] # 存储预测结果
K=20 # 进行K次预测
for i in range(K):
    test_model.eval()
    noisy_poison_H, noisy_poison_edge_weights = sample_noise_all(poison_H, poison_edge_weights, device)
    output = test_model(poison_x, noisy_poison_H, noisy_poison_edge_weights)
    train_attach_rate = (output.argmax(dim=1)[idx_attach]==args.target_class).float().mean()
    train_clean_rate = (output.argmax(dim=1)[idx_train]==data.y[idx_train]).float().mean()
    predictions.append(torch.exp(output[known_nodes]))

# KL divergence based robustness measure
epsilon = 1e-8
deviations = []
for sub_pred in predictions:
    sub_pred = sub_pred + epsilon
    deviation = F.kl_div(sub_pred.log(), ori_predict, reduce=False)
    deviations.append(deviation)

summed_deviations = torch.zeros_like(deviations[0]).to(deviations[0].device)
for deviation in deviations:
    summed_deviations += deviation

index_of_less_robust = torch.sort(torch.mean(summed_deviations,dim=-1),descending=True)[1]

def find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, target_class):
    labels_list = poison_labels[bkd_tn_nodes[index_of_less_robust]]
    for i in range(len(labels_list) - 1):
        if labels_list[i] != target_class and labels_list[i + 1] != target_class:
            return i - 1
    return None

result_index = find_index(poison_labels, bkd_tn_nodes, index_of_less_robust, args.target_class)
print("Index found:", result_index)

indexs = poison_labels[bkd_tn_nodes[index_of_less_robust][:result_index-1]]
count = 0
for i in indexs:
    if i == args.target_class:
        count += 1
correct = count
false = len(indexs) - count

# finetune on less robust nodes
test_model = model_construct(args, args.test_model, data, device).to(device) 
test_model.fit(poison_x, poison_H, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=400,verbose=False, finetune=True, attach=bkd_tn_nodes[index_of_less_robust][:result_index])

induct_H = torch.cat([poison_H, mask_H], dim=1)
induct_edge_weights = build_uniform_hyperedge_weights(induct_H, device)
induct_x, induct_H, induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_H,induct_edge_weights,device)
induct_x, induct_H,induct_edge_weights = induct_x.clone().detach(), induct_H.clone().detach(),induct_edge_weights.clone().detach()

output = test_model(induct_x,induct_H,induct_edge_weights)

print("****After Defense****")
train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
print("ASR: {:.4f}".format(train_attach_rate))
asr = train_attach_rate
ca = test_model.test(poison_x,induct_H,induct_edge_weights,data.y,idx_clean_test)
print("CA: {:.4f}".format(ca))


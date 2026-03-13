#%%
from copy import deepcopy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
from models.HyperGCN import HyperGraphGCN
from models.HGNN import HyperGraphHGNN
from models.HNHN import HyperGraphHNHN
from models.AllSetTransformer import HyperGraphAllSetTransformer
from models.hypersage import HyperGraphHyperSAGE
from models.AllDeepSets import HyperGraphAllDeepSets

EPS = 1e-12

class GraphTrojanNet(nn.Module):
    def __init__(self, device, nfeat, dim_num, layernum=2, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for _ in range(layernum - 1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat, dim_num)
        self.edge = nn.Linear(nfeat, int(dim_num * (dim_num - 1) / 2))
        self.device = device

    def forward(self, input, thrd):
        h = self.layers(input)
        feat = self.feat(h)
        return feat


class HomoLoss(nn.Module):
    """同质性损失"""
    def __init__(self, args, device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self, H: torch.Tensor, hyperedge_weight: Optional[torch.Tensor], x: torch.Tensor, thrd: float):
        # 1) 将 H 统一转换为稠密关联矩阵 incidence
        incidence = _to_incidence_matrix(H, x.size(0), x.device, x.dtype)
        # 2) 若给定了超边权重，只保留权重大于 0 的超边
        mask = None
        if hyperedge_weight is not None:
            mask = hyperedge_weight > 0
            if mask.any():
                incidence = incidence[mask]
            else:
                return torch.tensor(0.0, device=x.device)
        # 3) 指示矩阵：indicator[i, j] = True 表示节点 j 属于超边 i
        indicator = incidence > 0
        sims = [] # 用于收集所有超边内的成对相似度
        # 4) 逐条超边处理
        for idx in range(indicator.size(0)):
            # 4.1 找出当前超边包含的节点索引
            node_idx = torch.nonzero(indicator[idx], as_tuple=False).flatten()
            if node_idx.numel() < 2:
                # 少于 2 个节点无法成对，跳过
                continue
            # 4.2 取出这些节点的特征，做 L2 归一化，计算余弦相似度矩阵
            feats = F.normalize(x[node_idx], p=2, dim=1)
            sim_mat = torch.matmul(feats, feats.t())
            # 4.3 仅取下三角（不含对角）的成对相似度，避免重复
            pair_idx = torch.tril_indices(node_idx.numel(), node_idx.numel(), offset=-1)
            pair_sims = sim_mat[pair_idx[0], pair_idx[1]]
            sims.append(pair_sims)
        # 5) 若没有任何可用的成对相似度（例如所有超边都只有1个节点），损失为 0
        if not sims:
            return torch.tensor(0.0, device=x.device)
        # 6) 拼接所有超边的成对相似度
        sims = torch.cat(sims)
        # 7) 计算同质性损失：当相似度低于阈值 thrd 时产生正损失
        loss = torch.relu(thrd - sims).mean()
        return loss


class Backdoor:

    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.feature_indices = None
        self.hyperedge_weight = None
        self.H_structure = None
        self.H_dense = None
        self.labels = None
        self.features = None
        self.shadow_model = None
        self.trojan = None
        self.idx_attach = None
        self.hyperedge_mask = None
        self.cos_inv_loss_weight = getattr(args, 'cos_inv_loss_weight', 0.0)
        self.base_model_name = getattr(args, 'model', 'HyperGCN')
        self.hgnn_layers = getattr(args, 'hgnn_layers', 2)

    def _select_feature_indices(self, dim_num: int) -> np.ndarray:
        mapping = {
            'cora_cite': '32',
            'citeseer_cite': '32',
            
        }
        dataset = self.args.dataset
        if dataset not in mapping:
            raise ValueError(f"Unsupported dataset '{dataset}' for feature selection.")
        sage_epoch = mapping[dataset]
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f'../save_selected_feature/{dataset}/val_{sage_epoch}.npy')
        if os.path.exists(file_path):
            dim_all = np.load(file_path)
            dim = np.argsort(dim_all)[::-1][:dim_num]
            return dim.copy()
        else:
            print(f"Feature selection file not found: {file_path}, using first {dim_num} dimensions.")
            dim = np.arange(dim_num)
            return dim

    def inject_trigger(self, idx_attach, features, H, hyperedge_weight, device):
        features = features.to(device)
        idx_attach = idx_attach.to(device)
        hyperedge_weight = hyperedge_weight.to(device)

        self.trojan = self.trojan.to(device)
        self.shadow_model = self.shadow_model.to(device)
        self.trojan.eval()
        self.shadow_model.eval()

        embed = self.shadow_model.get_h(features, self.H_dense, self.hyperedge_weight)
        trojan_feat = self.trojan(embed[idx_attach], self.args.thrd)
        update_feat = features.clone()

        idx = self.feature_indices.to(device)
        update_feat[idx_attach[:, None], idx] = trojan_feat.detach()

        self.trojan = self.trojan.cpu()
        update_feat_cpu = update_feat.detach().cpu()
        H_cpu = H.detach().cpu()
        weight_cpu = hyperedge_weight.detach().cpu()
        return update_feat_cpu, H_cpu, weight_cpu

    def fit(self, features, H, hyperedge_weight, labels, idx_train, idx_attach, idx_unlabeled):
        args = self.args
        features = features.to(self.device)
        labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_attach = idx_attach.to(self.device)
        idx_unlabeled = idx_unlabeled.to(self.device)

        H_input = H.detach().clone()
        H_device = H.to(self.device)
        incidence = _to_incidence_matrix(H_device, features.size(0), self.device, features.dtype)
        num_hyperedges = incidence.size(0)

        self.idx_attach = idx_attach
        self.features = features
        self.H_structure = H_input
        self.H_dense = incidence

        if hyperedge_weight is None:
            hyperedge_weight = torch.ones(num_hyperedges, device=self.device, dtype=features.dtype)
        else:
            hyperedge_weight = hyperedge_weight.to(self.device, dtype=features.dtype)
            if hyperedge_weight.numel() != num_hyperedges:
                raise ValueError("Length of hyperedge_weight must match number of hyperedges.")
        self.hyperedge_weight = hyperedge_weight
        self.hyperedge_mask = self.hyperedge_weight > 0 if self.hyperedge_weight is not None else None

        dim_num = args.alpha_int
        dim = self._select_feature_indices(dim_num)
        self.feature_indices = torch.from_numpy(dim).long().to(self.device)

        common_model_kwargs = dict(
            nfeat=features.shape[1],
            nhid=self.args.hidden,
            nclass=int(labels.max().item() + 1),
            dropout=0.0,
            device=self.device,
            lr=self.args.shadow_lr,
            weight_decay=self.args.weight_decay,
        )
        if self.base_model_name == 'HGNN':
            self.shadow_model = HyperGraphHGNN(**common_model_kwargs, num_layers=self.hgnn_layers).to(self.device)
        elif self.base_model_name == 'HNHN':
            self.shadow_model = HyperGraphHNHN(**common_model_kwargs, num_layers=self.hgnn_layers).to(self.device)
        elif self.base_model_name == 'AllDeepSets':
            alldeep_kwargs = dict(common_model_kwargs)
            alldeep_hidden = getattr(self.args, 'alldeep_hidden', None)
            if alldeep_hidden is None:
                alldeep_hidden = self.args.hidden
            alldeep_kwargs['nhid'] = alldeep_hidden
            self.shadow_model = HyperGraphAllDeepSets(**alldeep_kwargs, base_args=self.args).to(self.device)
        elif self.base_model_name == 'AllSetTransformer':
            allset_kwargs = dict(common_model_kwargs)
            allset_hidden = getattr(self.args, 'allset_hidden', None)
            if allset_hidden is None:
                allset_hidden = self.args.hidden
            allset_kwargs['nhid'] = allset_hidden
            self.shadow_model = HyperGraphAllSetTransformer(**allset_kwargs, base_args=self.args).to(self.device)
        elif self.base_model_name == 'HyperSAGE':
            hypersage_kwargs = dict(common_model_kwargs)
            hypersage_hidden = getattr(self.args, 'hypersage_hidden', None)
            if hypersage_hidden is None:
                hypersage_hidden = self.args.hidden
            hypersage_kwargs['nhid'] = hypersage_hidden
            self.shadow_model = HyperGraphHyperSAGE(**hypersage_kwargs, base_args=self.args).to(self.device)
        else:
            self.shadow_model = HyperGraphGCN(**common_model_kwargs).to(self.device)
        embed = self.shadow_model.get_h(features, self.H_dense, self.hyperedge_weight)
        self.trojan = GraphTrojanNet(self.device, embed.shape[1], dim_num, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args, self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.shadow_lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.trojan_lr, weight_decay=args.weight_decay)

        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        poison_x = features.clone().detach()
        loss_best = float('inf')

        idx_unlabeled_cpu = idx_unlabeled.detach().cpu().numpy()
        rs = np.random.RandomState(self.args.seed) if len(idx_unlabeled_cpu) > 0 else None

        for i in range(args.trojan_epochs):
            self.trojan.train()
            for _ in range(self.args.inner):
                optimizer_shadow.zero_grad()

                embed = self.shadow_model.get_h(poison_x, self.H_dense, self.hyperedge_weight)
                trojan_feat = self.trojan(embed[idx_attach], args.thrd)
                poison_x = features.clone().detach()
                idx = self.feature_indices
                poison_x[idx_attach[:, None], idx] = trojan_feat.detach()

                output = self.shadow_model(poison_x, self.H_dense, self.hyperedge_weight)
                joint_idx = torch.cat([idx_train, idx_attach])
                loss_inner = F.nll_loss(output[joint_idx], self.labels[joint_idx])

                loss_inner.backward()
                optimizer_shadow.step()

            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])

            optimizer_trigger.zero_grad()
            if len(idx_unlabeled_cpu) == 0:
                outter = idx_attach.clone()
            else:
                sample_size = min(args.outter_size, len(idx_unlabeled_cpu))
                choices = rs.choice(len(idx_unlabeled_cpu), size=sample_size, replace=False)
                outter = torch.tensor(idx_unlabeled_cpu[choices], device=self.device, dtype=idx_attach.dtype)
            idx_outter = torch.cat([idx_attach, outter])

            embed = self.shadow_model.get_h(poison_x, self.H_dense, self.hyperedge_weight)
            trojan_feat = self.trojan(embed[idx_outter], args.thrd)

            update_feat = features.clone()
            idx = self.feature_indices
            update_feat[idx_outter[:, None], idx] = trojan_feat

            output = self.shadow_model(update_feat, self.H_dense, self.hyperedge_weight)

            # Cosine invariance loss keeps hyperedge embeddings close to clean ones
            loss_inv_cos = update_feat.new_tensor(0.0)
            if self.cos_inv_loss_weight > 0:
                with torch.no_grad():
                    clean_embed_nodes = self.shadow_model.get_h(self.features, self.H_dense, self.hyperedge_weight)
                attacked_embed_nodes = self.shadow_model.get_h(update_feat, self.H_dense, self.hyperedge_weight)
                hyperedge_sizes = self.H_dense.sum(dim=1, keepdim=True).clamp_min(EPS)
                clean_hyper = torch.matmul(self.H_dense, clean_embed_nodes) / hyperedge_sizes
                attacked_hyper = torch.matmul(self.H_dense, attacked_embed_nodes) / hyperedge_sizes
                clean_hyper = F.normalize(clean_hyper, p=2, dim=1)
                attacked_hyper = F.normalize(attacked_hyper, p=2, dim=1)
                cos_sim = (attacked_hyper * clean_hyper).sum(dim=1)
                weights = self.hyperedge_weight
                loss_inv_cos = (weights * (1 - cos_sim)).sum() / weights.sum().clamp_min(EPS)

            labels_outter = labels.clone().to(self.device)
            labels_outter[idx_outter] = args.target_class

            loss_sim = 1 - F.cosine_similarity(update_feat, self.features).mean()
            joint_idx = torch.cat([idx_train, idx_outter])
            loss_target = self.args.target_loss_weight * F.nll_loss(output[joint_idx], labels_outter[joint_idx])

            loss_outter = (
                loss_target
                + self.args.homo_loss_weight * loss_sim
                + self.cos_inv_loss_weight * loss_inv_cos
            )

            loss_outter.backward()
            optimizer_trigger.step()

            acc_train_outter = (output[idx_outter].argmax(dim=1) == args.target_class).float().mean()

            if loss_outter < loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

            if args.debug and i % 50 == 0:
                print(
                    'Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, loss_sim: {:.5f}, loss_inv_cos: {:.5f}'.format(
                        i,
                        loss_inner,
                        loss_target,
                        loss_sim,
                        loss_inv_cos,
                    )
                )
                print(
                    "ACC: {:.4f}, ASR_train: {:.4f}, OUTTER: {:.4f}".format(
                        acc_train_clean,
                        acc_train_attach,
                        acc_train_outter,
                    )
                )

        if args.debug:
            print(f"load best weight based on the loss outter {loss_best}")
        print(self.feature_indices.detach().cpu().numpy())
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

    def get_poisoned(self):
        with torch.no_grad():
            poison_x, poison_H, poison_weights = self.inject_trigger(
                self.idx_attach,
                self.features,
                self.H_structure,
                self.hyperedge_weight,
                self.device,
            )
        poison_labels = self.labels.detach().cpu()
        return poison_x, poison_H, poison_weights, poison_labels


def _to_incidence_matrix(H: torch.Tensor, num_nodes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """将H转换为稠密的关联矩阵形式"""
    # H 是 [2, E]，即 (node_idx, hyperedge_idx) 的配对形式
    if H.dim() == 2 and H.size(0) == 2:
        pair = H.long()
        if pair.numel() == 0:
            return torch.zeros((0, num_nodes), dtype=dtype, device=device)
        node_idx = pair[0]
        hyperedge_idx = pair[1]
        num_hyperedges = int(hyperedge_idx.max().item()) + 1
        indices = torch.stack([hyperedge_idx, node_idx])
        values = torch.ones(indices.size(1), dtype=dtype, device=device)
        incidence = torch.sparse_coo_tensor(indices, values, (num_hyperedges, num_nodes), device=device)
        return incidence.to_dense()

    # H 已经是矩阵形式
    incidence = H.to(device=device, dtype=dtype)
    if incidence.dim() != 2:
        raise ValueError("Hypergraph incidence must be a 2-D tensor.")
    # 如果列数与节点数对齐，直接返回
    if incidence.size(1) == num_nodes:
        return incidence
    # 如果行数与节点数对齐，则需要转置
    if incidence.size(0) == num_nodes:
        return incidence.t()
    # 其他情况均视为不匹配
    raise ValueError("Unable to align hypergraph incidence with given node count.")


def _hyperedge_label_distribution(
    incidence: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算每个超边的标签分布"""
    # 如果提供了掩码，只保留部分超边
    if mask is not None:
        incidence = incidence[mask]
    # 将标签转换为 one-hot 形式
    if labels.dim() == 1:
        labels_onehot = utils.tensor2onehot(labels)
    else:
        labels_onehot = labels
    labels_onehot = labels_onehot.to(device=incidence.device, dtype=incidence.dtype)
    # 计算每条超边内各类别的节点计数
    counts = incidence @ labels_onehot
    # 计算每条超边的节点数量
    sizes = incidence.sum(dim=1, keepdim=True).clamp_min(EPS)
    return counts / sizes


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """计算两个分布之间的平均KL散度"""
    if p.numel() == 0:
        return torch.tensor(0.0, device=q.device if q.is_cuda else p.device)
    q = q.clamp_min(EPS)
    p = p.clamp_min(EPS)
    kl = (p * (p.log() - q.log())).sum(dim=1)
    return kl.mean()




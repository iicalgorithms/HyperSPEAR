from copy import deepcopy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils

EPS = 1e-12

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

def _hypergraph_propagation(x: torch.Tensor, H: torch.Tensor, hyperedge_weight: Optional[torch.Tensor]) -> torch.Tensor:
    """基于超图的特征传播"""
    # 1) 将传入的 H 统一转换为稠密关联矩阵 Incidence (num_hyperedges x num_nodes)
    incidence = _to_incidence_matrix(H, x.size(0), x.device, x.dtype)
    num_hyperedges = incidence.size(0)

    # 2) 规范化超边权重 hyperedge_weight
    # - 若未提供：默认为全 1
    # - 若长度等于 H 的列数（E 条 (node, he) 对）：按超边聚合成每条超边的平均权重
    # - 若长度等于超边数：直接使用
    if hyperedge_weight is None or hyperedge_weight.numel() == 0:
        hyperedge_weight = torch.ones(num_hyperedges, device=x.device, dtype=x.dtype)
    else:
        hyperedge_weight = hyperedge_weight.to(x.device, dtype=x.dtype)
        if hyperedge_weight.numel() == H.size(1):
            he_ids = H[1].long().to(x.device)
            agg = torch.zeros(num_hyperedges, device=x.device, dtype=x.dtype)
            counts = torch.zeros(num_hyperedges, device=x.device, dtype=x.dtype)
            agg.index_add_(0, he_ids, hyperedge_weight)
            counts.index_add_(0, he_ids, torch.ones_like(he_ids, dtype=x.dtype))
            hyperedge_weight = agg / counts.clamp_min(1.0)
        elif hyperedge_weight.numel() == num_hyperedges:
            pass
        else:
            raise ValueError("Length of hyperedge_weight must match number of hyperedges or incidences.")

    # 3) 计算度量：
    # De: 超边度（每条超边包含的节点数）
    # Dv: 节点度（按超边权重汇总到节点）
    De = incidence.sum(dim=1) + EPS
    Dv = torch.matmul(incidence.t(), hyperedge_weight) + EPS
    Dv_inv_sqrt = torch.pow(Dv, -0.5)

    # 4) 规范化并进行传播：节点 -> 超边 -> 节点
    x_norm = x * Dv_inv_sqrt.unsqueeze(1)
    hx = torch.matmul(incidence, x_norm)
    hx = hx * (hyperedge_weight / De).unsqueeze(1)
    out = torch.matmul(incidence.t(), hx)
    out = out * Dv_inv_sqrt.unsqueeze(1)
    return out


class HyperGraphConv(nn.Module):
    """超图卷积层"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyperedge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.linear(x)
        return _hypergraph_propagation(x, H, hyperedge_weight)


class HyperGraphGCN(nn.Module):
    """超图卷积网络"""
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, device: torch.device, lr: float = 0.01, weight_decay: float = 5e-4):
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(HyperGraphConv(nfeat, nhid))
        self.final = HyperGraphConv(nhid, nclass)
        self.lr = lr
        self.weight_decay = weight_decay
        self.features = None
        self.structure = None
        self.edge_weight = None
        self.labels = None
        self.output = None

    def forward(self, x: torch.Tensor, H: torch.Tensor, hyperedge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.relu(self.convs[0](x, H, hyperedge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.final(x, H, hyperedge_weight)
        return F.log_softmax(x, dim=1)

    def get_h(self, x: torch.Tensor, H: torch.Tensor, hyperedge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.relu(self.convs[0](x, H, hyperedge_weight))

    def fit(self, features, structure, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, **kwargs):
        self.features = features.to(self.device)
        self.structure = structure.to(self.device) if structure is not None else None
        self.edge_weight = edge_weight.to(self.device) if edge_weight is not None else None
        self.labels = labels.to(self.device)
        idx_train = idx_train.to(self.device)
        idx_val = idx_val.to(self.device) if idx_val is not None else None

        if idx_val is None:
            self._train_without_val(idx_train, train_iters, verbose)
        else:
            self._train_with_val(idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, idx_train, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(train_iters):
            self.train()
            optimizer.zero_grad()
            out = self.forward(self.features, self.structure, self.edge_weight)
            loss = F.nll_loss(out[idx_train], self.labels[idx_train])
            loss.backward()
            optimizer.step()
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, training loss: {loss.item():.4f}')
        self.eval()
        self.output = self.forward(self.features, self.structure, self.edge_weight)

    def _train_with_val(self, idx_train, idx_val, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_acc = 0.0
        best_state = None
        for epoch in range(train_iters):
            self.train()
            optimizer.zero_grad()
            out = self.forward(self.features, self.structure, self.edge_weight)
            loss = F.nll_loss(out[idx_train], self.labels[idx_train])
            loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                out = self.forward(self.features, self.structure, self.edge_weight)
                acc_val = utils.accuracy(out[idx_val], self.labels[idx_val])
            if verbose and epoch % 10 == 0:
                print(f'Epoch {epoch}, training loss: {loss.item():.4f}, val acc: {acc_val:.4f}')
            if acc_val > best_acc:
                best_acc = acc_val
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                self.output = out
        if best_state is not None:
            self.load_state_dict(best_state)

    def test(self, features, structure, edge_weight, labels, idx_test):
        self.eval()
        with torch.no_grad():
            out = self.forward(features.to(self.device), structure.to(self.device), edge_weight.to(self.device) if edge_weight is not None else None)
        acc_test = utils.accuracy(out[idx_test], labels[idx_test])
        return float(acc_test)

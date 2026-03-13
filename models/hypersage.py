import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import timeit
import itertools
from copy import deepcopy 
from types import SimpleNamespace
from typing import Optional

from utils import accuracy

class HyperTensorGraphConvolution(Module):

    def __init__(self, a, b):
        super(HyperTensorGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        #self.edge_count = edge_count
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, power,num_sample):
        #self.edge_count = self.edge_count.cuda()
        W, b = self.W, self.bias
        AH = signal_shift_hypergraph_sample(structure,H, power, num_sample) 
        AHW = torch.mm(AH, W) 
        output = AHW + b
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'

class HyperSAGE(nn.Module):

    @staticmethod
    def generate_hyperedge_dict(data):
        """Build hyperedge_dict with node index tensors on the same device as x.

        Expects `data.edge_index` to be a tensor [2, E]. Returns `data` with
        `hyperedge_dict: Dict[he_id -> LongTensor(nodes_on_device)]`.
        """
        He_dict = get_HyperGCN_He_dict(data)
        device = data.x.device if hasattr(data, 'x') and isinstance(data.x, torch.Tensor) else None
        hyperedge_dict = {}
        for e, nodes in He_dict.items():
            if device is None:
                hyperedge_dict[e] = torch.tensor(nodes, dtype=torch.long)
            else:
                hyperedge_dict[e] = torch.tensor(nodes, dtype=torch.long, device=device)
        data.hyperedge_dict = hyperedge_dict
        return data

    def __init__(self, num_features, num_classes, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperSAGE, self).__init__()
        d, l, c = num_features, args.All_num_layers, num_classes
        
        h = [d]
        for i in range(l-1):
            power = l - i + 2
            h.append(2**power)
        h.append(c)

        if args.MLP_hidden >= 16:
            print('Caution: Too large hidden dimension. Running very slow.')

        self.hgc1 = HyperTensorGraphConvolution(d,16)
        self.hgc2 = HyperTensorGraphConvolution(16,c)
        self.do, self.l = args.dropout, args.All_num_layers
        self.power = args.HyperSAGE_power
        self.num_sample = args.HyperSAGE_num_sample

    def reset_parameters(self):
        self.hgc1.reset_parameters()
        self.hgc2.reset_parameters()

    def forward(self, data):
        """
        an l-layer GCN
        """
        H = data.x
        structure = data.hyperedge_dict
        do, l= self.do, self.l
        power = self.power
        num_sample = self.num_sample
        H = F.relu(self.hgc1(structure, H, power, num_sample))
        H = F.dropout(H, do, training=self.training)
        H = self.hgc2(structure, H, power, num_sample)      
        return F.log_softmax(H, dim=1)


class HyperGraphHyperSAGE(nn.Module):
    """Project-aligned HyperSAGE with unified interface.

    Provides forward(x,H,weights)->log_softmax, get_h, fit, test,
    and internal data preparation compatible with this repo.
    """

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        device: torch.device,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        base_args: Optional[SimpleNamespace] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.base_args = base_args or SimpleNamespace()

        self.features = None
        self.structure = None
        self.edge_weight = None
        self.labels = None
        self.output = None

        model_args = self._build_model_args(self.base_args)
        self.model = HyperSAGE(num_features=nfeat, num_classes=nclass, args=model_args).to(device)

    def forward(self, x: torch.Tensor, H: Optional[torch.Tensor], hyperedge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        data = self._prepare_data(x.to(self.device), H)
        return self.model(data)

    def get_h(self, x: torch.Tensor, H: Optional[torch.Tensor], hyperedge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        data = self._prepare_data(x.to(self.device), H)
        structure = data.hyperedge_dict
        power = self.model.power
        num_sample = self.model.num_sample
        h = F.relu(self.model.hgc1(structure, data.x, power, num_sample))
        return h

    def fit(
        self,
        features: torch.Tensor,
        structure: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor],
        labels: torch.Tensor,
        idx_train: torch.Tensor,
        idx_val: Optional[torch.Tensor] = None,
        train_iters: int = 200,
        verbose: bool = False,
        **kwargs,
    ) -> None:
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

    def _train_without_val(self, idx_train: torch.Tensor, train_iters: int, verbose: bool) -> None:
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(train_iters):
            self.train()
            optimizer.zero_grad()
            out = self.forward(self.features, self.structure, self.edge_weight)
            loss = F.nll_loss(out[idx_train], self.labels[idx_train])
            loss.backward()
            optimizer.step()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, training loss: {loss.item():.4f}")
        self.eval()
        self.output = self.forward(self.features, self.structure, self.edge_weight)

    def _train_with_val(self, idx_train: torch.Tensor, idx_val: torch.Tensor, train_iters: int, verbose: bool) -> None:
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
                acc_val = accuracy(out[idx_val], self.labels[idx_val])
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, training loss: {loss.item():.4f}, val acc: {acc_val:.4f}")
            if acc_val > best_acc:
                best_acc = acc_val
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
                self.output = out
        if best_state is not None:
            self.load_state_dict(best_state)

    def test(
        self,
        features: torch.Tensor,
        structure: Optional[torch.Tensor],
        edge_weight: Optional[torch.Tensor],
        labels: torch.Tensor,
        idx_test: torch.Tensor,
    ) -> float:
        self.eval()
        with torch.no_grad():
            out = self.forward(features.to(self.device), structure, edge_weight)
        acc_test = accuracy(out[idx_test.to(self.device)], labels[idx_test].to(self.device))
        return float(acc_test)

    # --- Caching helpers to avoid rebuilding hyperedge dict every forward ---
    def _ensure_prepared_structure(self, H: Optional[torch.Tensor]) -> None:
        """Build and cache edge_index + hyperedge_dict when structure changes.

        This avoids repeated CPU numpy conversions and Python dict builds
        inside every forward call on large hypergraphs.
        """
        # None/empty structure
        if H is None or H.numel() == 0:
            if not hasattr(self, "_cached_edge_index"):
                self._cached_edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                self._cached_hyperedge_dict = {}
                self._cached_structure_ptr = None
            return

        # If we have a cached pointer and it matches, reuse
        try:
            ptr = H.data_ptr()
        except Exception:
            ptr = None

        if getattr(self, "_cached_structure_ptr", None) == ptr and hasattr(self, "_cached_hyperedge_dict"):
            return

        # Build edge_index once
        if H.dim() == 2 and H.size(0) == 2:
            edge_index = H.long().to(self.device)
        else:
            incidence = H
            if incidence.dim() != 2:
                raise ValueError('Unsupported hypergraph structure format for HyperSAGE.')
            # Align to (E, N)
            if incidence.size(0) == 0 or incidence.size(1) == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            else:
                # self.features might be unset early; fall back to shape-based transpose
                if incidence.size(0) == incidence.size(1):
                    pass
                else:
                    # if rows equal num_nodes, transpose to (E,N)
                    # Cannot rely on self.features before fit; infer by min dimension
                    if incidence.size(0) > incidence.size(1):
                        incidence = incidence.t()
                mask = incidence != 0
                if mask.sum() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                else:
                    hyper_idx, node_idx = mask.nonzero(as_tuple=True)
                    edge_index = torch.stack([node_idx.long(), hyper_idx.long()]).to(self.device)

        # Build hyperedge_dict once (CPU numpy then back to torch tensors on device)
        data = SimpleNamespace(x=torch.empty(0, device=self.device), edge_index=edge_index)
        data = HyperSAGE.generate_hyperedge_dict(data)

        self._cached_edge_index = edge_index
        self._cached_hyperedge_dict = data.hyperedge_dict
        self._cached_structure_ptr = ptr

    def _prepare_data(self, x: torch.Tensor, H: Optional[torch.Tensor]) -> SimpleNamespace:
        # Ensure cached structure, then construct a lightweight data wrapper
        self._ensure_prepared_structure(H)
        data = SimpleNamespace(x=x, edge_index=self._cached_edge_index, hyperedge_dict=self._cached_hyperedge_dict)
        return data

    def _build_model_args(self, cfg: SimpleNamespace) -> SimpleNamespace:
        # Robustly resolve parameters with None-aware fallbacks
        All_num_layers = getattr(cfg, 'hypersage_layers', None)
        if All_num_layers is None:
            All_num_layers = getattr(cfg, 'alldeep_layers', 2)

        HyperSAGE_power = getattr(cfg, 'hypersage_power', 2.0)
        HyperSAGE_num_sample = getattr(cfg, 'hypersage_num_sample', 16)

        hidden = getattr(cfg, 'hypersage_hidden', None)
        if hidden is None:
            hidden = getattr(cfg, 'alldeep_hidden', None)
        if hidden is None:
            hidden = getattr(cfg, 'hidden', 32)

        args = SimpleNamespace(
            All_num_layers=All_num_layers,
            HyperSAGE_power=HyperSAGE_power,
            HyperSAGE_num_sample=HyperSAGE_num_sample,
            MLP_hidden=hidden,
            dropout=self.dropout,
        )
        return args

def signal_shift_graph2(hypergraph,edge_count,H):
    new_signal = H.clone()
    for edge,nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = nodes[nodes!=node_i]
            new_signal[node_i] = new_signal[node_i] + torch.sum(H[neighbor_nodes], dim=0)

    H = new_signal/(edge_count+1)
    return H

def signal_shift_hypergraph2(hypergraph,H):
    #new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()
    for edge,nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes!=node_i]).to(dtype=torch.long)
            node_i = node_i.to(dtype=torch.long)
            new_signal[node_i] = new_signal[node_i] + torch.sum(H[neighbor_nodes], dim=0)/(len(nodes)-1)
    return H



def signal_shift_hypergraph_power(hypergraph,H, power):
    min_value, max_value = 1e-7, 1e1
    H = torch.clamp(H, min_value, max_value)
    #new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()
    for edge,nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes!=node_i]).to(dtype=torch.long)
            node_i = node_i.to(dtype=torch.long)
            new_signal[node_i] = new_signal[node_i] + torch.pow(torch.sum(torch.pow(H[neighbor_nodes],power), dim=0)/(len(nodes)-1),1/power)

    return normalize(new_signal)

def signal_shift_hypergraph_sample(hypergraph,H, power, num_sample):
    min_value, max_value = 1e-7, 1e1
    H = torch.clamp(H, min_value, max_value)
    #new_signal = torch.zeros(H.shape[0],H.shape[1]).cuda()
    new_signal = H.clone()

    for edge,nodes in hypergraph.items():
        for node_i in nodes:
            neighbor_nodes = (nodes[nodes!=node_i]).to(dtype=torch.long)
            node_i = node_i.to(dtype=torch.long)        
            if (len(neighbor_nodes)>num_sample):
                shuffled_neighborhood = H[neighbor_nodes][torch.randperm(H[neighbor_nodes].size()[0])]
                new_signal[node_i] = new_signal[node_i] + torch.pow(torch.sum(torch.pow(shuffled_neighborhood[0:num_sample],power), dim=0)/(len(nodes)-1),1/power)
            else:
                new_signal[node_i] = new_signal[node_i] + torch.pow(torch.sum(torch.pow(H[neighbor_nodes],power), dim=0)/(len(nodes)-1),1/power)
    return normalize(new_signal)

def normalize(mx):
    """Row-normalize tensor matrix"""
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    mx = mx * r_inv[:,None]
    return mx

def normalize_np(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def get_HyperGCN_He_dict(data):
    # Assume edge_index = [V;E], sorted
    edge_index = np.array(data.edge_index.cpu())
    """
    For each he, clique-expansion. Note that we allow the weighted edge.
    Note that if node pair (vi,vj) is contained in both he1, he2, we will have (vi,vj) twice in edge_index. (weighted version CE)
    We default no self loops so far.
    """
    # edge_index[1, :] = edge_index[1, :]-edge_index[1, :].min()
    He_dict = {}
    for he in np.unique(edge_index[1, :]):
        nodes_in_he = list(edge_index[0, :][edge_index[1, :] == he])
        He_dict[he.item()] = nodes_in_he

    return He_dict

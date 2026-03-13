import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from types import SimpleNamespace
from typing import Optional

from utils import accuracy
from models.AllDeepSets import SetGNN  # reuse SetGNN backbone with PMA path


class HyperGraphAllSetTransformer(nn.Module):
    """AllSetTransformer wrapper aligned to HyperGraph* interface.

    Reuses SetGNN with attention (PMA=True) and multi-heads, while keeping the
    same public methods: forward, get_h, fit, test. It also prepares edge_index
    from either sparse [2, E] indices or a dense incidence matrix.
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
        self.base_args = base_args

        self.features: Optional[torch.Tensor] = None
        self.structure: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

        self.model_args = self._build_model_args(nfeat, nhid, nclass, dropout, base_args)
        # Force attention path
        self.model_args.PMA = True
        self.model_args.GPR = False
        self.model_args.LearnMask = False
        self.model = SetGNN(self.model_args).to(device)

    def forward(
        self,
        x: torch.Tensor,
        H: Optional[torch.Tensor],
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._prepare_data(x.to(self.device), H, hyperedge_weight)
        logits, _, _ = self.model(data, return_emb=False)
        return F.log_softmax(logits, dim=1)

    def get_h(
        self,
        x: torch.Tensor,
        H: Optional[torch.Tensor],
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._prepare_data(x.to(self.device), H, hyperedge_weight)
        _, node_emb, _ = self.model(data, return_emb=True)
        return node_emb

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

    def _train_with_val(
        self,
        idx_train: torch.Tensor,
        idx_val: torch.Tensor,
        train_iters: int,
        verbose: bool,
    ) -> None:
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

    def _prepare_data(
        self,
        x: torch.Tensor,
        H: Optional[torch.Tensor],
        hyperedge_weight: Optional[torch.Tensor],
    ) -> SimpleNamespace:
        if H is None or H.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            norm = torch.empty(0, device=self.device, dtype=x.dtype)
        else:
            H = H.to(self.device)
            if H.dim() == 2 and H.size(0) == 2:
                edge_index = H.long()
            else:
                if H.dim() == 2:
                    incidence = H
                else:
                    raise ValueError('Unsupported hypergraph structure format for AllSetTransformer.')
                if incidence.size(0) == x.size(0) and incidence.size(1) != x.size(0):
                    incidence = incidence.t()
                mask = incidence != 0
                if mask.sum() == 0:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
                else:
                    hyper_idx, node_idx = mask.nonzero(as_tuple=True)
                    edge_index = torch.stack([node_idx.long(), hyper_idx.long()]).to(self.device)
            if edge_index.numel() == 0:
                norm = torch.empty(0, device=self.device, dtype=x.dtype)
            else:
                he_ids = edge_index[1]
                num_hyperedges = int(he_ids.max().item()) + 1
                he_sizes = torch.bincount(he_ids, minlength=num_hyperedges).to(x.dtype)
                he_sizes = he_sizes.clamp_min(1.0)
                if hyperedge_weight is None or hyperedge_weight.numel() == 0:
                    he_weight = torch.ones(num_hyperedges, device=self.device, dtype=x.dtype)
                else:
                    he_weight = hyperedge_weight.to(self.device, dtype=x.dtype)
                    if he_weight.numel() < num_hyperedges:
                        padding = torch.ones(num_hyperedges - he_weight.numel(), device=self.device, dtype=x.dtype)
                        he_weight = torch.cat([he_weight, padding], dim=0)
                norm = he_weight[he_ids] / he_sizes[he_ids]
        return SimpleNamespace(x=x, edge_index=edge_index, norm=norm)

    def _build_model_args(self, nfeat: int, nhid: int, nclass: int, dropout: float, cfg: Optional[SimpleNamespace]) -> SimpleNamespace:
        cfg = cfg or SimpleNamespace()
        hidden = getattr(cfg, 'allset_hidden', None)
        if hidden is None:
            hidden = nhid
        classifier_hidden = getattr(cfg, 'allset_classifier_hidden', None)
        if classifier_hidden is None:
            classifier_hidden = hidden
        return SimpleNamespace(
            num_features=nfeat,
            num_classes=nclass,
            dropout=dropout,
            All_num_layers=getattr(cfg, 'allset_layers', 2),
            MLP_hidden=hidden,
            MLP_num_layers=getattr(cfg, 'allset_mlp_layers', 2),
            Classifier_hidden=classifier_hidden,
            Classifier_num_layers=getattr(cfg, 'allset_classifier_layers', 1),
            normalization=getattr(cfg, 'allset_norm', 'bn'),
            deepset_input_norm=getattr(cfg, 'allset_input_norm', False),
            heads=getattr(cfg, 'allset_heads', 4),
            aggregate=getattr(cfg, 'allset_aggr', 'add'),
            GPR=False,
            LearnMask=False,
            PMA=True,
        )


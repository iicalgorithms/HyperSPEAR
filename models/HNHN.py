import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, List
import importlib

def _import_dhg():
    """Lazily import `dhg` and its nn submodule to avoid importing optional
    dependencies at module import time.
    """
    try:
        dhg_mod = importlib.import_module("dhg")
        dhg_nn = importlib.import_module("dhg.nn")
        return dhg_mod, dhg_nn
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise ImportError(
            "HNHN requires the `dhg` package. Install via `pip install dhg` "
            "(note: some versions also require `optuna`)."
        ) from exc


class HNHN(nn.Module):
    r"""The HNHN backbone defined in `HNHN: Hypergraph Networks with Hyperedge Neurons` (ICML 2020).

    Args:
        in_channels (int): Dimension of input node features.
        hid_channels (int): Dimension of hidden representation.
        num_classes (int): Number of output classes.
        num_layers (int, optional): Number of HNHNConv layers. Default: 2.
        drop_rate (float, optional): Dropout ratio. Default: 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_layers: int = 2,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        # Lazy import
        _, dhg_nn = _import_dhg()
        HNHNConv = getattr(dhg_nn, "HNHNConv")

        self.layers = nn.ModuleList()
        self.inlinear = nn.Linear(in_channels, hid_channels)
        self.outlinear = nn.Linear(hid_channels, num_classes)

        nn.init.xavier_uniform_(self.inlinear.weight)
        nn.init.xavier_uniform_(self.outlinear.weight)

        self.layers = nn.ModuleList(
            [
                HNHNConv(
                    in_channels=hid_channels,
                    out_channels=hid_channels,
                    drop_rate=drop_rate,
                )
                for _ in range(max(num_layers, 1))
            ]
        )
        self.act = nn.LeakyReLU()

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph", return_emb: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        X = self.inlinear(X)
        for layer in self.layers:
            X = layer(X, hg)
        logits = self.outlinear(X)
        if return_emb:
            return logits, X, None
        return logits, None, None


def _to_incidence_matrix(
    H: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if H is None:
        return torch.zeros((0, num_nodes), device=device, dtype=dtype)
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

    incidence = H.to(device=device, dtype=dtype)
    if incidence.dim() != 2:
        raise ValueError("Hypergraph incidence must be a 2-D tensor.")
    if incidence.size(1) == num_nodes:
        return incidence
    if incidence.size(0) == num_nodes:
        return incidence.t()
    raise ValueError("Unable to align hypergraph incidence with given node count.")


def _incidence_to_edge_list(incidence: torch.Tensor) -> Tuple[List[List[int]], List[int]]:
    bool_incidence = incidence > 0
    e_list: List[List[int]] = []
    valid_ids: List[int] = []
    for idx in range(bool_incidence.size(0)):
        nodes = torch.nonzero(bool_incidence[idx], as_tuple=False).flatten()
        if nodes.numel() == 0:
            continue
        e_list.append(nodes.cpu().tolist())
        valid_ids.append(idx)
    return e_list, valid_ids


def _build_hypergraph(
    H: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> "dhg.Hypergraph":
    # Lazy import
    dhg, _ = _import_dhg()
    incidence = _to_incidence_matrix(H, num_nodes, device=torch.device("cpu"), dtype=torch.float32)
    e_list, _ = _incidence_to_edge_list(incidence)
    hypergraph = dhg.Hypergraph(num_v=num_nodes, e_list=e_list)
    return hypergraph.to(device)


class HyperGraphHNHN(nn.Module):
    """Wrapper providing the same interface as HyperGraphGCN but using HNHN as backbone."""

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        device: torch.device,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.device = device
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = HNHN(
            in_channels=nfeat,
            hid_channels=nhid,
            num_classes=nclass,
            num_layers=num_layers,
            drop_rate=dropout,
        ).to(device)
        self.features: Optional[torch.Tensor] = None
        self.structure: Optional[torch.Tensor] = None
        self.edge_weight: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.hypergraph: Optional["dhg.Hypergraph"] = None

    def _ensure_hypergraph(self, H: Optional[torch.Tensor], num_nodes: int) -> Optional["dhg.Hypergraph"]:
        if H is None:
            self.hypergraph = None
            return None
        if self.hypergraph is None or self.structure is None or self.structure.data_ptr() != H.data_ptr():
            self.hypergraph = _build_hypergraph(H.detach().cpu(), num_nodes, self.device)
            self.structure = H
        return self.hypergraph

    def forward(
        self,
        x: torch.Tensor,
        H: Optional[torch.Tensor],
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        hg = self._ensure_hypergraph(H, x.size(0))
        logits, _, _ = self.model(x, hg, return_emb=False)
        return F.log_softmax(logits, dim=1)

    def get_h(
        self,
        x: torch.Tensor,
        H: Optional[torch.Tensor],
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x.to(self.device)
        hg = self._ensure_hypergraph(H, x.size(0))
        _, embeddings, _ = self.model(x, hg, return_emb=True)
        return embeddings

    def fit(
        self,
        features: torch.Tensor,
        structure: torch.Tensor,
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
        structure: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        labels: torch.Tensor,
        idx_test: torch.Tensor,
    ) -> float:
        self.eval()
        with torch.no_grad():
            out = self.forward(
                features.to(self.device),
                structure.to(self.device) if structure is not None else None,
                edge_weight.to(self.device) if edge_weight is not None else None,
            )
        acc_test = accuracy(out[idx_test.to(self.device)], labels[idx_test].to(self.device))
        return float(acc_test)


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if not hasattr(labels, "__len__"):
        labels = torch.tensor([labels], device=output.device)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, device=output.device)
    labels = labels.long()
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    return correct / len(labels)

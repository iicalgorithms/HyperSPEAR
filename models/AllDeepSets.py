import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from types import SimpleNamespace
from typing import Optional

from utils import accuracy
from torch_geometric.nn import MessagePassing

class SetGNN(nn.Module):
    def __init__(self, args, norm=None):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """
#         V_in_dim = V_dict['in_dim']
#         V_enc_hid_dim = V_dict['enc_hid_dim']
#         V_dec_hid_dim = V_dict['dec_hid_dim']
#         V_out_dim = V_dict['out_dim']
#         V_enc_num_layers = V_dict['enc_num_layers']
#         V_dec_num_layers = V_dict['dec_num_layers']

#         E_in_dim = E_dict['in_dim']
#         E_enc_hid_dim = E_dict['enc_hid_dim']
#         E_dec_hid_dim = E_dict['dec_hid_dim']
#         E_out_dim = E_dict['out_dim']
#         E_enc_num_layers = E_dict['enc_num_layers']
#         E_dec_num_layers = E_dict['dec_num_layers']

#         Now set all dropout the same, but can be different
        self.All_num_layers = args.All_num_layers
        self.dropout = args.dropout
        # Respect external configuration (e.g., AllSetTransformer sets PMA=True).
        # Only set default when not provided.
        if not hasattr(args, 'PMA'):
            args.PMA = False
        if not hasattr(args, 'heads'):
            args.heads = 1
        self.aggr = 'add'
        self.NormLayer = args.normalization
        self.InputNorm = args.deepset_input_norm
        self.GPR = args.GPR
        self.LearnMask = args.LearnMask
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.V2EConvs = nn.ModuleList()
        self.E2VConvs = nn.ModuleList()
        self.bnV2Es = nn.ModuleList()
        self.bnE2Vs = nn.ModuleList()

        if self.LearnMask:
            self.Importance = nn.Parameter(torch.ones(norm.size()))

        if self.All_num_layers == 0:
            self.classifier = MLP(in_channels=args.num_features,
                                  hidden_channels=args.Classifier_hidden,
                                  out_channels=args.num_classes,
                                  num_layers=args.Classifier_num_layers,
                                  dropout=self.dropout,
                                  Normalization=self.NormLayer,
                                  InputNorm=False)
        else:
            self.V2EConvs.append(HalfNLHconv(in_dim=args.num_features,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA,
                                             aggr=args.aggregate))
            self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
            self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                             hid_dim=args.MLP_hidden,
                                             out_dim=args.MLP_hidden,
                                             num_layers=args.MLP_num_layers,
                                             dropout=self.dropout,
                                             Normalization=self.NormLayer,
                                             InputNorm=self.InputNorm,
                                             heads=args.heads,
                                             attention=args.PMA,
                                             aggr=args.aggregate))
            self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            for _ in range(self.All_num_layers-1):
                self.V2EConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA,
                                                 aggr=args.aggregate))
                self.bnV2Es.append(nn.BatchNorm1d(args.MLP_hidden))
                self.E2VConvs.append(HalfNLHconv(in_dim=args.MLP_hidden,
                                                 hid_dim=args.MLP_hidden,
                                                 out_dim=args.MLP_hidden,
                                                 num_layers=args.MLP_num_layers,
                                                 dropout=self.dropout,
                                                 Normalization=self.NormLayer,
                                                 InputNorm=self.InputNorm,
                                                 heads=args.heads,
                                                 attention=args.PMA,
                                                 aggr=args.aggregate))
                self.bnE2Vs.append(nn.BatchNorm1d(args.MLP_hidden))
            if self.GPR:
                self.MLP = MLP(in_channels=args.num_features,
                               hidden_channels=args.MLP_hidden,
                               out_channels=args.MLP_hidden,
                               num_layers=args.MLP_num_layers,
                               dropout=self.dropout,
                               Normalization=self.NormLayer,
                               InputNorm=False)
                self.GPRweights = nn.Linear(self.All_num_layers+1, 1, bias=False)
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)
            else:
                self.classifier = MLP(in_channels=args.MLP_hidden,
                                      hidden_channels=args.Classifier_hidden,
                                      out_channels=args.num_classes,
                                      num_layers=args.Classifier_num_layers,
                                      dropout=self.dropout,
                                      Normalization=self.NormLayer,
                                      InputNorm=False)


#         Now we simply use V_enc_hid=V_dec_hid=E_enc_hid=E_dec_hid
#         However, in general this can be arbitrary.


    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        if self.GPR:
            self.MLP.reset_parameters()
            self.GPRweights.reset_parameters()
        if self.LearnMask:
            nn.init.ones_(self.Importance)

    def forward(self, data, return_emb=False):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
#             The data should contain the follows
#             data.x: node features
#             data.V2Eedge_index:  edge list (of size (2,|E|)) where
#             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges
#      
        x, edge_index, norm = data.x, data.edge_index, data.norm
        if self.LearnMask:
            norm = self.Importance*norm
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        if self.GPR:
            xs = []
            xs.append(F.relu(self.MLP(x)))
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.E2VConvs[i](x, reversed_edge_index, norm, self.aggr)
                x = F.relu(x)
                xs.append(x)
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = torch.stack(xs, dim=-1)
            x = self.GPRweights(x).squeeze()
            outs = self.classifier(x)
        else:
            x = F.dropout(x, p=0.2, training=self.training) # Input dropout
            for i, _ in enumerate(self.V2EConvs):
                x = F.relu(self.V2EConvs[i](x, edge_index, norm, self.aggr))
#                 x = self.bnV2Es[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = F.relu(self.E2VConvs[i](
                    x, reversed_edge_index, norm, self.aggr))
#                 x = self.bnE2Vs[i](x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            outs = self.classifier(x)

        node_emb = x if return_emb else None
        return outs, node_emb, None


class HyperGraphAllDeepSets(nn.Module):
    """Wrapper to align SetGNN with the project training interface."""

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
        # Force non-attention path for AllDeepSets to distinguish from AllSetTransformer
        self.model_args.PMA = False
        if not hasattr(self.model_args, 'heads'):
            self.model_args.heads = 1
        if getattr(self.model_args, 'LearnMask', False):
            raise ValueError("Learnable mask is not supported in this integration.")
        self.model = SetGNN(self.model_args).to(device)

    def forward(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._prepare_data(x.to(self.device), H, hyperedge_weight)
        logits, _, _ = self.model(data, return_emb=False)
        return F.log_softmax(logits, dim=1)

    def get_h(
        self,
        x: torch.Tensor,
        H: torch.Tensor,
        hyperedge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        data = self._prepare_data(x.to(self.device), H, hyperedge_weight)
        _, node_emb, _ = self.model(data, return_emb=True)
        return node_emb

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
                    raise ValueError('Unsupported hypergraph structure format for AllDeepSets.')
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
        hidden = getattr(cfg, 'alldeep_hidden', None)
        if hidden is None:
            hidden = nhid
        classifier_hidden = getattr(cfg, 'alldeep_classifier_hidden', None)
        if classifier_hidden is None:
            classifier_hidden = hidden
        return SimpleNamespace(
            num_features=nfeat,
            num_classes=nclass,
            dropout=dropout,
            All_num_layers=getattr(cfg, 'alldeep_layers', 2),
            MLP_hidden=hidden,
            MLP_num_layers=getattr(cfg, 'alldeep_mlp_layers', 2),
            Classifier_hidden=classifier_hidden,
            Classifier_num_layers=getattr(cfg, 'alldeep_classifier_layers', 1),
            normalization=getattr(cfg, 'alldeep_norm', 'bn'),
            deepset_input_norm=getattr(cfg, 'alldeep_input_norm', False),
            heads=getattr(cfg, 'alldeep_heads', 1),
            aggregate=getattr(cfg, 'alldeep_aggr', 'add'),
            GPR=getattr(cfg, 'alldeep_gpr', False),
            LearnMask=getattr(cfg, 'alldeep_learn_mask', False),
        )
    

class HalfNLHconv(MessagePassing):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 Normalization='bn',
                 InputNorm=False,
                 heads=1,
                 attention=True,
                 aggr='mean'
                 ):
        super(HalfNLHconv, self).__init__(aggr=aggr)

        self.attention = attention
        self.dropout = dropout
        self.aggr = aggr

        if self.attention:
            self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        else:
            if num_layers > 0:
                self.f_enc = MLP(in_dim, hid_dim, hid_dim, num_layers, dropout, Normalization, InputNorm)
                self.f_dec = MLP(hid_dim, hid_dim, out_dim, num_layers, dropout, Normalization, InputNorm)
            else:
                self.f_enc = nn.Identity()
                self.f_dec = nn.Identity()
#         self.bn = nn.BatchNorm1d(dec_hid_dim)
#         self.dropout = dropout
#         self.Prop = S2SProp()

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ == 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ == 'Identity'):
                self.f_dec.reset_parameters()
#         self.bn.reset_parameters()

    def forward(self, x, edge_index, norm, aggr='add'):
        """
        input -> MLP -> Prop
        """
        
        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            x = F.relu(self.f_dec(x))
            
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index,
                  dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

from torch_geometric.typing import Adj, Size, OptTensor
from torch import Tensor


class PMA(MessagePassing):
    """
        PMA part:
        Note that in original PMA, we need to compute the inner product of the seed and neighbor nodes.
        i.e. e_ij = a(Wh_i,Wh_j), where a should be the inner product, h_i is the seed and h_j are neightbor nodes.
        In GAT, a(x,y) = a^T[x||y]. We use the same logic.
    """
    _alpha: OptTensor

    def __init__(self, in_channels, hid_dim,
                 out_channels, num_layers, heads=1, concat=True,
                 negative_slope=0.2, dropout=0.0, bias=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(PMA, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.hidden = hid_dim // heads
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.aggr = kwargs.get('aggr', 'add')
#         self.input_seed = input_seed

#         This is the encoder part. Where we use 1 layer NN (Theta*x_i in the GATConv description)
#         Now, no seed as input. Directly learn the importance weights alpha_ij.
#         self.lin_O = Linear(heads*self.hidden, self.hidden) # For heads combining
        # For neighbor nodes (source side, key)
        self.lin_K = nn.Linear(in_channels, self.heads*self.hidden)
        # For neighbor nodes (source side, value)
        self.lin_V = nn.Linear(in_channels, self.heads*self.hidden)
        self.att_r = nn.Parameter(torch.Tensor(
            1, heads, self.hidden))  # Seed vector
        self.rFF = MLP(in_channels=self.heads*self.hidden,
                       hidden_channels=self.heads*self.hidden,
                       out_channels=out_channels,
                       num_layers=num_layers,
                       dropout=.0, Normalization='None',)
        self.ln0 = nn.LayerNorm(self.heads*self.hidden)
        self.ln1 = nn.LayerNorm(self.heads*self.hidden)
#         if bias and concat:
#             self.bias = Parameter(torch.Tensor(heads * out_channels))
#         elif bias and not concat:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:

#         Always no bias! (For now)
        self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        #         glorot(self.lin_l.weight)
        glorot(self.lin_K.weight)
        glorot(self.lin_V.weight)
        self.rFF.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
#         glorot(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
#         zeros(self.bias)

    def forward(self, x, edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.hidden

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_K = self.lin_K(x).view(-1, H, C)
            x_V = self.lin_V(x).view(-1, H, C)
            alpha_r = (x_K * self.att_r).sum(dim=-1)
#         else:
#             x_l, x_r = x[0], x[1]
#             assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
#             x_l = self.lin_l(x_l).view(-1, H, C)
#             alpha_l = (x_l * self.att_l).sum(dim=-1)
#             if x_r is not None:
#                 x_r = self.lin_r(x_r).view(-1, H, C)
#                 alpha_r = (x_r * self.att_r).sum(dim=-1)

#         assert x_l is not None
#         assert alpha_l is not None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
#         ipdb.set_trace()
        out = self.propagate(edge_index, x=x_V,
                             alpha=alpha_r, aggr=self.aggr)

        alpha = self._alpha
        self._alpha = None

#         Note that in the original code of GMT paper, they do not use additional W^O to combine heads.
#         This is because O = softmax(QK^T)V and V = V_in*W^V. So W^O can be effectively taken care by W^V!!!
        out += self.att_r  # This is Seed + Multihead
        # concat heads then LayerNorm. Z (rhs of Eq(7)) in GMT paper.
        out = self.ln0(out.view(-1, self.heads * self.hidden))
        # rFF and skip connection. Lhs of eq(7) in GMT paper.
        out = self.ln1(out+F.relu(self.rFF(out)))

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j, alpha_j,
                index, ptr,
                size_j):
        #         ipdb.set_trace()
        alpha = alpha_j
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, index.max()+1)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def aggregate(self, inputs, index,
                  dim_size=None, aggr=None):
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """
#         ipdb.set_trace()
        if aggr is None:
            # 如果没有传递aggr参数，则使用self.aggr作为默认值
            aggr = self.aggr
        return scatter(inputs, index, dim=self.node_dim, reduce=aggr)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
    
import math

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

from torch import Tensor

from torch_geometric.utils import scatter, segment
from torch_geometric.utils.num_nodes import maybe_num_nodes

def softmax(
    src: Tensor,
    index: Optional[Tensor] = None,
    ptr: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 0,
) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`

    Examples:

        >>> src = torch.tensor([1., 1., 1., 1.])
        >>> index = torch.tensor([0, 0, 1, 2])
        >>> ptr = torch.tensor([0, 2, 3, 4])
        >>> softmax(src, index)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> softmax(src, None, ptr)
        tensor([0.5000, 0.5000, 1.0000, 1.0000])

        >>> src = torch.randn(4, 4)
        >>> ptr = torch.tensor([0, 4])
        >>> softmax(src, index, dim=-1)
        tensor([[0.7404, 0.2596, 1.0000, 1.0000],
                [0.1702, 0.8298, 1.0000, 1.0000],
                [0.7607, 0.2393, 1.0000, 1.0000],
                [0.8062, 0.1938, 1.0000, 1.0000]])
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        count = ptr[1:] - ptr[:-1]
        ptr = ptr.view(size)
        src_max = segment(src.detach(), ptr, reduce='max')
        src_max = src_max.repeat_interleave(count, dim=dim)
        out = (src - src_max).exp()
        out_sum = segment(out, ptr, reduce='sum') + 1e-16
        out_sum = out_sum.repeat_interleave(count, dim=dim)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src.detach(), index, dim, dim_size=N, reduce='max')
        out = src - src_max.index_select(dim, index)
        out = out.exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum') + 1e-16
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / out_sum

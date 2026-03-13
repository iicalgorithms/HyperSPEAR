# %%
from models.MLP import MLP
from models.HyperGCN import HyperGraphGCN
from models.HGNN import HyperGraphHGNN
from models.HNHN import HyperGraphHNHN
from models.AllDeepSets import HyperGraphAllDeepSets
from models.hypersage import HyperGraphHyperSAGE
from models.AllSetTransformer import HyperGraphAllSetTransformer

def model_construct(args, model_name, data, device, add_selfloop=True):
    features = data.X if hasattr(data, 'X') else data.x
    labels = data.Y if hasattr(data, 'Y') else data.y
    is_hypergraph = hasattr(data, 'H') and data.H is not None

    nfeat = features.shape[1]
    nclass = int(labels.max().item() + 1)

    if is_hypergraph:
        if model_name == 'HyperGCN':
            model = HyperGraphGCN(
                nfeat=nfeat,
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                device=device,
            )
        elif model_name == 'HGNN':
            model = HyperGraphHGNN(
                nfeat=nfeat,
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                device=device,
                lr=args.train_lr,
                weight_decay=args.weight_decay,
                num_layers=getattr(args, 'hgnn_layers', 2),
            )
        elif model_name == 'HNHN':
            model = HyperGraphHNHN(
                nfeat=nfeat,
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                device=device,
                lr=args.train_lr,
                weight_decay=args.weight_decay,
                num_layers=getattr(args, 'hgnn_layers', 2),
            )
        elif model_name == 'AllDeepSets':
            alldeep_hidden = getattr(args, 'alldeep_hidden', None)
            default_hidden = args.hidden if hasattr(args, 'hidden') else nfeat
            model = HyperGraphAllDeepSets(
                nfeat=nfeat,
                nhid=alldeep_hidden if alldeep_hidden is not None else default_hidden,
                nclass=nclass,
                dropout=args.dropout,
                device=device,
                lr=args.train_lr,
                weight_decay=args.weight_decay,
                base_args=args,
            )
        elif model_name == 'AllSetTransformer':
            allset_hidden = getattr(args, 'allset_hidden', None)
            default_hidden = args.hidden if hasattr(args, 'hidden') else nfeat
            model = HyperGraphAllSetTransformer(
                nfeat=nfeat,
                nhid=allset_hidden if allset_hidden is not None else default_hidden,
                nclass=nclass,
                dropout=args.dropout,
                device=device,
                lr=args.train_lr,
                weight_decay=args.weight_decay,
                base_args=args,
            )
        elif model_name == 'HyperSAGE':
            hypersage_hidden = getattr(args, 'hypersage_hidden', None)
            default_hidden = args.hidden if hasattr(args, 'hidden') else nfeat
            model = HyperGraphHyperSAGE(
                nfeat=nfeat,
                nhid=hypersage_hidden if hypersage_hidden is not None else default_hidden,
                nclass=nclass,
                dropout=args.dropout,
                device=device,
                lr=args.train_lr,
                weight_decay=args.weight_decay,
                base_args=args,
            )
        elif model_name == 'MLP':
            model = MLP(
                nfeat=nfeat,
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                lr=args.train_lr,
                weight_decay=args.weight_decay,
                device=device,
            )
        else:
            raise NotImplementedError(f"Model '{model_name}' is not supported for hypergraph data.")
        return model

    # if model_name == 'GCN':
    #     model = GCN(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         dropout=args.dropout,
    #         lr=args.train_lr,
    #         weight_decay=args.weight_decay,
    #         device=device,
    #         use_ln=use_ln,
    #         layer_norm_first=layer_norm_first,
    #         add_selfloop=add_selfloop,
    #     )
    # elif model_name == 'GAT':
    #     model = GAT(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         heads=8,
    #         dropout=args.dropout,
    #         lr=args.train_lr,
    #         weight_decay=args.weight_decay,
    #         device=device,
    #     )
    # elif model_name == 'GraphSage':
    #     model = GraphSage(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         dropout=args.dropout,
    #         lr=args.train_lr,
    #         weight_decay=args.weight_decay,
    #         device=device,
    #     )
    # elif model_name == 'GCN_Encoder':
    #     model = GCN_Encoder(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         dropout=args.dropout,
    #         lr=args.train_lr,
    #         weight_decay=args.weight_decay,
    #         device=device,
    #         use_ln=use_ln,
    #         layer_norm_first=layer_norm_first,
    #     )
    # elif model_name == 'GNNGuard':
    #     model = GNNGuard(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         dropout=args.dropout,
    #         lr=args.train_lr,
    #         weight_decay=args.weight_decay,
    #         use_ln=use_ln,
    #         device=device,
    #     )
    # elif model_name == 'RobustGCN':
    #     model = RobustGCN(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         dropout=args.dropout,
    #         lr=args.train_lr,
    #         weight_decay=args.weight_decay,
    #         device=device,
    #     )
    # elif model_name == 'MLP':
    #     model = MLP(
    #         nfeat=nfeat,
    #         nhid=args.hidden,
    #         nclass=nclass,
    #         dropout=args.dropout,
    #         lr=args.train_lr,
    #         weight_decay=args.weight_decay,
    #         device=device,
    #     )
    # else:
    #     raise NotImplementedError(f"Model '{model_name}' is not implemented.")
    # return model




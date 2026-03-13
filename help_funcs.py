import numpy as np
import torch.nn.functional as F
import torch
import scipy.sparse as sp
from models.reconstruct import MLPAE

import torch
import torch.nn.functional as F

def prune_unrelated_hyperedge(args, edge_structure, edge_weights, x, device, large_graph=True):
    """超图剪枝：移除节点特征相似性低的超边"""
    edge_structure = edge_structure.to(device)
    x = x.to(device)
    # 统一 edge_weights，到设备并允许 None
    # 若为 None 或空，则稍后根据形状补为按关联条目的全 1
    # 注意：暂不直接 .to(device) 以避免 None 引发异常
    
    # 只保留权重大于0 的超边
    # 只保留权重大于 0 的关联条目
    # 在此之前，我们先推断并统一 edge_weights 形状
    node_indices = edge_structure[0]
    hyperedge_indices = edge_structure[1]
    num_incidences = edge_structure.size(1)
    num_hyperedges = int(hyperedge_indices.max().item()) + 1 if hyperedge_indices.numel() > 0 else 0

    if edge_weights is None or (hasattr(edge_weights, 'numel') and edge_weights.numel() == 0):
        edge_weights = torch.ones(num_incidences, device=device, dtype=torch.float)
    else:
        edge_weights = edge_weights.to(device)
        if edge_weights.numel() == num_incidences:
            pass
        elif edge_weights.numel() == num_hyperedges:
            edge_weights = edge_weights[hyperedge_indices]
        else:
            raise ValueError("edge_weights 长度需等于关联条目数或超边数")

    hyperedge_mask = edge_weights > 0
    edge_structure = edge_structure[:, hyperedge_mask]
    edge_weights = edge_weights[hyperedge_mask]

    # 从超边结构中获取节点索引和超边索引
    node_indices = edge_structure[0]
    hyperedge_indices = edge_structure[1]
    # 计算超边总数
    num_hyperedges = int(hyperedge_indices.max().item()) + 1 if hyperedge_indices.numel() > 0 else 0

    # 初始化一个布尔掩码，默认所有超边都保留
    keep_mask = torch.ones(num_hyperedges, device=device, dtype=torch.bool)

    # 建立一个列表，记录每条超边包含的节点
    hyperedge_nodes = [[] for _ in range(num_hyperedges)]
    for idx in range(hyperedge_indices.numel()):
        he = int(hyperedge_indices[idx])
        node = int(node_indices[idx])
        hyperedge_nodes[he].append(node)

    # 遍历每条超边，计算其中节点的特征相似度
    for he, nodes in enumerate(hyperedge_nodes):
        if len(nodes) < 2:
            continue  # 超边中少于2 个节点时跳过
        nodes = torch.tensor(nodes, device=device)
        node_feats = x[nodes]
        if large_graph:
            # 大图场景下：用超边内所有节点的中心向量（均值）计算相似度
            centroid = node_feats.mean(dim=0, keepdim=True)
            sims = F.cosine_similarity(node_feats, centroid, dim=1)
        else:
            # 小图场景下：逐对节点计算余弦相似度矩阵
            pairwise_matrix = F.normalize(node_feats, p=2, dim=1) @ F.normalize(node_feats, p=2, dim=1).T
            triu_indices = torch.triu_indices(len(nodes), len(nodes), offset=1)
            sims = pairwise_matrix[triu_indices[0], triu_indices[1]]
        # 如果该超边内部节点的平均相似度小于设定阈值，则标记为移除
        if sims.mean().item() < args.prune_thr:
            keep_mask[he] = False

    # 根据保留的超边更新结构和权重
    mask = keep_mask[hyperedge_indices]
    updated_edge_structure = edge_structure[:, mask]
    updated_edge_weights = edge_weights[mask]
    return updated_edge_structure, updated_edge_weights

def reconstruct_prune_unrelated_hyperedge(args, poison_structure, poison_weights, poison_x, ori_x, ori_structure, device, idx, large_graph=True):
    """超图重构：使用节点自编码器重构并根据误差过滤超边"""
    poison_x = poison_x.to(device)
    # 统一 poison_weights 到按关联条目的形状，允许 None
    he_idx = poison_structure[1].to(device)
    num_incidences = poison_structure.size(1)
    num_hyperedges = (int(he_idx.max().item()) + 1) if he_idx.numel() > 0 else 0
    if poison_weights is None or (hasattr(poison_weights, 'numel') and poison_weights.numel() == 0):
        poison_weights = torch.ones(num_incidences, device=device, dtype=torch.float)
    else:
        poison_weights = poison_weights.to(device)
        if poison_weights.numel() == num_incidences:
            pass
        elif poison_weights.numel() == num_hyperedges:
            poison_weights = poison_weights[he_idx]
        else:
            raise ValueError("poison_weights 长度需等于关联条目数或超边数")

    # 初始化自动编码器并进行训练
    AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE.fit()
    # 推理得到每个节点的重构误差分数
    rec_score_ori = AE.inference(poison_x)
    # 依据重构误差设置节点异常阈值
    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), 97)
    # 标记高误差节点
    mask = rec_score_ori > threshold

    node_indices = poison_structure[0].to(device)
    hyperedge_indices = poison_structure[1].to(device)
    # 初始化一个布尔张量，默认所有超边都保留
    keep_hyperedge = torch.ones(int(hyperedge_indices.max().item()) + 1 if hyperedge_indices.numel() > 0 else 0, device=device, dtype=torch.bool)

    # 遍历每条超边，若包含任何高误差节点则移除该超边
    for he in torch.unique(hyperedge_indices):
        he = int(he.item())
        nodes = node_indices[hyperedge_indices == he]
        if mask[nodes].any():
            keep_hyperedge[he] = False

    mask_edges = keep_hyperedge[hyperedge_indices]
    filtered_structure = poison_structure[:, mask_edges]
    filtered_weights = poison_weights[mask_edges]
    return filtered_structure, filtered_weights

def select_target_nodes(args, seed, model, features, edge_structure, edge_weights, labels, idx_val, idx_test):
    """Select target, clean, and poisoning nodes for hypergraph data."""
    device = features.device
    if edge_weights is None:
        num_edges = edge_structure.shape[1] if edge_structure.numel() > 0 else 0
        edge_weights = torch.ones(num_edges, device=device, dtype=torch.float)
    else:
        edge_weights = edge_weights.to(device)

    features = features.to(device)
    edge_structure = edge_structure.to(device)
    labels = labels.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    test_ca, test_correct_index = model.test_with_correct_nodes(features, edge_structure, edge_weights, labels, idx_test)
    test_correct_index = test_correct_index.tolist()
    test_correct_nodes = idx_test[test_correct_index].tolist()

    target_class_nodes_test = [int(nid) for nid in idx_test.tolist() if labels[nid] == args.target_class]

    idx_val_list = idx_val.tolist()
    idx_test_list = idx_test.tolist()
    rs = np.random.RandomState(seed)

    cand_atk_test_nodes = list(set(test_correct_nodes) - set(target_class_nodes_test))
    if len(cand_atk_test_nodes) < args.target_test_nodes_num:
        raise ValueError('Not enough candidate attack test nodes to sample from.')
    atk_test_nodes = rs.choice(cand_atk_test_nodes, args.target_test_nodes_num, replace=False)

    cand_clean_test_nodes = list(set(idx_test_list) - set(atk_test_nodes))
    if len(cand_clean_test_nodes) < args.clean_test_nodes_num:
        raise ValueError('Not enough candidate clean test nodes to sample from.')
    clean_test_nodes = rs.choice(cand_clean_test_nodes, args.clean_test_nodes_num, replace=False)

    N = features.shape[0]
    cand_poi_train_nodes = list(set(idx_val_list) - set(atk_test_nodes) - set(clean_test_nodes))
    poison_nodes_num = max(1, int(N * args.vs_ratio))
    if len(cand_poi_train_nodes) < poison_nodes_num:
        poison_nodes_num = len(cand_poi_train_nodes)
    poi_train_nodes = rs.choice(cand_poi_train_nodes, poison_nodes_num, replace=False)

    return atk_test_nodes, clean_test_nodes, poi_train_nodes

def normalize(mx):
    """对稀疏矩阵进行行归一化"""
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
def normalize_adj(adj):
    """对邻接矩阵进行对称归一化"""
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()








#%%
import torch
import numpy as np

def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.
    Parameters
    ----------
    labels : torch.LongTensor
        node labels
    Returns
    -------
    torch.LongTensor
        onehot labels tensor
    """
    labels = labels.long()
    eye = torch.eye(int(labels.max().item()) + 1, device=labels.device)
    onehot_mx = eye[labels]
    return onehot_mx

def accuracy(output, labels):
    """Return accuracy of output compared to labels.
    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels
    Returns
    -------
    float
        accuracy
    """
    if not hasattr(labels, '__len__'):
        labels = [labels]
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def idx_to_mask(indices, n):
    mask = torch.zeros(n, dtype=torch.bool)
    mask[indices] = True
    return mask
import scipy.sparse as sp
def sys_normalized_adjacency(adj):
   adj = sp.coo_matrix(adj)
   adj = adj + sp.eye(adj.shape[0])
   row_sum = np.array(adj.sum(1))
   row_sum=(row_sum==0)*1+row_sum
   d_inv_sqrt = np.power(row_sum, -0.5).flatten()
   d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
   d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
   return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


import torch
import pickle
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T

# 归一化节点特征
def normalize_features(X):
    if isinstance(X, torch.Tensor):
        row_sum = X.abs().sum(dim=1, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        return X / row_sum
    transform = T.Compose([T.NormalizeFeatures()])
    return transform(X)

# 创建超图数据对象
class HypergraphData:
    def __init__(self, H, x, y, train_mask, val_mask, test_mask):
        self.H = H  # 超边关联矩阵
        self.X = x  # 节点特征
        self.Y = y  # 节点标签
        self.train_mask = train_mask  # 训练集掩码
        self.val_mask = val_mask  # 验证集掩码
        self.test_mask = test_mask  # 测试集掩码

# 根据 dataset_name 加载对应的数据集
def load_hypergraph_data(dataset_name, root_dir='./data/'):

    # 加载 Cora 超图数据（假设文件名为 H_cora.pt 等）
    H = torch.load(f"{root_dir}/{dataset_name}/H.pt")
    x = torch.load(f"{root_dir}/{dataset_name}/X.pt")
    y = torch.load(f"{root_dir}/{dataset_name}/Y.pt")

    nNode = x.size(0)
    train_mask = torch.zeros(nNode, dtype=torch.bool)
    val_mask = torch.zeros(nNode, dtype=torch.bool)
    test_mask = torch.zeros(nNode, dtype=torch.bool)

    # 返回 HypergraphData 对象
    return HypergraphData(H, x, y, train_mask, val_mask, test_mask)

def get_split_hypergraph(args, data, device):
    rs = np.random.RandomState(10)
    
    # 获取节点总数
    num_nodes = data.x.shape[0]
    # 随机打乱节点顺序
    perm = rs.permutation(num_nodes)
    
    # 训练集划分：20%的节点用于训练
    train_number = int(0.2 * len(perm))
    idx_train = torch.tensor(sorted(perm[:train_number])).to(device)
    data.train_mask = torch.zeros_like(data.train_mask)
    data.train_mask[idx_train] = True
    
    # 验证集划分：10%的节点用于验证
    val_number = int(0.1 * len(perm))
    idx_val = torch.tensor(sorted(perm[train_number:train_number + val_number])).to(device)
    data.val_mask = torch.zeros_like(data.val_mask)
    data.val_mask[idx_val] = True
    
    # 测试集划分：20%的节点用于测试
    test_number = int(0.2 * len(perm))
    idx_test = torch.tensor(sorted(perm[train_number + val_number:train_number + val_number + test_number])).to(device)
    data.test_mask = torch.zeros_like(data.test_mask)
    data.test_mask[idx_test] = True
    
    # 划分清洗测试集和攻击测试集
    idx_clean_test = idx_test[:int(len(idx_test) // 2)]
    idx_atk = idx_test[int(len(idx_test) // 2):]
    
    return data, idx_train, idx_val, idx_clean_test, idx_atk

def subgraph_hypergraph(subset, H, relabel_nodes: bool = False):
    """Generate the induced sub-hypergraph containing nodes in `subset`."""
    node_indices = H[0].long()
    hyperedge_indices = H[1].long()

    if subset.dtype == torch.bool:
        subset_indices = subset.nonzero(as_tuple=False).flatten().long()
    else:
        subset_indices = subset.long()

    if subset_indices.numel() == 0:
        empty = H.new_empty((2, 0))
        return empty, empty.new_empty(0)

    mask = torch.isin(node_indices, subset_indices)
    selected_hyperedges = torch.unique(hyperedge_indices[mask])

    if selected_hyperedges.numel() == 0:
        empty = H.new_empty((2, 0))
        return empty, selected_hyperedges

    keep_mask = torch.isin(hyperedge_indices, selected_hyperedges)
    new_H = H[:, keep_mask]

    # if relabel_nodes:
    #     max_index = int(node_indices.max().item()) + 1 if node_indices.numel() else 0
    #     mapping = torch.full((max_index,), -1, dtype=torch.long, device=H.device)
    #     mapping[subset_indices] = torch.arange(subset_indices.numel(), device=H.device)
    #     new_H = torch.stack((mapping[new_H[0]], new_H[1]))

    return new_H, selected_hyperedges

def build_uniform_hyperedge_weights(H: torch.Tensor, device: torch.device) -> torch.Tensor:
    # 如果 H 张量没有元素（空超图），直接返回一个空张量
    if H.numel() == 0:
        return torch.empty(0, device=device)
    # 从 H[1] 中提取所有的超边 ID（H[0] 是节点索引，H[1] 是超边索引）
    he_ids = torch.unique(H[1]).long().to(device)
    # 如果没有任何超边 ID，返回一个空张量
    if he_ids.numel() == 0:
        return torch.empty(0, device=device)
    # 计算超边的总数
    num_hyperedges = int(he_ids.max().item()) + 1
    # 初始化一个大小为 [num_hyperedges] 的权重张量，初始值全为 0
    weights = torch.zeros(num_hyperedges, device=device)
    # 将实际存在的超边 ID 的权重设为 1.0
    weights[he_ids] = 1.0
    return weights
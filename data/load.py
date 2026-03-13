import torch

# 加载 .pt 文件
file_path = 'data/dblp_coauth/X.pt'  # 替换为你的文件路径
model = torch.load(file_path)

# 检查内容并输出维度
if isinstance(model, dict):
    # 如果模型是字典形式（如模型的 state_dict）
    for key, value in model.items():
        print(f"Key: {key}, Shape: {value.shape}")
elif isinstance(model, torch.nn.Module):
    # 如果是 PyTorch 模型（torch.nn.Module），检查每个层的参数
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")
else:
    # 如果是其他类型的数据（比如张量）
    print(f"Shape of the loaded data: {model.shape}")

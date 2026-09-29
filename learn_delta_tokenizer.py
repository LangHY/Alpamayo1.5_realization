import torch

xyz = torch.tensor([[
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [2.0, 1.0, 0.0]
    ]])
print(xyz.shape)

zero_fix = torch.zeros(1, 1, 3) # 创建全零张量
xyz = torch.cat([xyz, zero_fix], dim=1) # 在时间点维度拼接，其余维度要保持相同

print(xyz.shape)
print(xyz)

delta = xyz[:, 1:] - xyz[:, :-1]
print(delta.shape)

xyz_min = torch.tensor([-4.0, -4.0, -10.0])
xyz_max = torch.tensor([4.0, 4.0, 10.0])

print(xyz_min.shape)
delta_lenth = xyz_max - xyz_min
data_lenth = delta - xyz_min

print(delta_lenth)
print(data_lenth)

norm = data_lenth / delta_lenth
print(norm)

nums_bins = 1000

tokenid = (norm * nums_bins).round().long().clamp(min=0, max=999)

print(tokenid)


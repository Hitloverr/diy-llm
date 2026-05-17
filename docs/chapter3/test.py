import torch

# x = torch.ones((3,3)).triu()
# print(x)

x = torch.ones(4,8,16,32)
w = torch.ones(32, 2)
print( (x @ w).shape)

x = torch.ones(2,2,3)
y = torch.ones(2,2,3)
z = x @ y.transpose(-2,-1)
from jaxtyping import Float

x : Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2,3,4)
y : Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 3, 4)

from einops import einsum
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
print(z.shape)

# 指数代表了能代表的数范围；尾数代表了数的精度。
x = torch.tensor([
    [0,1,2,3],
    [4,5,6,7],
    [8,9,10,11],
    [12,13,14,15]
])
assert x.stride(0) == 4
assert x.stride(1) == 1

r, c = 1, 2
index = r * x.stride(0) + c * x.stride(1)

# if torch.cuda.is_available():
#     print(torch.cuda.device_count())
#     for i in range(torch.cuda.device_count()):
#         properties = torch.cuda.get_device_properties(i)

# memory_allocated = torch.cuda.memory_allocated()

# y = x.to("cuda:0")
# assert y.device == torch.device('cuda:0')

# z = torch.zeros(32,32, device= "cuda:0")

# new_memory_allocated = torch.cuda.memory_allocated()
# memory_used = new_memory_allocated - memory_allocated
# print(memory_used)


x = torch.tensor([1.,2,3])
w = torch.tensor([1.,1,1], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)

print(loss)
loss.backward()
assert loss.grad is None
assert pred_y.grad is None # 中间变量默认也不保存梯度，除非pred_y.retain_grad()
assert x.grad is None # 没有requires_grad
assert w.grad is not None

assert torch.equal(w.grad, torch.tensor([1,2,3]))
print('done')

print('--------------')

import torch.nn as nn
input_dim = 128
output_dim = 128
w = nn.Parameter(torch.randn(input_dim, output_dim))
import numpy as np
# 一种对输入维度 input_dim 不敏感的xaiver初始化.avoid 梯度爆炸
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

# 但正态分布的尾部是无界的，仍然存在产生极端值的可能性；用截断正态分布。将产生的随机数限制在一个合理的范围内。
w = nn.Parameter(nn.init.trunc_normal_(
    torch.empty(input_dim, output_dim), std = 1 / np.sqrt(input_dim),
    a = -3, b =-3
))

class Cruncher(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_features= dim, out_features=dim)
            for _ in range(num_layers)
        ])
        self.final = nn.Linear(dim, 1)

    def forward(self, x):
        B,D = x.size()
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        assert x.size() == torch.Size([B, 1])
        x = x.squeeze(-1)
        assert x.size() == torch.Size([B])
        return x



D = 64
num_layers = 2
model = Cruncher(dim=D, num_layers=num_layers)

param_sizes = sum(param.numel() for name, param in model.named_parameters())
print(param_sizes)

# model = model.to("cuda:0")
model = model.to("cpu")
B = 8
x = torch.randn(B, D)
print(type(x))
y = model(x)
assert y.size() == torch.Size([B])

# 如何管理随机性
"""
1. 参数初始化：模型权重通常从随机分布（正态分布）中采样
2. dropout
3. 数据排序：数据加载器通常会打乱数据顺序
4. 其他：数据增强，优化器中的动量。

"""
import torch
import numpy as np
import random
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

orig_data = np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.int32)
orig_data.tofile('data.npy')

data = np.memmap("data.npy", dtype=np.int32)
assert np.array_equal(orig_data, data)

def get_batch(data:np.array, batch_size:int, seq_len:int,device:str) -> torch.Tensor:
    start_indices = torch.randint(len(data) - seq_len,(batch_size,))
    assert start_indices.size() == torch.Size([batch_size])

    x = torch.tensor([ data[start:start + seq_len] for start in start_indices])
    assert x.size() == torch.Size([batch_size, seq_len])
    if torch.cuda.is_available():
        # 将cpu张量显示标记为“固定
        x = x.pin_memory()
    # 后台异步流水线执行
    """
    可以实现以下高效的流水线操作：

- 在 GPU 上处理当前批次的数据。
- 在 CPU 上同时加载下一个批次的数据（例如，从硬盘读取或从内存映射文件中加载）。

这种并行化可以显著减少 GPU 的空闲等待时间，从而大幅提升整体训练吞吐量
    """
    x = x.to(device, non_blocking=True)
    return x

"""
① 分页内存（默认，普通内存）
操作系统自动管理，物理地址不固定
内存紧张时，系统会把数据临时挪到硬盘（虚拟内存）
GPU 无法直接访问这种内存

② 固定内存（Pinned Memory，锁页内存）
调用 .pin_memory() 后，物理地址钉死不变
系统绝对不会把它挪到硬盘
GPU 驱动可以直接读取这块内存

2. 数据传输流程对比（最核心！）
❌ 默认情况（分页内存）→ 慢、阻塞

CPU张量(分页内存)
  ↓ 【额外拷贝：操作系统强制复制】
临时固定内存
  ↓ 【PCIe总线】
GPU
多了一步内存拷贝，纯浪费时间

同步阻塞：CPU 必须等着拷贝完成，才能干别的事
✅ 固定内存（pin_memory）→ 快、无阻塞

CPU张量(固定内存)
  ↓ 【直接DMA传输，无拷贝】
GPU
跳过了中间拷贝步骤
硬件直接传输，CPU 不用参与，解放算力
"""
   
   
class AdaGrad(torch.optim.Optimizer):
    # staet:存储每个参数的优化器状态，对于adaGrad，就是历史梯度平方的累计和。
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                grad = p.grad.data
                state = self.state[p]
                if 'sum_squared_grad' not in state:
                    state['sum_squared_grad'] = torch.zeros_like(p.data)
                # 更新状态，累加梯度的平方
                state['sum_squared_grad'] += grad**2

                # 更新参数：除以根号下（状态）
                std = state['sum_squared_grad'].sqrt()
                p.data -= group['lr'] * grad / std

optimizer = AdaGrad(model.parameters(), lr = 0.01)
model.zero_grad(set_to_none=True)
y = model(x)
loss = y.mean()
loss.backward()
optimizer.step()

# 模型检查点要包含的内容：模型参数、优化器状态，比如Adam存储了动量、方差的移动平均值。
# AdaGrad：存储了历史梯度平方的累加和。
"""
如果只保存了模型参数而没有保存优化器状态，那么在恢复训练时，优化器会从零开始，这会导致训练过程不连续，性能下降，甚至可能无法收敛
"""
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
torch.save(checkpoint,"model_checkpoint.pth")
load_checkpoint = torch.load('model_checkpoint.pth')

model.load_state_dict(load_checkpoint["model"])
optimizer.load_state_dict(load_checkpoint['optimizer'])
"""
- 前向传播（Forward Pass）：使用 bfloat16 或 fp8。这包括所有中间激活值（activations）。因为激活值通常不需要极高的精度，使用低精度可以显著节省内存。
- 其余部分：使用 float32。这包括模型参数（parameters）、梯度（gradients）以及优化器状态（optimizer states）。这些是训练的核心，需要更高的精度来保证数值稳定性和收敛性。

> 核心思想：将低精度用于“消耗大但对精度要求不高”的部分（激活值），将高精度用于“对精度敏感”的部分（参数和梯度）。

"""
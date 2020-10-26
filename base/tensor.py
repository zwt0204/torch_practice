#!/user/bin/env python
# coding=utf-8
"""
@file: tensor.py
@author: zwt
@time: 2020-10-24 11:40
@desc: 
"""
from __future__ import print_function
import torch


# 没有初始化的矩阵
x = torch.empty(5, 3)
print(x)
# 随机初始化的矩阵
x = torch.rand(5, 3)
print(x)
# 指定类型，并初始化为0
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接指定数据生成矩阵
x = torch.tensor([5.5, 3])
print(x)
x = x.new_ones(5, 3, dtype=torch.float32)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)
print(x.size())

y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

print(x)
print(x[:, 1])

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

a = torch.ones(5)
print(a)

b = a.numpy()

a.add_(1)
print(a)
print(b)

# change from numpy

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# Tensors can be moved onto any device using the .to method.

if torch.cuda.is_available():
    device = torch.device('cuda')
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
else:
    y = torch.ones_like(x)
    print("===", y)


x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)  # y was created as a result of an operation, so it has a grad_fn.

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

# .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place.
# The input flag defaults to False if not given.

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
# print(a.requires_grad)
a.requires_grad_(True)
# print(a.requires_grad)
b = (a * a).sum()
# print(b.grad_fn)

print('=================')
out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

















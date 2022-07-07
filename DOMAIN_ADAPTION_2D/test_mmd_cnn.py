import torch
import numpy as np

a_source = np.array([[[1,2],[3,4]],[[1,2],[3,4]], [[1,2],[3,4]]])
a_source = np.swapaxes(a_source,0,1)
a_source = torch.from_numpy(a_source)
a_source = a_source.type(dtype=torch.float)
a_target = np.array([[[5,6],[7,8]],[[5,6],[7,8]], [[5,6],[7,8]]])
a_target = np.swapaxes(a_target,0,1)
a_target = torch.from_numpy(a_target)
a_target = a_target.type(dtype=torch.float)


b_source = torch.tensor([[1,2],[3,4]], dtype=torch.float)
b_target = torch.tensor([[5,6],[7,8]], dtype=torch.float)


b = torch.cat([b_source, b_target],axis=0)
a = torch.cat([a_source, a_target],axis=0)

b1_1 = b.unsqueeze(0)
b1_2 = b1_1.expand(int(b.size(0)), int(b.size(0)), int(b.size(1)))

b2_1 = b.unsqueeze(1)
b2_2 = b2_1.expand(int(b.size(0)), int(b.size(0)), int(b.size(1)))


b_total_1 = ((b1_2 - b2_2)**2)
b_total_2 = b_total_1.sum(2) 

kernel_val_b = torch.exp(-b_total_2 / 2)

XX = kernel_val_b[:2, :2]
YY = kernel_val_b[2:, 2:]
XY = kernel_val_b[:2, 2:]
YX = kernel_val_b[2:, :2]
loss_b = (XX + YY - XY -YX)
loss__b = torch.mean(loss_b)
print("§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§")

a1_1 = a.unsqueeze(0)
a1_2 = a1_1.expand(int(a.size(0)), int(a.size(0)), int(a.size(1)), int(a.size(2)))

a2_1 = a.unsqueeze(1)
a2_2 = a2_1.expand(int(a.size(0)), int(a.size(0)), int(a.size(1)), int(a.size(2)))

a_total_1 = (a1_2 - a2_2)**2
a_total_2 = a_total_1.sum(3)

kernel_val_a = torch.exp(-a_total_2 / 2) 
XX = kernel_val_a[:2, :2, :]
YY = kernel_val_a[2:, 2:, :]
XY = kernel_val_a[:2, 2:, 2:]
YX = kernel_val_a[2:,:2, :]
loss_a = (XX + YY - XY -YX)
loss__a = torch.mean(loss_a, dim = 0)
loss__a = torch.mean(loss__a, dim = 0)
loss__a = torch.sum(loss__a)

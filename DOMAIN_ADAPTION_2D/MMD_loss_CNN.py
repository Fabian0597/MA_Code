import torch
import torch.nn as nn

class MMD_loss_CNN(nn.Module):
    def __init__(self, fix_sigma = None, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss_CNN, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return
    def gaussian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)), int(total.size(2)))
        L2_distance = ((total0-total1)**2).sum(3) 
        kernel_val = [torch.exp(-L2_distance / sigma) for sigma in self.fix_sigma]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target)
        XX = kernels[:batch_size, :batch_size, :]
        YY = kernels[batch_size:, batch_size:, :]
        XY = kernels[:batch_size, batch_size:, :]
        YX = kernels[batch_size:, :batch_size, :]
        loss_1 = XX + YY - XY -YX
        loss_2 = torch.mean(loss_1, dim = 0)
        loss_3 = torch.mean(loss_2, dim = 0)
        loss_4 = torch.sum(loss_3)
        return loss_4
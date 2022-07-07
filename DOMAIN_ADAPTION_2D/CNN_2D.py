import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN_2D(nn.Module):
    def __init__(self, input_size, image_size, hidden_fc_size_1, random_seed):
        super(CNN_2D, self).__init__()
        
        self.input_size = input_size
        self.image_size = image_size
        self.input_fc_size = self.__get_input_fc_size()
        
        self.conv1 = nn.Conv2d(input_size, 64, kernel_size=30, stride=1)#input: ((64+2*0-(30-1)-1)/1)+1 = 35
        self.batch1 =nn.BatchNorm2d(64) #301
        self.conv2 = nn.Conv2d(64,32,kernel_size=15, stride = 1, padding=1)#input: ((35+2*1-(15-1)-1)/1)+1 = 23
        self.batch2 =nn.BatchNorm2d(32) #301
        self.conv3 = nn.Conv2d(32,32,kernel_size=10, stride = 1, padding=1) #((23+2*1-(10-1)-1)/1)+1 = 16
        self.batch3 =nn.BatchNorm2d(32) #299
        self.fc1 = nn.Linear(self.input_fc_size, hidden_fc_size_1)

        torch.manual_seed(random_seed)

    def forward(self, x):
        x_conv_1 = self.conv1(x) #conv1
        x = self.batch1(x_conv_1) #batch1
        x = F.relu(x) #relu
        x_conv_2 = self.conv2(x) #conv2
        x = self.batch2(x_conv_2) #batch1
        x = F.relu(x) #relu
        x_conv_3 = self.conv3(x) #conv3 
        x = self.batch3(x_conv_3) #batch2
        x = F.relu(x) #relu
        x_flatten = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])) #flatten
        x_normalize = torch.nn.functional.normalize(x_flatten)
        x_fc1 = self.fc1(x_normalize) #fc1
        
        return x_conv_1, x_conv_2, x_conv_3, x_flatten, x_fc1

    def __get_input_fc_size(self):
        out_dimension = ((self.image_size+2*0-(30-1)-1)/1)+1
        out_dimension = ((out_dimension+2*1-(15-1)-1)/1)+1
        out_dimension = ((out_dimension+2*1-(10-1)-1)/1)+1

        return int(out_dimension * out_dimension * 32)

import torch.nn as nn
import torch
torch.manual_seed(0)
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_size, hidden_fc_size_1, num_pool, window_size):
        super(CNN, self).__init__()
        
        self.num_pool = num_pool
        self.window_size = window_size
        self.input_fc_size = self.__get_input_fc_size()
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=100, stride=1)#input: ((1024+2*0-(100-1)-1)/1)+1 = 925
        self.pool1 = nn.MaxPool1d(4, stride=3) #((925+2*0-1*(4-1)-1)/3)+1 = 308
        self.conv2 = nn.Conv1d(64,32,kernel_size=10, stride = 1, padding=1)#input: ((308+2*1-(10-1)-1)/1)+1 = 301
        self.batch1 =nn.BatchNorm1d(32) #301
        self.pool2 = nn.MaxPool1d(4, stride=3) #((301+2*0-1*(4-1)-1)/3)+1 = 100
        self.conv3 = nn.Conv1d(32,32,kernel_size=5, stride = 1, padding=1) #((301+2*1-(5-1)-1)/1)+1 = 299
        self.batch2 =nn.BatchNorm1d(32) #299
        self.pool3 = nn.MaxPool1d(5, stride=3) #((98+2*0-1*(5-1)-1)/3)+1 = 3
        self.fc1 = nn.Linear(self.input_fc_size, hidden_fc_size_1)

    def forward(self, x):
        x_conv_1 = self.conv1(x) #conv1
        x = F.relu(x_conv_1) #relu
        if self.num_pool >=1:
            x = self.pool1(x) #pool1
        x_conv_2 = self.conv2(x) #conv2
        x = self.batch1(x_conv_2) #batch1
        x = F.relu(x) #relu
        if self.num_pool >=2:
            x = self.pool2(x) #pool2
        x_conv_3 = self.conv3(x) #conv3 
        x = self.batch2(x_conv_3) #batch2
        x = F.relu(x) #relu
        if self.num_pool >=3:
            x = self.pool3(x) #pool3
        x_flatten = torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2])) #flatten
        x_normalize = torch.nn.functional.normalize(x_flatten)
        x_fc1 = self.fc1(x_normalize) #fc1
        
        return x_conv_1, x_conv_2, x_conv_3, x_flatten, x_fc1

    def __get_input_fc_size(self):
        out_dimension = ((self.window_size+2*0-(100-1)-1)/1)+1
        if self.num_pool >=1:   
            out_dimension = ((out_dimension+2*0-1*(4-1)-1)/3)+1
        out_dimension = ((out_dimension+2*1-(10-1)-1)/1)+1
        if self.num_pool >=2:
            out_dimension = ((out_dimension+2*0-1*(4-1)-1)/3)+1
        out_dimension = ((out_dimension+2*1-(5-1)-1)/1)+1
        if self.num_pool >=3:
            out_dimension = ((out_dimension+2*0-1*(5-1)-1)/3)+1

        return int(out_dimension * 32)
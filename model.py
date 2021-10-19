"""
Model class for Aircraft CNN
Merwan Yeditha and Rohith Tatineti
"""
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchviz import make_dot
import torch
import torch.nn
import torch.optim as optim
import torchvision

# Dataset params
classes = [0, 1]
image_dims = 3, 20, 20
n_training_samples = 1000
n_test_samples = 500

class AirplaneCNN(nn.Module):
    def __init__(self):        
        super(AirplaneCNN, self).__init__()

        num_kernels_conv1 = 32
        num_kernels_conv2 = 12
        self.conv1 = nn.Conv2d(3, num_kernels_conv1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, num_kernels_conv2, kernel_size=3, stride=1, padding=1)
        self.pool_conv1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool_output_size_conv1 = int(num_kernels_conv1 * 100) #int(num_kernels * (image_dims[1] / stride) * (image_dims[2] / stride))
        self.pool_conv2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.maxpool_output_size_conv2 = int(num_kernels_conv2 * 25)

        fc1_size = 64
        self.fc1 = nn.Linear(self.maxpool_output_size_conv2, fc1_size)

        self.activation_func = torch.nn.Sigmoid()

        fc2_size = len(classes)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool_conv1(x)
        x = self.activation_func(x)
        x = self.conv2(x)
        x = self.pool_conv2(x)
        x = self.activation_func(x)
        x = x.view(-1, self.maxpool_output_size_conv2)
        x = self.fc1(x)
        x = self.activation_func(x)
        x = self.fc2(x)
        return x

    def get_loss(self, learning_rate):
      loss = nn.CrossEntropyLoss()
      optimizer = optim.Adam(self.parameters(), lr=learning_rate)
      return loss, optimizer
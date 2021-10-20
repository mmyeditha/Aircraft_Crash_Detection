import os
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn
import pandas as pd
import cv2

class AirplaneCrashDataset(Dataset):
    """Airplane Crash Dataset"""

    def __init__(self, labels, split_set='train', transform=None):
        """
        Args:
            labels (dict): dictionary object containing image name-label pairs
            root_dir (string): Directory with all the images.
        """
        self.labels = labels
        self.split = split_set
        self.root_dir = f'{split_set}_set'

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # find image in train_set data
        img_name = os.listdir(self.root_dir)[idx]
        #print(img_name)
        image = cv2.imread(f'{self.root_dir}/{img_name}').T
        image = image/255 #added
        # find label from the dictionary
        label = self.labels[img_name[0:len(img_name)-4]]
        #print(label)
        sample = {'image': image.astype('Float32'), 'label': label}

        return sample
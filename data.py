import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision

label_dic = {'ants': 0, 'bees': 1}


class Data(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_single_path = os.path.join(self.path, img_name)
        img = Image.open(img_single_path)
        ts = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256),antialias=True),torchvision.transforms.ToTensor()])
        img = ts(img)
        label = label_dic[self.label_dir]
        return img, label

    def __len__(self):
        return len(self.img_path)

# Collect data


root_dir = "hymenoptera_data\\train"
test_dir = 'hymenoptera_data\\val'
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Data(root_dir, ants_label_dir)
bees_dataset = Data(root_dir, bees_label_dir)
ants_testset = Data(test_dir, ants_label_dir)
bees_testset = Data(test_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset
test_dataset = ants_testset + bees_testset

# Load data

train_load = DataLoader(train_dataset, batch_size=16)
test_load = DataLoader(test_dataset,batch_size=16)

train_size = len(train_dataset)
test_size = len(test_dataset)
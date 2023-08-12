from torch.utils.data import Dataset
import cv2
import os
class Data(Dataset):
    
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
    
    def __getitem__(self, index):
        img_name =  self.img_path[index]
        img_single_path = os.path.join(self.path,img_name)
        img = cv2.imread(img_single_path,cv2.IMREAD_COLOR)
        label = self.label_dir
        return img,label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = Data(root_dir,ants_label_dir)
bees_dataset = Data(root_dir,bees_label_dir)
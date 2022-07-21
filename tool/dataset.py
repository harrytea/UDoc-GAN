import cv2
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import re
from torchvision.transforms import InterpolationMode

# read color file
def readfile(file_path):
    dic = {}
    with open(os.path.join(file_path, "colorresize.txt"), "r") as f:  # 打开文件
        for lines in f:
            lines = lines[6:-1]
            file_index = lines.find("RGB")
            file_name = lines[0: file_index-1]

            lines = lines[file_index:]
            sRGB = re.findall("\d+", lines)
            sRGB = [int(c) for c in sRGB]
            dic[file_name] = sRGB
    return dic



class LPNet_Dataset(Dataset):
    def __init__(self, file_path, model="LPNet", ps=256):
        super().__init__()
        self.ps = ps
        self.model = model
        # root of rec_img gt_img msk_img
        self.train_root = os.path.join(file_path, "doc_color_train")   # degrade image
        self.train_list = sorted(os.listdir(self.train_root))
        self.test_root  = os.path.join(file_path, "doc_color_val")     # mask: avoid non-word regoin
        self.test_list  = sorted(os.listdir(self.test_root))
        # background color
        self.color_list = readfile(file_path)
        # use transforms
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((ps,ps)),
                                            transforms.RandomHorizontalFlip(), transforms.Normalize(mean=(0.5,), std=(0.5,)),])

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        '''  images  '''
        img = cv2.imread(os.path.join(self.train_root, self.train_list[idx]))[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        '''  colors  '''
        rgb = torch.Tensor(self.color_list[self.train_list[idx]])/255.
        rgb = (rgb - 0.5) / 0.5   # Background color
        '''  file path  '''
        file_img = self.train_list[idx]
        return file_img, img, rgb



'''  Training datasets Doc-GAN  '''
class CustomDataset(Dataset):
    def __init__(self, file_path, ps=256):
        super().__init__()
        self.ps = ps
        # root of rec_img gt_img msk_img
        self.img_root = os.path.join(file_path, "img_rec")   # degrade image
        self.img_list = sorted(os.listdir(self.img_root))
        self.msk_root = os.path.join(file_path, "msk_rec")   # mask: avoid non-word regoin
        self.msk_list = sorted(os.listdir(self.msk_root))
        self.gt_root = os.path.join(file_path, "gt")         # ground truth
        self.gt_list = []
        for i in self.img_list:
            gt_path = i[:3] + '.png'
            self.gt_list.append(gt_path)
        # use transforms
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(int(ps*1.12),
                                            InterpolationMode.BICUBIC), transforms.RandomCrop((ps,ps)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.Normalize(mean=(0.5,), std=(0.5,)),])
        self.transform_GT = transforms.Compose([transforms.ToTensor(), transforms.RandomCrop((ps,ps)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.Normalize(mean=(0.5,), std=(0.5,)),])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        '''  images  '''
        img = cv2.imread(os.path.join(self.img_root, self.img_list[idx]))[:, :, :3]
        msk = cv2.imread(os.path.join(self.msk_root, self.msk_list[idx]))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk  = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
        # find a region that doesn't contain msk
        h, w, _ = img.shape
        find = False
        while not find:
            find_w = np.random.randint(200, w-self.ps)
            find_h = np.random.randint(200, h-self.ps)
            if msk[find_h, find_w, 0]==0 and msk[find_h+self.ps, find_w, 0]==0 \
                and msk[find_h, find_w+self.ps, 0]==0 \
                    and msk[find_h+self.ps, find_w+self.ps, 0]==0:
                find=True
        # apply this region
        img = img[find_h:find_h+self.ps, find_w:find_w+self.ps, :]
        img = self.transform(img)

        '''  file path  '''
        file_img = self.img_list[idx]
        index   = np.random.randint(0, len(self.img_list))
        file_gt = self.gt_list[index]
        gt      = cv2.imread(os.path.join(self.gt_root,  self.gt_list[index]))[:, :, :3]
        gt      = cv2.cvtColor(gt,  cv2.COLOR_BGR2RGB)
        gt      = self.transform_GT(gt)
        return file_img, file_gt, img, gt



class TestUDocNet(Dataset):
    def __init__(self, file_path):
        super().__init__()
        # self.file_path = os.path.join(file_path, "rec_img_bench")
        # self.file_path = os.path.join(file_path, "Dewarpnet")
        self.file_path = os.path.join("/data4/wangyh/doc/wyh/DocScanner_L")
        self.img_list = sorted(os.listdir(self.file_path))

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,)),])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.file_path, self.img_list[idx]))[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        file = self.img_list[idx]

        # padding: let model to process
        pad = True
        padw = 0
        padh = 0
        pad_size = 256
        if pad:
            h, w, c = img.shape
            # if w%pad_size!=0 or h%pad_size!=0:
            padw = pad_size-w%pad_size
            padh = pad_size-h%pad_size
            transform = transforms.Pad([0, 0, padw, padh])  # l, t, r, b
            img = self.transform(img)
            img = transform(img)
            return file, img, padw, padh
        return file, img
#===================================================================
# ​针对单张图像进行验证/测试的数据加载器，主要用于 ​模型推理阶段的单图像处理
#===================================================================
import os
import h5py
import torch
import random
from PIL import Image
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as FF


# ############### 这边 Percent 表示 Percent 的来训练 ###################
image_width224 = 224
image_height224 = 224
# ##########################################################


class valid_dataloader_patch(data.Dataset):
    def __init__(self, scoreFilePath):
        self.waterpath = scoreFilePath

    def __getitem__(self, index):
        waterimage = Image.open(self.waterpath)
        img = self.augData(waterimage)
        return img, self.waterpath

    def __len__(self):
        return len(self.waterpath)

    def augData(self, water):
        water = transforms.Resize([image_width224, image_height224])(water)
        water = transforms.ToTensor()(water)
        return water
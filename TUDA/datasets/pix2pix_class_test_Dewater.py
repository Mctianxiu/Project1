#==================================================================
# 数据加载模块，用于​图像到图像转换任务
# ​数据加载：从指定目录加载图像文件，构建 PyTorch 数据集（Dataset 类）。
# ​预处理：对图像进行数据增强或格式转换（如裁剪、归一化等）。
# ​数据配对：生成配对的输入（A）和目标（B）图像，用于监督式图像生成任务。
#==================================================================
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from Test_global import test_num_id  # 从外部模块导入测试编号标识

# 定义支持的图像文件扩展名列表
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
]

def is_image_file(filename):
    """检查文件是否为支持的图像格式"""
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    """创建图像数据集列表
    Args:
        dir (str): 图像目录路径
    Returns:
        list: 包含所有图像路径的列表
    """
    images = []
    if not os.path.isdir(dir):
        raise Exception('Check dataroot')
    # 遍历目录树
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):  # 过滤非图像文件
                path = os.path.join(dir, fname)
                item = path
                images.append(item)
    return images

def default_loader(path):
    """默认图像加载器
    使用PIL库加载RGB格式图像
    """
    return Image.open(path).convert('RGB')

class pix2pix(data.Dataset):
    """面向pix2pix/CycleGAN等图像转换任务的自定义数据集类
    
    关键功能：
    - 加载配对图像数据（A->B）
    - 应用数据增强变换
    - 支持特殊测试模式
    """
    
    def __init__(self, root, transform=None, loader=default_loader, seed=None):
        """
        Args:
            root (str): 数据根目录
            transform (callable, optional): 数据增强函数
            loader (function, optional): 图像加载函数
            seed (int, optional): 随机种子
        """
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n" 
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.root = root          # 数据存储路径
        self.imgs = imgs          # 图像路径列表
        self.transform = transform# 数据增强变换函数
        self.loader = loader      # 图像加载器
        
        if seed is not None:      # 固定随机种子
            np.random.seed(seed)

    def __getitem__(self, index):
        """获取单个数据样本
        特殊处理逻辑：
        - 测试模式下根据test_num_id动态构造路径
        - 应用数据增强变换
        """
        path = 0  # 初始化路径变量
        
        ########################
        # 测试模式特殊处理路径
        ########################
        path_test_data = R'C:\Users\YuanBao\Desktop\1/'  # 硬编码测试路径（需优化）
        
        # 当测试模式为'groups_supply_singles'时
        if test_num_id == 'groups_supply_singles':  
            # 构造文本文件路径
            path_txt = path_test_data + str(test_num_id) + '.txt'
            f = open(path_txt, 'r')
            self.txt = f.readlines()
            # 构造完整图像路径
            path = path_test_data + str(test_num_id) + '/' + self.txt[index].split('\n')[0]
        
        ########################
        # 加载并处理图像
        ########################
        img = self.loader(path)  # 加载图像
        
        # 应用数据增强变换（假设transform处理多图对齐）
        if self.transform is not None:
            # 注意：这里传入了4个相同的img，可能是为了适配某些特殊变换
            imgA, imgB, _, _ = self.transform(img, img, img, img)
            
        #############
        # 返回数据字典
        return {
            'A': imgA,       # 输入图像（转换前）
            'B': imgB,       # 目标图像（转换后） 
            'A_paths': path  # 原始路径信息
        }

    def __len__(self):
        """返回数据集总样本数"""
        return len(self.imgs)
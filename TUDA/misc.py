# ================================================================
# ​深度学习模型训练工具包(misc一般是单词miscellaneous杂项，混合的缩写)
# ================================================================
import torch
import os
import numpy as np

# 实验目录管理模块
def create_exp_dir(exp):
    """
    创建实验目录（用于保存模型、日志等）
    
    参数：
        exp (str): 实验目录路径
    
    返回值：
        bool: 总是返回True（创建成功或目录已存在）
    """
    try:
        os.makedirs(exp)  # 尝试创建目录
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass  # 如果目录已存在则忽略
    return True


# 神经网络权重初始化模块
def weights_init(m):
    """
    初始化神经网络权重（适用于GAN模型）
    
    参数：
        m (nn.Module): 待初始化的神经网络模块
    
    初始化策略：
        - 卷积层：正态分布初始化（均值0，标准差0.02）
        - BatchNorm层：gamma参数N(1,0.02)，beta参数初始化为0
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # 卷积层初始化
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:  # BN层初始化
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# 数据加载器构建模块
def getLoader(datasetName, dataroot, originalSize, imageSize, batchSize=64, workers=4,
              mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), split='train', shuffle=True, seed=None):
    """
    创建数据加载器（支持训练集和验证集的不同预处理）
    
    参数：
        datasetName (str): 数据集名称（当前仅支持CycleGAN_Dewater_test）
        dataroot (str): 数据根目录
        originalSize (int): 原始图像尺寸（调整大小用）
        imageSize (int): 最终输入网络的图像尺寸
        batchSize (int): 批量大小，默认64
        workers (int): 数据加载线程数，默认4
        mean (tuple): 归一化均值，默认(0.5, 0.5, 0.5)
        std (tuple): 归一化标准差，默认(0.5, 0.5, 0.5)
        split (str): 数据集划分（train/val/test）
        shuffle (bool): 是否打乱数据，默认True
        seed (int): 随机种子，默认None
    
    返回值：
        DataLoader: PyTorch数据加载器
    """
    if datasetName == 'CycleGAN_Dewater_test':
        from datasets.pix2pix_class_test_Dewater import pix2pix as commonDataset
        import transforms.pix2pix as transforms

    # 训练集预处理流程（包含数据增强）
    if split == 'train':
        dataset = commonDataset(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Scale(originalSize),   # 调整到原始尺寸
                transforms.RandomCrop(imageSize),  # 随机裁剪
                transforms.augData(),             # 数据增强（需查看具体实现）
                transforms.ToTensor(),            # 转为Tensor
                transforms.Normalize(mean, std)    # 标准化
            ]),
            seed=seed)
    # 验证集/测试集预处理流程
    else:
        dataset = commonDataset(
            root=dataroot,
            transform=transforms.Compose([
                transforms.Scale(imageSize),       # 直接缩放到目标尺寸
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            seed=seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=int(workers))


# 训练指标跟踪模块
class AverageMeter(object):
    """
    计算并存储平均值和当前值（用于跟踪损失、准确率等指标）
    
    使用示例：
        meter = AverageMeter()
        for data in loader:
            loss = ...
            meter.update(loss.item(), batch_size)
        print(f"Average loss: {meter.avg}")
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有统计量"""
        self.val = 0    # 当前值
        self.avg = 0    # 平均值
        self.sum = 0    # 总值
        self.count = 0  # 计数

    def update(self, val, n=1):
        """
        更新统计量
        
        参数：
            val (float): 新的观测值
            n (int): 该观测值对应的样本数（默认1）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # 更新平均值


# 生成图像缓存池（用于稳定GAN训练）
class ImagePool:
    """
    历史图像缓存池（缓解GAN模式崩溃问题）
    
    工作机制：
        - 当池未满时：存储当前图像
        - 当池已满时：50%概率替换池中随机一张图像
    
    论文参考：CycleGAN (https://arxiv.org/abs/1703.10593)
    """
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if pool_size > 0:
            self.num_imgs = 0     # 当前存储数量
            self.images = []      # 图像存储列表

    def query(self, image):
        """
        查询/更新图像池
        
        参数：
            image (Tensor): 当前生成的图像
            
        返回值：
            Tensor: 来自缓存池或当前生成的图像
        """
        if self.pool_size == 0:  # 禁用缓存池
            return image
        
        # 池未满时直接存储
        if self.num_imgs < self.pool_size:
            self.images.append(image.clone())
            self.num_imgs += 1
            return image
        
        # 池已满时随机替换
        if np.random.uniform(0, 1) > 0.5:
            random_id = np.random.randint(self.pool_size, size=1)[0]
            tmp = self.images[random_id].clone()
            self.images[random_id] = image.clone()
            return tmp
        else:
            return image


# 学习率调整模块
def adjust_learning_rate(optimizer, init_lr, epoch, factor, every):
    """
    线性衰减学习率（每every轮衰减固定值）
    
    参数：
        optimizer: 优化器对象
        init_lr (float): 初始学习率
        epoch (int): 当前训练轮次
        factor: 未使用的参数（保留接口）
        every (int): 衰减周期（每多少轮衰减一次）
    """
    lrd = init_lr / every  # 计算衰减步长
    old_lr = optimizer.param_groups[0]['lr']
    lr = old_lr - lrd      # 线性衰减
    lr = max(lr, 0)        # 学习率不低于0
    
    # 更新所有参数组的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

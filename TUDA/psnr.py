#======================================================
# 图像质量评估指标 SSIM (结构相似性) 和 PSNR (峰值信噪比)
# 适用于图像生成任务（如超分辨率、去噪等）的效果评估
#======================================================
from math import exp
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def gaussianffa(window_size, sigma):
    """
    生成一维高斯核
    Args:
        window_size (int): 窗口大小（核长度）
        sigma (float): 高斯分布的标准差
    Returns:
        Tensor: 一维高斯核，已归一化（总和为1）
    """
    # 计算高斯函数值：exp(-(x - center)^2 / (2*sigma^2))
    gauss = torch.Tensor([exp(-(x - window_size//2)**2 / (2*sigma**2)) 
                        for x in range(window_size)])
    return gauss / gauss.sum()  # 归一化处理

def create_windowffa(window_size, channel):
    """
    创建二维高斯窗口（用于SSIM的滑动窗口计算）
    Args:
        window_size (int): 窗口边长
        channel (int): 输入数据的通道数
    Returns:
        Variable: 四维张量 [channel, 1, window_size, window_size]
    """
    # 生成一维高斯核 [window_size, 1]
    _1D_window = gaussianffa(window_size, 1.5).unsqueeze(1)
    
    # 通过矩阵乘法生成二维高斯核 [window_size, window_size]
    _2D_window = _1D_window.mm(_1D_window.t())
    
    # 扩展为四维张量 [channel, 1, window_size, window_size]
    window = Variable(_2D_window.float().unsqueeze(0).unsqueeze(0))
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssimffa(img1, img2, window, window_size, channel, size_average=True):
    """
    计算SSIM核心函数（单次前向计算）
    Args:
        img1 (Tensor): 图像1 [B, C, H, W]
        img2 (Tensor): 图像2 [B, C, H, W]
        window (Tensor): 高斯窗口 [C, 1, win, win]
        window_size (int): 窗口大小
        channel (int): 输入通道数
        size_average (bool): 是否返回全局平均值
    Returns:
        Tensor: SSIM值（标量或按通道值）
    """
    # 计算局部均值（通过高斯滤波）
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    # 计算局部方差和协方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # SSIM计算公式（添加稳定性常数C1/C2）
    C1 = (0.01 ** 2)  # 亮度对比度常数
    C2 = (0.03 ** 2)  # 结构对比度常数
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-8)  # 防止除以零
    
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

def ssimffa(img1, img2, window_size=11, size_average=True):
    """
    计算SSIM主函数（输入范围限制+窗口创建）
    Args:
        img1 (Tensor): 图像1 [B, C, H, W]
        img2 (Tensor): 图像2 [B, C, H, W]
        window_size (int): 滑动窗口大小，默认11
        size_average (bool): 是否返回全局平均值
    Returns:
        Tensor: SSIM值（标量）
    """
    # 输入范围限制在[0,1]之间
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    
    # 获取通道数并创建窗口
    (_, channel, _, _) = img1.size()
    window = create_windowffa(window_size, channel)
    
    # 将窗口移动到对应设备
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssimffa(img1, img2, window, window_size, channel, size_average)

def norm_ip(img, min, max):
    """
    线性归一化到指定范围
    Args:
        img (Tensor): 输入图像
        min (float): 当前范围最小值
        max (float): 当前范围最大值
    Returns:
        Tensor: 归一化到[0,1]的图像
    """
    img.clamp_(min=min, max=max)  # 限制原始范围
    img.add_(-min).div_(max - min + 1e-5)  # 线性映射到[0,1]
    return img

def norm_range(t, range):
    """
    智能归一化函数
    Args:
        t (Tensor): 输入图像
        range (tuple): 指定归一化范围，None表示自动计算
    Returns:
        Tensor: 归一化后的图像
    """
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, t.min(), t.max())

def psnrffa(pred, gt):
    """
    基于RMSE的PSNR计算（输入范围[0,1]）
    Args:
        pred (Tensor): 预测图像
        gt (Tensor): 真实图像
    Returns:
        float: PSNR值（单位：dB）
    """
    # 数据准备
    pred = pred.data.cpu().squeeze()
    gt = gt.data.cpu().squeeze()
    
    # 归一化处理
    pred_norm = norm_range(pred, None).clamp(0, 1)
    gt_norm = norm_range(gt, None).clamp(0, 1)
    
    # 转换到numpy计算
    pred_np = pred_norm.numpy()
    gt_np = gt_norm.numpy()
    
    # 计算RMSE
    mse = np.mean((pred_np - gt_np) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def psnrffa_wang(pred, gt):
    """
    基于MSE的PSNR计算（输入范围[0,255]）
    Args:
        pred (Tensor): 预测图像
        gt (Tensor): 真实图像
    Returns:
        float: PSNR值（单位：dB）
    """
    # 数据准备
    pred = pred.data.cpu().squeeze()
    gt = gt.data.cpu().squeeze()
    
    # 归一化并扩展到[0,255]
    pred_norm = norm_range(pred, None).clamp(0, 1) * 255
    gt_norm = norm_range(gt, None).clamp(0, 1) * 255
    
    # 转换到numpy计算
    pred_np = pred_norm.numpy()
    gt_np = gt_norm.numpy()
    
    # 计算MSE
    mse = np.mean((pred_np - gt_np) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255.0 ** 2 / mse)
#==================================
# 单通道适配VGGLoss完整实现
#==================================
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

class VGGLoss(nn.Module):
    def __init__(self, device, layer_weights=None):
        super().__init__()
        # 初始化单通道适配的VGG
        original_vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # 修改首层卷积为单通道输入
        first_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            # 取原三通道权重平均作为初始化
            first_conv.weight.data = original_vgg.features[0].weight.data.mean(dim=1, keepdim=True)
            first_conv.bias.data = original_vgg.features[0].bias.data.clone()
        
        # 构建特征提取网络
        self.feature_extractor = nn.Sequential(
            first_conv,
            *list(original_vgg.features.children())[1:20]  # 取到conv5_1前
        )
        
        # 配置特征层权重
        self.layer_weights = {
            '4': 1.0/32,   # conv1_1
            '9': 1.0/16,   # conv2_1
            '14': 1.0/8,   # conv3_1
            '23': 1.0/4,   # conv4_1
            '28': 1.0      # conv5_1
        } if layer_weights is None else layer_weights
        
        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        # 注册归一化参数
        self.register_buffer('mean', torch.tensor([0.5]).view(1,1,1,1))
        self.register_buffer('std', torch.tensor([0.5]).view(1,1,1,1))
        
        self.to(device)
        self.feature_extractor.eval()

    def _normalize(self, x):
        """将[0,1]范围输入标准化为VGG需要的格式"""
        return (x - self.mean) / self.std

    def forward(self, input, target):
        """
        输入参数：
            input: 单通道图像张量 [B,1,H,W] 或 [B,H,W]
            target: 同input格式
        返回：
            加权特征损失
        """
        # 维度处理
        if input.dim() == 3:
            input = input.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        # 标准化处理
        input_norm = self._normalize(input)
        target_norm = self._normalize(target)
        
        # 复制通道适配VGG输入
        input_feats = input_norm.repeat(1,3,1,1)  # [B,3,H,W]
        target_feats = target_norm.repeat(1,3,1,1)
        
        total_loss = 0.0
        for name, module in self.feature_extractor.named_children():
            # 前向传播
            input_feats = module(input_feats)
            target_feats = module(target_feats)
            
            # 计算当前层损失
            if name in self.layer_weights:
                layer_loss = torch.nn.functional.l1_loss(input_feats, target_feats)
                total_loss += self.layer_weights[name] * layer_loss
                
        return total_loss

    @staticmethod
    def preprocess_tif(path, transform=None):
        """
        TIFF预处理工具方法
        用法：
            img_tensor = VGGLoss.preprocess_tif("image.tif")
        """
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.type(torch.float32)),
            transforms.Resize(512),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        transform = transform or default_transform
        
        with Image.open(path) as img:
            # 转换为单通道灰度图
            gray_img = img.convert('L')
            # 应用转换
            return transform(gray_img)
    
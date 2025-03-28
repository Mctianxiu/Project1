#====================
# TRDA训练代码
#====================
import os
import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.optim import Adam
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from models.networks import define_G, define_D, NLayerDiscriminator, SpectralDiscriminator
from models.losses import VGGLoss                       # 自定义的VGGLoss
from torchvision.models import vgg19, VGG19_Weights     # PyTorch 1.12+需要这样导入
from torchvision.models.feature_extraction import create_feature_extractor
#====================
# 定义训练参数
#====================
class Opt:
    def __init__(self):
        # 数据集参数
        self.dataroot = "./datasets"             # 数据集根目录
        self.phase = "train"                     # 训练阶段
        self.load_size = 512                     # 图像加载尺寸
        self.crop_size = 512                     # 随机裁剪尺寸
        self.batch_size = 4                      # 批次大小
        self.num_workers = 4                     # 数据加载线程数
        
        # 模型参数
        self.input_nc = 1                        # 输入通道数
        self.output_nc = 1                       # 输出通道数
        self.ndf = 64                            # 判别器基础通道数
        self.gpu_ids = [0]                       # 使用的GPU编号
        
        # 训练参数
        self.epochs = 200                        # 总训练轮数

        # 图片保存位置
        self.checkpoint_dir = './checkpoints'  
        self.result_dir = './results'            # 新增结果目录
#===============================
# 自定义数据集类
#===============================
class TripleDomainDataset(Dataset):
    def __init__(self, opt):
        self.source_dir = os.path.join(opt.dataroot, opt.phase, 'source')
        self.target_dir = os.path.join(opt.dataroot, opt.phase, 'target')
        self.real_domain_dir = os.path.join(opt.dataroot, opt.phase, 'real_domain')
        
        # 获取文件名列表（假设所有目录文件同名）
        self.filenames = sorted(os.listdir(self.source_dir))
        
        # 定义变换
        self.transform = transforms.Compose([
            transforms.Resize(opt.load_size),
            transforms.RandomCrop(opt.crop_size),
            transforms.Lambda(lambda img: torch.from_numpy(np.array(img).astype(np.float32))),  # 直接转换为float32张量
            transforms.Lambda(lambda x: x.unsqueeze(0)),                                        # 添加通道维度 [1, H, W]
            transforms.Lambda(lambda x: x / 65535.0),                                           # 16位转[0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])                                         # [-1,1]范围
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        def load_16bit_tif(path):
            """加载16位TIFF的专用方法"""
            with Image.open(path) as img:
                # 确保为16位模式
                if img.mode != 'I;16':
                    img = img.convert('I;16')
                return self.transform(img)
            
        return {
            'x_s': load_16bit_tif(os.path.join(self.source_dir, self.filenames[index])),
            'y_s': load_16bit_tif(os.path.join(self.target_dir, self.filenames[index])),
            'x_r': load_16bit_tif(os.path.join(self.real_domain_dir, self.filenames[index]))
        }
#=========================
# 定义域间自适应训练函数
#=========================
class InterDomainTrainer:
    def __init__(self, opt):
        # 确定设备
        self.device = torch.device('cuda' if opt.gpu_ids else 'cpu')
        # 确定文件保存路径
        self.checkpoint_dir = opt.checkpoint_dir
        self.result_dir = opt.result_dir  # 新增结果目录
        # 初始化生成器
        self.G_trans = define_G(opt.input_nc, opt.output_nc, 'Correction')  # 图像翻译模块
        self.G_enhance = define_G(opt.output_nc, opt.output_nc, 'unet')     # 图像增强模块
        
        # 初始化判别器
        self.D_img = define_D(opt.output_nc, opt.ndf, 'n_layers')           # 图像级判别器
        self.D_feat = SpectralDiscriminator(input_nc=64)                    # 特征级判别器
        self.D_out = NLayerDiscriminator(opt.output_nc, opt.ndf)            # 输出级判别器
        
        # 优化器
        self.opt_G = Adam(list(self.G_trans.parameters()) + list(self.G_enhance.parameters()),
                          lr=1e-4, betas=(0.5, 0.999))
        self.opt_D = Adam(list(self.D_img.parameters()) + list(self.D_feat.parameters()) + list(self.D_out.parameters()),
                          lr=2e-4, betas=(0.5, 0.999))
        
        # 损失函数
        self.criterion_adv = nn.MSELoss()                       # 对抗损失
        self.criterion_content = nn.L1Loss()                    # 内容损失
        self.criterion_task = nn.L1Loss()                       # 任务L1损失
        self.criterion_vgg = VGGLoss(device=self.device)        # VGG感知损失
        
        # 超参数
        self.lambda_img = 1
        self.lambda_cont = 100
        self.lambda_task = 10
        self.lambda_feat = 0.0005
        self.lambda_out = 0.0005
    # 用于保存最后的 y_r 图像
    def _save_images(self, epoch, batch_idx, x_s, x_st, y_st, y_r):
        """保存16位无符号整数TIFF"""
        # 反归一化流程
        def denormalize(tensor):
            # [-1,1] -> [0,1]
            return (tensor + 1.0) / 2.0
        
        # 创建保存目录
        save_dir = os.path.join(self.result_dir, f"epoch_{epoch+1}")
        os.makedirs(save_dir, exist_ok=True)

        # 处理每个样本
        for i in range(x_s.size(0)):
            # 反归一化并转换为numpy数组
            def process(t):
                arr = denormalize(t[i].cpu().detach().squeeze()).numpy()
                return (arr * 65535).astype(np.uint16)  # 转16位整数
            
            # 生成各通道数据
            data_dict = {
                'x_s': process(x_s),
                'x_st': process(x_st),
                'y_st': process(y_st),
                'y_r': process(y_r)
            }

            # 保存为16位TIFF
            for name, arr in data_dict.items():
                img = Image.fromarray(arr, mode='I;16')
                img.save(os.path.join(save_dir, 
                    f"batch{batch_idx}_sample{i}_{name}.tif"))

        # 保存对比图（可选）
        sample_idx = 0  # 取第一个样本
        grid = np.hstack([
            data_dict['x_s'], 
            data_dict['x_st'],
            data_dict['y_st'],
            data_dict['y_r']
        ])
        Image.fromarray(grid, mode='I;16').save(
            os.path.join(save_dir, f"comparison_batch{batch_idx}.tif"))
    # 用于内容损失的带权重VGG层
    def _get_vgg_layers(self):
        vgg = vgg19(pretrained=True).features.eval()    # 加载预训练的VGG19模型，提取VGG19的卷积部分（排除全连接层），设置为评估模式，关闭Dropout和BatchNorm的随机性
        return nn.ModuleList([
            vgg[:2],   # conv1_1
            vgg[2:5],  # conv2_1
            vgg[5:7],  # conv3_1
            vgg[7:10], # conv4_1
            vgg[10:12] # conv5_1
        ]).to(self.criterion_vgg.mean.device)
    # 定义训练流程
    def train_step(self, batch):
        # 解包批次数据（batch是一个字典，包含三个键值对）
        x_s = batch['x_s'].to(self.device)  # 合成域输入
        y_s = batch['y_s'].to(self.device)  # 监督目标
        x_r = batch['x_r'].to(self.device)  # 真实域输入
        
        # ----------------------
        # 1. 图像翻译模块前向
        # ----------------------
        x_st = self.G_trans(x_s)     # 翻译后的合成图像
        
        # ----------------------
        # 2. 增强模块前向
        # ----------------------
        y_st, feat_st = self.G_enhance(x_st,x_r)  # 传入真实域输入x_r
        y_r, feat_r = self.G_enhance(x_r,x_s)     # 跨域增强
        
        # ----------------------
        # 3. 计算所有损失
        # ----------------------
        # （1）图像级对抗损失，(x_st, x_r)
        d_img_loss = self._calc_img_adv_loss(x_st, x_r)
        
        # （2）内容保留损失（VGG多尺度特征匹配），(x_s, x_st)
        # 论文中的权重（按 conv1-1 到 conv5-1 顺序）
        content_weights = [1/32, 1/16, 1/8, 1/4, 1.0]
        content_loss = 0
        for i in range(5):                                              # 遍历 conv1-1 到 conv5-1
            layer_loss = self.criterion_content(self.vgg_layers[i](x_s), 
                                        self.vgg_layers[i](x_st))
            content_loss += content_weights[i] * layer_loss             # 关键：乘以对应权重
        
        # （3）任务损失（L1 + VGG感知）
        task_loss = 0.8*self.criterion_task(y_st, y_s) + 0.2*self.criterion_vgg(y_st, y_s)
        
        # （4）特征级对抗损失
        feat_loss = self._calc_feat_adv_loss(feat_st, feat_r)
        
        # （5）输出级对抗损失
        out_loss = self._calc_out_adv_loss(y_st, y_r)
        
        # 总生成器损失
        G_total_loss = (self.lambda_img * d_img_loss +
                        self.lambda_cont * content_loss +
                        self.lambda_task * task_loss +
                        self.lambda_feat * feat_loss +
                        self.lambda_out * out_loss)
        
        # ----------------------
        # 4. 反向传播与优化
        # ----------------------
        self.opt_G.zero_grad()
        G_total_loss.backward(retain_graph=True)
        self.opt_G.step()
        
        # 更新判别器
        self._update_discriminators(x_st, x_r, feat_st, feat_r, y_st, y_r)
        
        return {
            'G_loss': G_total_loss.item(),
            'D_img_loss': d_img_loss.item(),
            'content_loss': content_loss.item(),
            'task_loss': task_loss.item()
        }
    # WGAN-GP对抗损失
    def _calc_img_adv_loss(self, x_st, x_r):
        
        real_pred = self.D_img(x_r)
        fake_pred = self.D_img(x_st.detach())
        
        # 梯度惩罚项
        alpha = torch.rand(x_r.size(0), 1, 1, 1).to(x_r.device)
        interpolates = (alpha * x_r + (1 - alpha) * x_st).requires_grad_(True)
        d_interpolates = self.D_img(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        
        d_loss = fake_pred.mean() - real_pred.mean() + grad_penalty
        return d_loss
    # 特征空间对抗
    def _calc_feat_adv_loss(self, feat_st, feat_r):
        
        real_feat = self.D_feat(feat_r)
        fake_feat = self.D_feat(feat_st.detach())
        return fake_feat.mean() - real_feat.mean()
    # 输出空间对抗
    def _calc_out_adv_loss(self, y_st, y_r):
        
        real_out = self.D_out(y_r)
        fake_out = self.D_out(y_st.detach())
        return fake_out.mean() - real_out.mean()
    # 更新三个判别器
    def _update_discriminators(self, x_st, x_r, feat_st, feat_r, y_st, y_r):
        # 更新图像判别器
        self.opt_D.zero_grad()
        d_img_loss = self._calc_img_adv_loss(x_st, x_r)
        d_img_loss.backward()
        
        # 更新特征判别器
        d_feat_loss = self._calc_feat_adv_loss(feat_st, feat_r)
        d_feat_loss.backward()
        
        # 更新输出判别器
        d_out_loss = self._calc_out_adv_loss(y_st, y_r)
        d_out_loss.backward()
        
        self.opt_D.step()
#=====================
# 训练循环函数
#=====================
def train(opt):
    # 初始化数据集
    dataset = TripleDomainDataset(opt)
    # 初始化数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers  # 建议添加这个参数加速数据加载
    )
    # 初始化训练器
    trainer = InterDomainTrainer(opt)
    
    # 训练循环
    for epoch in range(opt.epochs):
        for batch_idx, batch in enumerate(train_loader):
            # 数据转移到设备
            x_s = batch['x_s'].to(trainer.device)
            y_s = batch['y_s'].to(trainer.device)
            x_r = batch['x_r'].to(trainer.device)
            
            # 执行训练步骤
            metrics = trainer.train_step({'x_s':x_s, 'y_s':y_s, 'x_r':x_r})
            
            # 每100个批次保存一次生成结果
            # 保存的对比图像将包含四列：​原始合成图像 (x_s)，​翻译后图像 (x_st)，​增强合成结果 (y_st)，​增强真实图像 (y_r)     
            if batch_idx % 100 == 0:
                # ========== 保存生成图像y_r =======================================
                with torch.no_grad():
                    x_st = trainer.G_trans(x_s)
                    y_st, _ = trainer.G_enhance(x_st, x_r)
                    y_r, _ = trainer.G_enhance(x_r, x_s)
                    trainer._save_images(epoch, batch_idx, x_s, x_st, y_st, y_r)
                # =================================================================
                
                # 打印日志
                print(f'Epoch [{epoch+1}/{opt.epochs}] Batch [{batch_idx}/{len(train_loader)}]')
                print(f'G Loss: {metrics["G_loss"]:.4f} | D Loss: {metrics["D_img_loss"]:.4f}')

# ===================== 最后执行训练 =====================
if __name__ == "__main__":
    opt = Opt()
    train(opt)  # 此时train函数已定义
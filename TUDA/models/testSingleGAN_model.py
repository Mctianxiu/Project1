#========================================================
# TUDA模型的完整测试流程实现---------------------(测试代码)
# 包含域间（Inter-Domain）和域内（Intra-Domain）两阶段自适应
#========================================================
import os
import torch
import warnings
import numpy as np
from PIL import Image
from . import networks                                 # 网络结构定义模块，这里的点号（.）代表当前包（package）
import util.util as util                               # 工具函数（如张量转图像）
from .base_model import BaseModel                      # 模型基类
from torchvision import transforms                     # 图像变换
from torch.autograd import Variable                    # 兼容旧版PyTorch的变量封装
from collections import OrderedDict
warnings.filterwarnings("ignore")                      # 忽略所有警告信息
os.environ['CUDA_VISIBLE_DEVICES'] = "0"               # 指定使用第一个GPU
device_ids = [0]                                       # 多GPU训练设备ID（此处仅使用0号单卡）

# 全局测试配置
from Test_global import test_num_id                    # 测试编号（用于结果目录命名）
IQA_threshold = 63.968                                 # RUIQA质量评分阈值（论文中的Easy/Hard分类阈值）

# 图像归一化工具函数
def norm_ip(img, min, max):
    """将图像像素值压缩到[min,max]范围并归一化到[0,1]"""
    img.clamp_(min=min, max=max)             # clamp如果某个像素值小于 min，则将其设置为 min；如果某个像素值大于 max，则将其设置为 max
    img.add_(-min).div_(max - min)
    return img
def norm_range(t, range):
    if range is not None:                    # 如果用户指定了归一化范围，使用用户指定的范围 [range[0], range[1]] 对张量 t 进行归一化
        return norm_ip(t, range[0], range[1])
    else:                                    # 如果用户没有指定归一化范围，使用张量 t 的最小值 t.min() 和最大值 t.max() 作为归一化范围
        return norm_ip(t, t.min(), t.max())
#==================================
#TUDA测试阶段模型（继承自BaseModel）
#==================================
class TestSingleGANModel(BaseModel):
    
    def name(self):
        """返回模型名称"""
        return 'TestSingleGANModel'

    def initialize(self, opt):
        """初始化模型组件"""
        assert (not opt.isTrain)                                # 确保是测试模式
        BaseModel.initialize(self, opt)                         # 调用基类初始化
        
        # 初始化输入张量（A为输入图像，B未使用）
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.imageSize, opt.imageSize)
        self.input_B = self.Tensor(opt.batchSize, opt.input_nc, opt.imageSize, opt.imageSize)
        
        #===============================================
        # 加载RUIQA质量评估模型（论文第二阶段的关键组件）
        #===============================================
        from models.Hyper_Model_Have_Pre_Resnet50_have_5_5 import UWIQN
        checkpoint = R'C:\Users\YuanBao\Desktop\...\epoch200.pth'                                   # 预训练权重路径
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化并加载模型
        self.net_IQA = UWIQN(224).to(device)                                                        # 输入尺寸224x224
        self.net_IQA.load_state_dict(torch.load(checkpoint, map_location='cuda'))
        for p in self.net_IQA.parameters():
            p.requires_grad = False                                                                 # 冻结参数
        self.net_IQA.eval()                                                                         # 评估模式

        #===============================================
        # 初始化生成器组件（论文两个阶段的核心网络）
        #===============================================
        # 域间生成器（Inter-Domain Adaptation）
        self.netG_B_inter = networks.define_G(opt.input_nc, opt.output_nc, opt.which_model_netG_B)
        # 域内生成器（Intra-Domain Adaptation）
        self.netG_B_intra = networks.define_G(opt.input_nc, opt.output_nc, opt.which_model_netG_B)
        # 基线生成器（Baseline，可能用于对比）
        self.netG_B_bl = networks.define_G(opt.input_nc, opt.output_nc, opt.which_model_netG_B)
        
        # GPU并行化（如果可用）
        if torch.cuda.is_available():
            self.netG_B_inter = torch.nn.DataParallel(self.netG_B_inter, device_ids=device_ids).cuda()
            self.netG_B_intra = torch.nn.DataParallel(self.netG_B_intra, device_ids=device_ids).cuda()
            self.netG_B_bl = torch.nn.DataParallel(self.netG_B_bl, device_ids=device_ids).cuda()

        # 模型名称列表（用于权重加载/保存）
        self.model_names = ['G_B_inter', 'G_B_intra', 'G_B_bl']
        which_epoch = opt.which_epoch
        self.load_networks(which_epoch)                          # 加载预训练权重

        # 打印网络结构
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_B_inter)
        networks.print_network(self.netG_B_intra)
        networks.print_network(self.netG_B_bl)
        print('-----------------------------------------------')

    def set_input(self, input):
        """设置输入数据"""
        self.input_A = input['A']                           # 输入图像张量
        self.image_name = input['A_paths']                  # 图像路径列表
        self.image_paths = input['A_paths']

    def test(self):
        """执行测试流程（核心逻辑）"""
        self.syn_A = Variable(self.input_A, volatile=True)          # 封装输入（兼容旧版）
        self.flag = 0                                               # 标记当前处理阶段（0:域间，1:域内）
        
        with torch.no_grad():                                       # 禁用梯度计算
            # Step 1: 基线生成器推理（可能用于对比或初始化）
            self.enhanced_A_bl, _ = self.netG_B_bl(self.syn_A)
            
            # Step 2: 域间生成器推理（论文第一阶段）
            # enhanced_A_inter: 增强结果，enhanced_A_feat_inter: 中间特征（可能用于后续处理）
            self.enhanced_A_inter, self.enhanced_A_feat_inter = self.netG_B_inter(self.syn_A)
            
            # 创建结果保存目录
            path_e = './Inter_' + str(test_num_id) + '/'
            if not os.path.exists(path_e):
                os.makedirs(path_e)
            
            # 保存域间增强结果
            filename = path_e + 'Inter_' + self.image_name[0].split('/')[-1]
            print(filename)                                                              # 打印输出路径
            image_numpy = util.tensor2im(self.enhanced_A_inter.data)                     # 张量转图像
            util.save_image(image_numpy, filename)                                       # 保存图像
            
            # Step 3: RUIQA质量评估（论文第二阶段决策逻辑）
            temp = Image.open(filename)                                         # 重新加载生成的图像
            temp = transforms.Resize([224, 224])(temp)                          # 调整尺寸适配IQA模型
            temp = transforms.ToTensor()(temp)                                  # 转为张量
            temp1 = torch.unsqueeze(temp, 0).cuda()                             # 添加批次维度
            IQA_score = self.net_IQA(temp1)                                     # 质量评分
            score = np.reshape(np.array(IQA_score.data.cpu()), (-1, 1))[0][0]   # 提取标量值
            
            # 记录评分结果
            with open('result_Score.txt', "a") as file:
                file.write(f"pic:{self.image_name[0]}, score:{score}\n")
            
            # Step 4: 动态路由（根据质量分数选择处理分支）
            if score < IQA_threshold:                                           # Hard样本，进入域内适应
                # 域内生成器推理（论文第二阶段）
                self.enhanced_A_intra, self.enhanced_A_feat_intra = self.netG_B_intra(self.syn_A)
                self.enhanced_A = self.enhanced_A_intra                         # 最终结果
                self.enhanced_intra_inter = self.enhanced_A_inter               # 保留域间结果
                self.flag = 1                                                   # 标记为域内处理
                
                # 记录Hard样本
                with open('result_Hard_Score.txt', "a") as file:
                    file.write(f"pic:{self.image_name[0]}, score:{score}\n")
            else:                                                                # Easy样本，直接使用域间结果
                self.enhanced_A = self.enhanced_A_inter
                self.flag = 0

    def get_image_paths(self):
        """获取输入图像路径"""
        return self.image_paths

    def get_current_visuals(self):
        """获取可视化结果（用于结果展示）"""
        enhanced_A = util.tensor2im(self.enhanced_A.data)                         # 最终增强结果
        
        if self.flag == 0:                                                        # 域间结果
            return OrderedDict([('enhanced_inter', enhanced_A)])
        elif self.flag == 1:                                                      # 域内结果
            enhanced_AB = util.tensor2im(self.enhanced_intra_inter.data)
            return OrderedDict([('enhanced_intra', enhanced_A)])
#=================================================================
# PyTorch模型的基础类，用于定义模型的基本结构和方法
#=================================================================
import os
import torch
from collections import OrderedDict
from util import util  # 自定义工具模块
from . import networks  # 网络结构模块

class BaseModel():
    """模型基类，定义深度学习模型的通用接口和基础功能"""
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """修改命令行参数（需子类重写）"""
        return parser

    def name(self):
        """返回模型名称（需子类重写）"""
        return 'BaseModel'

    def initialize(self, opt):
        """初始化模型基础配置"""
        self.opt = opt  # 配置参数
        self.gpu_ids = opt.gpu_ids  # 可用的GPU设备ID列表
        self.isTrain = opt.isTrain  # 训练/测试模式标志
        
        # 张量类型选择（GPU优先）
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        
        # 计算设备选择（GPU优先）
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        
        # 模型保存路径（示例：./checkpoints/experiment_name）
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        
        # 初始化记录容器
        self.loss_names = []    # 损失函数名称列表（如['G', 'D']）
        self.model_names = []   # 子模型名称列表（如['G', 'D']）
        self.visual_names = []  # 可视化内容名称列表（如['real_A', 'fake_B']）
        self.image_paths = []   # 图像路径列表

    def set_input(self, input):
        """设置模型输入数据（需子类重写）"""
        self.input = input

    def forward(self):
        """定义前向传播过程（需子类重写）"""
        pass

    def eval(self):
        """将模型设置为评估模式（关闭BN和Dropout）"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)  # 动态获取子网络
                net.eval()  # 设置eval模式

    def test(self, x):
        """测试模式前向传播（禁用梯度计算）"""
        with torch.no_grad():  # 关闭梯度计算
            self.forward()

    def get_image_paths(self):
        """获取输入图像路径"""
        return self.image_paths

    def optimize_parameters(self, epoch_iter):
        """参数优化步骤（需子类重写）"""
        pass

    def get_current_visuals(self):
        """获取当前可视化结果（用于TensorBoard等）"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                value = getattr(self, name)
                # 处理多输出情况（如GAN的多阶段输出）
                if isinstance(value, list):
                    visual_ret[name] = util.tensor2im(value[-1].data)  # 取最新结果
                else:
                    visual_ret[name] = util.tensor2im(value.data)  # 张量转图像
        return visual_ret

    def get_current_errors(self):
        """获取当前损失值（用于日志记录）"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # 动态获取损失值并转换为Python数值
                errors_ret[name] = getattr(self, 'loss_' + name).item()
        return errors_ret

    def save_networks(self, which_epoch):
        """保存模型权重到磁盘"""
        for name in self.model_names:
            if isinstance(name, str):
                # 构造文件名（如10_netG.pth）
                save_filename = '%s_net%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                
                # 获取网络并保存
                net = getattr(self, 'net' + name)
                torch.save(net.cpu().state_dict(), save_path)  # 转CPU保存
                
                # 恢复GPU状态（如果使用GPU）
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.cuda()

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """修复旧版本InstanceNorm层的状态字典兼容性问题"""
        key = keys[i]
        if i + 1 == len(keys):  # 遍历到参数名末尾
            # 移除无效的running_mean/running_var
            if module.__class__.__name__.startswith('InstanceNorm'):
                if key in ['running_mean', 'running_var'] and getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
                if key == 'num_batches_tracked':
                    state_dict.pop('.'.join(keys))
        else:
            # 递归处理子模块
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def init_with_pretrained_model(self, model_name, pretrained=""):
        """使用预训练权重初始化模型"""
        if not pretrained == " ":
            net = getattr(self, 'net' + model_name)
            # 加载预训练权重
            state_dict = torch.load(pretrained, map_location=str(self.device))
            del state_dict._metadata  # 移除元数据（旧版兼容）
            net.load_state_dict(state_dict, strict=False)  # 非严格模式加载
            print("initialize {} with {}".format(model_name, pretrained))

    def load_networks(self, which_epoch):
        """从磁盘加载模型权重"""
        for name in self.model_names:
            if isinstance(name, str):
                # 构造加载路径（如10_netG.pth）
                load_filename = '%s_net%s.pth' % (which_epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                
                print('loading the model from %s' % load_path)
                net = getattr(self, 'net' + name)
                
                # 加载状态字典
                state_dict = torch.load(load_path, map_location=str(self.device))
                del state_dict._metadata  # 清理元数据
                net.load_state_dict(state_dict)  # 加载参数

    def print_networks(self, verbose):
        """打印网络结构信息"""
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                # 计算参数量
                num_params = sum(param.numel() for param in net.parameters())
                if verbose:  # 详细模式打印网络结构
                    print(net)
                print('[Network %s] Total params: %.3f M' % (name, num_params / 1e6))
        print('---------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """设置网络参数的梯度计算开关"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad  # 冻结/解冻参数
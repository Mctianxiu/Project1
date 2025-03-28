import torch
import functools
from torch import nn
import numpy as np
from models.NEDB_IN import Dense_Block_IN
from models.base_network import BaseNetwork
from models.SpectralNorm import SpectralNorm


# 打印网络结构
def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)


# 打印网络，然后初始化整个网络结构
def init_net(net, gpu_ids):
	#
	print_network(net)
	#
	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())
		net = torch.nn.DataParallel(net, device_ids= gpu_ids)
		net.cuda()
	return net

# 通道注意力
class CALayer(nn.Module):
	def __init__(self, channel):
		super(CALayer, self).__init__()
		out_channel = channel
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
		#
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ca = nn.Sequential(
				nn.Conv2d(out_channel, out_channel // 8, 1, padding=0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(out_channel // 8, channel, 1, padding=0, bias=True),
				nn.Sigmoid()
		)

	def forward(self, x):
		t1 = self.conv1(x)  # in
		t2 = self.relu(t1)  # in, 64
		y = self.avg_pool(t2)  # torch.Size([1, in, 1, 1])
		y = self.ca(y)  # torch.Size([1, in, 1, 1])
		m = t2 * y      # torch.Size([1, in, 64, 64]) * torch.Size([1, in, 1, 1])
		return x + m

# 空间注意力
class PALayer(nn.Module):
	def __init__(self, channel):
		super(PALayer, self).__init__()
		self.pa = nn.Sequential(
				nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
				nn.Sigmoid()
		)

	def forward(self, x):
		y = self.pa(x)
		return x * y


# 定义上采样
class Trans_Up(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Up, self).__init__()
		self.conv0 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out


# 定义下采样
class Trans_Down(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Down, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out


#======================================================================
# 定义cyclegan生成器
# input_nc：输入图像的通道数，例如 RGB 图像的通道数为 3。
# output_nc：输出图像的通道数，同样，RGB 图像的通道数为 3。
# which_model_netG：一个字符串类型的参数，用于指定要使用的生成器网络类型。
#======================================================================
def define_G(input_nc, output_nc, which_model_netG):
	netG = None
	if which_model_netG == 'unet':
		netG = NLEDN_IN_32_16_32(input_nc, output_nc)
	if which_model_netG == "Correction":
		netG = Self_Correction(input_nc, output_nc)
	if which_model_netG == "Correction_Post":
		netG = Self_Correction_Post(input_nc, output_nc)
	return netG
#================================
# 翻译模块（自校正模块）
#================================
class Self_Correction(nn.Module):
	def __init__(self, input_nc, output_nc):
		super(Self_Correction, self).__init__()
		# 初始卷积层
		self.conv1 = nn.Sequential(
			nn.Conv2d(input_nc, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		
		# 密集残差块（DenseBlock）
		self.conv2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.conv3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 128 -> 64
		self.conv4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 64 -> 32
		# 输出融合层
		self.fusion = nn.Sequential(
			nn.Conv2d(64, output_nc, 3, 1, 1),
			nn.Tanh(),				    # 输出归一化到 [-1, 1]
		)
		#

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x31 = x3 - x1        			# 减法残差操作---------------通过减法抑制源域特有的风格信息，保留跨域共享的内容特征
		x4 = self.conv4(x31)
		x41 = x2 + x4   				# 跳跃连接
		outputs = self.fusion(x41)
		return outputs

# 自校正模块（代码同上相同，可能是不同阶段的组件或代码重复）---------------可能作为增强模块 G 的后处理单元
class Self_Correction_Post(nn.Module):
	def __init__(self, input_nc, output_nc):
		super(Self_Correction_Post, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(input_nc, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		# 几个 conv, 中间 channel, 输入 channel
		self.conv2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  	# 256 -> 128
		self.conv3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   	# 128 -> 64
		self.conv4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   	# 64 -> 32
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, output_nc, 3, 1, 1),
			nn.Tanh(),
		)
		#

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x31 = x3 - x1					# 减法残差操作---------------通过减法抑制源域特有的风格信息，保留跨域共享的内容特征
		x4 = self.conv4(x31)
		x41 = x2 + x4					# 跳跃连接（输入数据直接添加到某一层的输出上面）
		outputs = self.fusion(x41)
		return outputs

#============================================================================================
# 域间自适应的增强模块
# NLEDN 的核心设计,​编码器-解码器架构
# 一，
# 1,编码器（Encoder）​：通过下采样提取多尺度特征，包含多个 DenseBlock。
# 2,​解码器（Decoder）​：通过上采样重建图像，并与编码器通过跳跃连接（Skip Connections）融合特征。
# 3,​潜在层（Latent Layer）​：位于编码器和解码器之间，用于特征对齐（域适应的关键）
# 二，
# 密集连接块（DenseBlock）​------------ 类似 DenseNet，每个层的输出会与后续所有层的输入拼接，促进特征复用
# 三， 
# ​注意力机制
# 1，通道注意力（CALayer）​：动态调整通道权重，抑制无关特征
# 2，空间注意力（PALayer）​：聚焦关键空间区域，增强跨域一致性
# 四，
# ​跳跃连接（Skip Connections）​-----------将编码器的多尺度特征与解码器融合，避免信息丢失。
# 在跳跃连接中插入 CALayer 和 PALayer，实现注意力引导的特征融合
#============================================================================================
class NLEDN_IN_32_16_32(nn.Module):  				# 类U-Net结构
	def __init__(self, input_nc, output_nc):
		super(NLEDN_IN_32_16_32, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(input_nc, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		# 上采样----------这些是普通的卷积层，用于初步特征提取。可能属于增强模块的一部分，处理输入图像的基本特征。
		self.up_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)  # 256 -> 128
		self.up_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 128 -> 64
		self.up_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 64 -> 32
		self.up_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 32 -> 16
		# 潜在层---------可能位于U-Net的瓶颈部分，处理最深层特征，属于增强模块的核心处理部分
		self.Latent = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)   # 16 -> 8
		# 下采样--------同样使用Dense_Block_IN，可能对应U-Net的解码器部分，逐步重建图像，属于增强模块的重建阶段。
		self.down_4 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 16
		self.down_3 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 32
		self.down_2 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 64
		self.down_1 = Dense_Block_IN(block_num=3, inter_channel=32, channel=64)     # 128
		# 通道注意力层-------------用于特征校准，可能增强模块中的特征对齐，帮助模型关注重要通道信息，可能用于特征级对齐。
		self.CALayer4 = CALayer(128)
		self.CALayer3 = CALayer(128)
		self.CALayer2 = CALayer(128)
		self.CALayer1 = CALayer(128)
		# 上下采样的转换层（trans_down和trans_up）----------同样使用Dense_Block_IN，可能对应U-Net的解码器部分，逐步重建图像，属于增强模块的重建阶段。
		self.trans_down1 = Trans_Down(64, 64)
		self.trans_down2 = Trans_Down(64, 64)
		self.trans_down3 = Trans_Down(64, 64)
		self.trans_down4 = Trans_Down(64, 64)
		
		self.trans_up4 = Trans_Up(64, 64)
		self.trans_up3 = Trans_Up(64, 64)
		self.trans_up2 = Trans_Up(64, 64)
		self.trans_up1 = Trans_Up(64, 64)
		# 融合层（down_4_fusion到down_1_fusion）---------将不同层次的特征进行融合，可能用于整合编码器和解码器的特征，增强模块的特征融合部分。
		self.down_4_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.down_3_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.down_2_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		self.down_1_fusion = nn.Sequential(
			nn.Conv2d(64 + 64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		# ​最终的融合层（fusion和fusion2）-----------将处理后的特征转换为输出图像，可能对应增强模块的最终输出部分，生成增强后的图像。
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, output_nc, 3, 1, 1),
			nn.Tanh(),
		)
		#

	def forward(self, x):   # 1, 3, 256, 256
		#
		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
		#######################################################
		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128

		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64

		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32

		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16

		#######################################################
		Latent = self.Latent(up_4)  # 1, 64, 16, 16
		#######################################################

		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
		down_4 = torch.cat([up_41, down_4], dim=1)  # 1, 128, 32, 32
		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
		down_4 = self.down_4(down_4)       # 1, 64, 32, 32

		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
		down_3 = torch.cat([up_31, down_3], dim=1)  # 1, 128, 64, 64
		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
		down_3 = self.down_3(down_3)       # 1, 64, 64, 64

		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
		down_2 = torch.cat([up_21, down_2], dim=1)  # 1, 128, 128,128
		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
		down_2 = self.down_2(down_2)       # 1, 64, 128,128

		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
		down_1 = torch.cat([up_11, down_1], dim=1)  # 1, 128, 256, 256
		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
		#
		feature = self.fusion(down_1)  # 1, 64, 256, 256
		#
		feature = feature + feature_neg_1  # 1, 64, 256, 256
		#
		outputs = self.fusion2(feature)
		return outputs, Latent


# 设置中间层的加速层
def get_norm_layer(norm_type='instance'):
	if norm_type == 'batch':
		norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
	elif norm_type == 'instance':
		norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
	else:
		raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
	return norm_layer


# 定义鉴别器
def define_D(input_nc, ndf, which_model_netD, n_layers_D=5, norm='batch'):
	netD = None
	norm_layer = get_norm_layer(norm_type=norm)
	#
	if which_model_netD == 'basic':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers=5, norm_layer=norm_layer)
	elif which_model_netD == 'n_layers':
		netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
	return netD


# 用指定的参数定义PatchGAN鉴别器。
class NLayerDiscriminator(nn.Module):
	def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
		super(NLayerDiscriminator, self).__init__()
		#
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d
		else:
			use_bias = norm_layer == nn.InstanceNorm2d

		kw = 4
		padw = int(np.ceil((kw-1)/2))
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
		]

		nf_mult = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2**n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
				norm_layer(ndf * nf_mult),
				nn.LeakyReLU(0.2, True)
			]
		nf_mult_prev = nf_mult
		nf_mult = min(2**n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
					  kernel_size=kw, stride=1, padding=padw, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
		]

		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		return self.model(input)



def define_featureD(input_nc, n_layers=2):
	net = _FeatureDiscriminator(input_nc, n_layers)
	return net

# 定义特征鉴别器
class _FeatureDiscriminator(nn.Module):
	def __init__(self, input_nc, n_layers=2):
		super(_FeatureDiscriminator, self).__init__()

		model = [
			nn.Linear(input_nc * 16 * 16, input_nc),
			nn.LeakyReLU(0.2, True),
		]

		for i in range(1, n_layers):
			model +=[
				nn.Linear(input_nc, input_nc),
				nn.LeakyReLU(0.2, True)
			]

		model += [nn.Linear(input_nc, 1)]

		self.model = nn.Sequential(*model)

	def forward(self, input):
		input = input.view(-1, 64 * 16 * 16)
		output = self.model(input)
		return output



def define_SpectralD(input_nc, n_channels):
	net = SpectralDiscriminator(input_nc, n_channels)
	return net

# 谱归一化判别器
class SpectralDiscriminator(BaseNetwork):
	def __init__(self, input_nc=3, n_channels=64, downsample_num=0):
		super(SpectralDiscriminator, self).__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(input_nc, n_channels, kernel_size=4, stride=2, padding=2),
			nn.LeakyReLU(0.2, True),
		)
		self.layer2 = nn.Sequential(
			SpectralNorm(nn.Conv2d(n_channels, n_channels * 2, kernel_size=4, stride=2, padding=2)),
			nn.LeakyReLU(0.2, True)
		)
		self.layer3 = nn.Sequential(
			SpectralNorm(nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=4, stride=2, padding=2)),
			nn.LeakyReLU(0.2, True),
		)
		self.layer4 = nn.Sequential(
			SpectralNorm(nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=4, stride=1, padding=2)),
			nn.LeakyReLU(0.2, True),
			nn.Conv2d(n_channels * 8, 1, kernel_size=4, stride=1, padding=2)
		)
		#
		self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
		self.is_downsample = downsample_num

	def forward(self, x):
		if self.is_downsample == 1:
			x = self.downsample(x)
		elif self.is_downsample == 2:
			x = self.downsample(self.downsample(x))

		res1 = self.layer1(x)
		res2 = self.layer2(res1)
		res3 = self.layer3(res2)
		res4 = self.layer4(res3)
		return res1, res2, res3, res4


if __name__ == "__main__":
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	device_ids = [0]
	#
	x = torch.randn(1, 3, 128, 128)
	x = x.cuda()
	#
	net = NLEDN_IN_32_16_32()
	# print(net)
	pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
	print("Total_params: {}".format(pytorch_total_params))
	net = init_net(net, device_ids)
	output = net(x)
	# print(output)
	print("ok~")

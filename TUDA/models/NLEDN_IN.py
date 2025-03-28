import torch
from torch import nn
from models.NEDB_IN import Dense_Block_IN


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


class Trans_Up(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Up, self).__init__()
		self.conv0 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out


class Trans_Down(nn.Module):
	def __init__(self, in_planes, out_planes):
		super(Trans_Down, self).__init__()
		self.conv0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1)
		self.IN1 = nn.InstanceNorm2d(out_planes)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.relu(self.IN1(self.conv0(x)))
		return out


# class Prior_Guide(nn.Module):
# 	def __init__(self, feature_channel, Prior_Map_channel,):
# 		super(Prior_Guide, self).__init__()
# 		#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(Prior_Map_channel, 64, kernel_size=5, stride=1, padding=2),
# 			nn.ReLU(),
# 		)
#
# 		scale_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
# 		scale_conv = [scale_conv2]
# 		self.scale_conv = nn.Sequential(*scale_conv)
#
# 		sift_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
# 		sift_conv = [sift_conv2]
# 		self.sift_conv = nn.Sequential(*sift_conv)
#
# 		self.trans_down = Trans_Down(64, 64)
# 		#
# 		self.Nrom = nn.Sequential(
# 			nn.InstanceNorm2d(feature_channel, affine=False),
# 		)
#
# 	def forward(self, feature_map, Prior_Map):
# 		# 这边 feature_map 与 prior_map 同样大小
# 		t_num = int(Prior_Map.shape[2] / feature_map.shape[2])  # 2
# 		if t_num == 2:
# 			Prior_Map = self.trans_down(Prior_Map)  # 1, 3, 128, 128;
# 		else:
# 			Prior_Map = Prior_Map
# 		feature_map_norm = self.Nrom(feature_map)  # 1, 64, 128, 128
# 		Prior_Map = self.conv1(Prior_Map)
# 		sifted_feature = feature_map_norm * (1 + self.scale_conv(Prior_Map)) + self.sift_conv(Prior_Map)
# 		return sifted_feature
#
#
# class Prior_Guide_add(nn.Module):
# 	def __init__(self, feature_channel, Prior_Map_channel,):
# 		super(Prior_Guide_add, self).__init__()
# 		self.trans_down = Trans_Down(64, 64)
# 		#
#
# 	def forward(self, feature_map, Prior_Map):
# 		# 这边 feature_map 与 prior_map 同样大小
# 		t_num = int(Prior_Map.shape[2] / feature_map.shape[2])  # 2
# 		if t_num == 2:
# 			Prior_Map = self.trans_down(Prior_Map)  # 1, 3, 128, 128;
# 		else:
# 			Prior_Map = Prior_Map
# 		map = feature_map + Prior_Map
# 		return map


class NLEDN_IN_32_16_32(nn.Module):
	def __init__(self):
		super(NLEDN_IN_32_16_32, self).__init__()

		self.conv1 = nn.Sequential(
			nn.Conv2d(3, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		#
		self.conv2 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
		)
		# 几个 conv, 中间 channel, 输入 channel
		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
		#
		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16 -> 8
		#
		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
		#
		self.CALayer4 = CALayer(128)
		self.CALayer3 = CALayer(128)
		self.CALayer2 = CALayer(128)
		self.CALayer1 = CALayer(128)
		#
		self.trans_down1 = Trans_Down(64, 64)
		self.trans_down2 = Trans_Down(64, 64)
		self.trans_down3 = Trans_Down(64, 64)
		self.trans_down4 = Trans_Down(64, 64)
		#
		self.trans_up4 = Trans_Up(64, 64)
		self.trans_up3 = Trans_Up(64, 64)
		self.trans_up2 = Trans_Up(64, 64)
		self.trans_up1 = Trans_Up(64, 64)
		#
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
		#
		self.fusion = nn.Sequential(
			nn.Conv2d(64, 64, 1, 1, 0),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU()
		)
		self.fusion2 = nn.Sequential(
			nn.Conv2d(64, 3, 3, 1, 1),
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


# class NLEDN_IN_Prior_A_3_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_A_3_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class NLEDN_IN_32_16_32_Add_A_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16 -> 8
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_3_5()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y):   # 1, 3, 256, 256
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(y)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_Class_1_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_Class_1_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(1, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 16
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class NLEDN_IN_32_16_32_Add_Class_In_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_Class_In_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16 -> 8
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_Class_1_5()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		# self.Prior_Guide1_de = Prior_Guide(64, 64)
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y):   # 1, 3, 256, 256
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(y)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_Edge_2_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_Edge_2_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(2, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class NLEDN_IN_32_16_32_Add_Edge_In_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_Edge_In_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16 -> 8
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_Edge_2_5()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		# self.Prior_Guide1_de = Prior_Guide(64, 64)
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y):   # 1, 3, 256, 256
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(y)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_Depth_1_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_Depth_1_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(1, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 16
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class NLEDN_IN_32_16_32_Add_Depth_In_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_Depth_In_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16 -> 8
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_Depth_1_5()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		# self.Prior_Guide1_de = Prior_Guide(64, 64)
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide1_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y):   # 1, 3, 256, 256
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(y)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide4_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide2_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide1_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_A_Class_4_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_A_Class_4_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(16 + 16, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class Multi_2Conv_Prior_Norm(nn.Module):
# 	def __init__(self):
# 		super(Multi_2Conv_Prior_Norm, self).__init__()
# 		#
# 		self.convt1 = nn.Sequential(
# 			nn.Conv2d(3, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.convt2 = nn.Sequential(
# 			nn.Conv2d(1, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(32, 64, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 32, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.AdaptiveAvgPool2d(1),
# 			#
# 			nn.Conv2d(32, 16, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(16, 8, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(8, 2, 1, 1, 0),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, t1, t2):  # 1, 3, 256, 256
# 		t11 = self.convt1(t1)
# 		t12 = self.convt2(t2)
# 		x1 = torch.cat([t11, t12], dim=1)  # torch.Size([1, 9, 256, 256])
# 		#
# 		w0 = self.conv1(x1)  # 1, 3, 1, 1
# 		w111, w112 = torch.split(w0, [1, 1], dim=1)
# 		#
# 		w11 = w111 / (w111 + w112)
# 		w12 = w112 / (w111 + w112)
# 		#
# 		output = torch.cat([t11 * w11, t12 * w12], dim=1)
# 		#
# 		return output
#
#
# class NLEDN_IN_32_16_32_Add_A_Class_In_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_Class_In_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_Class_4_5()
# 		self.prior_2_guided_weight = Multi_2Conv_Prior_Norm()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y, y_class):   # 1, 3, 256, 256
# 		t_prior = self.prior_2_guided_weight(y, y_class)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_A_Class_4_5_gate(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_A_Class_4_5_gate, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3 + 1, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class Multi_2Conv_Prior_Norm_gate(nn.Module):
# 	def __init__(self):
# 		super(Multi_2Conv_Prior_Norm_gate, self).__init__()
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3 + 1, 64, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 32, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(32, 32, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(32, 32, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(32, 16, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(16, 2, 1, 1, 0),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, t1, t2):  # 1, 3, 256, 256
# 		x1 = torch.cat([t1, t2], dim=1)  # torch.Size([1, 9, 256, 256])
# 		#
# 		w0 = self.conv1(x1)  # 1, 3, 1, 1
# 		w11, w12 = torch.split(w0, [1, 1], dim=1)
# 		#
# 		output = torch.cat([t1 * w11, t2 * w12], dim=1)
# 		#
# 		return output
#
#
# class NLEDN_IN_32_16_32_Add_A_Class_In_En_De_gate(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_Class_In_En_De_gate, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_Class_4_5_gate()
# 		self.prior_2_guided_weight = Multi_2Conv_Prior_Norm_gate()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y, y_class):   # 1, 3, 256, 256
# 		t_prior = self.prior_2_guided_weight(y, y_class)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_A_Class_4_5_concat(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_A_Class_4_5_concat, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3 + 1, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class Multi_2Conv_Prior_Norm_concat(nn.Module):
# 	def __init__(self):
# 		super(Multi_2Conv_Prior_Norm_concat, self).__init__()
# 		self.conv1 = nn.Sequential(
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, t1, t2):  # 1, 3, 256, 256
# 		x1 = torch.cat([t1, t2], dim=1)  # torch.Size([1, 9, 256, 256])
# 		#
# 		return x1
#
#
# class NLEDN_IN_32_16_32_Add_A_Class_In_En_De_concat(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_Class_In_En_De_concat, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_Class_4_5_concat()
# 		self.prior_2_guided_weight = Multi_2Conv_Prior_Norm_concat()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y, y_class):   # 1, 3, 256, 256
# 		t_prior = self.prior_2_guided_weight(y, y_class)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_A_Edge_5_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_A_Edge_5_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(16 + 16, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class Multi_2Conv_3_2_Prior_Norm(nn.Module):
# 	def __init__(self):
# 		super(Multi_2Conv_3_2_Prior_Norm, self).__init__()
# 		#
# 		self.convt1 = nn.Sequential(
# 			nn.Conv2d(3, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.convt2 = nn.Sequential(
# 			nn.Conv2d(2, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(32, 64, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 32, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.AdaptiveAvgPool2d(1),
# 			#
# 			nn.Conv2d(32, 16, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(16, 8, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(8, 2, 1, 1, 0),
# 			nn.Sigmoid()
# 		)
# 		#
#
# 	def forward(self, t1, t2):   # 1, 3, 256, 256
# 		t11 = self.convt1(t1)
# 		t12 = self.convt2(t2)
# 		x1 = torch.cat([t11, t12], dim=1)		# torch.Size([1, 9, 256, 256])
# 		#
# 		w0 = self.conv1(x1)  # 1, 3, 1, 1
# 		w111, w112 = torch.split(w0, [1, 1], dim=1)
# 		w11 = w111 / (w111 + w112)
# 		w12 = w112 / (w111 + w112)
# 		#
# 		output = torch.cat([t11 * w11, t12 * w12], dim=1)
# 		#
# 		return output
#
#
# class NLEDN_IN_32_16_32_Add_A_Edge_In_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_Edge_In_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_Edge_5_5()
# 		self.prior_2_guided_weight = Multi_2Conv_3_2_Prior_Norm()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y, y_Edge):   # 1, 3, 256, 256
# 		t_prior = self.prior_2_guided_weight(y, y_Edge)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class Multi_2Conv_2_1_Prior_Norm(nn.Module):
# 	def __init__(self):
# 		super(Multi_2Conv_2_1_Prior_Norm, self).__init__()
# 		#
# 		self.convt1 = nn.Sequential(
# 			nn.Conv2d(1, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.convt2 = nn.Sequential(
# 			nn.Conv2d(2, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(32, 64, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 32, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.AdaptiveAvgPool2d(1),
# 			#
# 			nn.Conv2d(32, 16, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(16, 8, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(8, 2, 1, 1, 0),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, t1, t2):  # 1, 3, 256, 256
# 		t11 = self.convt1(t1)
# 		t12 = self.convt2(t2)
# 		x1 = torch.cat([t11, t12], dim=1)  # torch.Size([1, 9, 256, 256])
# 		#
# 		w0 = self.conv1(x1)  # 1, 3, 1, 1
# 		w111, w112 = torch.split(w0, [1, 1], dim=1)
# 		w11 = w111 / (w111 + w112)
# 		w12 = w112 / (w111 + w112)
# 		#
# 		output = torch.cat([t11 * w11, t12 * w12], dim=1)
# 		#
# 		return output
#
#
# class NLEDN_IN_Prior_Class_Edge_3_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_Class_Edge_3_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(16 + 16, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class NLEDN_IN_32_16_32_Add_Class_Edge_In_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_Class_Edge_In_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_Class_Edge_3_5()
# 		self.prior_2_guided_weight = Multi_2Conv_2_1_Prior_Norm()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y_class, y_Edge):   # 1, 3, 256, 256
# 		t_prior = self.prior_2_guided_weight(y_class, y_Edge)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class Multi_2Conv_2_1_Prior_Norm_gate(nn.Module):
# 	def __init__(self):
# 		super(Multi_2Conv_2_1_Prior_Norm_gate, self).__init__()
# 		#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(1+2, 64, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 32, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(32, 32, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(32, 32, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(32, 16, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(16, 2, 1, 1, 0),
# 			nn.Sigmoid()
# 		)
#
# 	def forward(self, t1, t2):  # 1, 3, 256, 256
# 		x1 = torch.cat([t1, t2], dim=1)  # torch.Size([1, 9, 256, 256])
# 		#
# 		w0 = self.conv1(x1)  # 1, 3, 1, 1
# 		w11, w12 = torch.split(w0, [1, 1], dim=1)
# 		#
# 		output = torch.cat([t1 * w11, t2 * w12], dim=1)
# 		#
# 		return output
#
#
# class NLEDN_IN_Prior_Class_Edge_3_5_gate(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_Class_Edge_3_5_gate, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(1 + 2, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class NLEDN_IN_32_16_32_Add_Class_Edge_In_En_De_gate(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_Class_Edge_In_En_De_gate, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_Class_Edge_3_5_gate()
# 		self.prior_2_guided_weight = Multi_2Conv_2_1_Prior_Norm_gate()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y_class, y_Edge):   # 1, 3, 256, 256
# 		t_prior = self.prior_2_guided_weight(y_class, y_Edge)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_A_6_5(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_A_6_5, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(16 + 16 + 16, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class Multi_3Conv_3_1_2_Prior_Norm(nn.Module):
# 	def __init__(self):
# 		super(Multi_3Conv_3_1_2_Prior_Norm, self).__init__()
# 		#
# 		self.convt1 = nn.Sequential(
# 			nn.Conv2d(3, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.convt2 = nn.Sequential(
# 			nn.Conv2d(1, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.convt3 = nn.Sequential(
# 			nn.Conv2d(2, 16, 1, 1, 0),
# 			nn.ReLU(),  # torch.Size([1, 128, 256, 256])
# 		)
# 		#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(32 + 16, 64, 7, 1, 3),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 5, 1, 2),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(64, 32, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.AdaptiveAvgPool2d(1),
# 			#
# 			nn.Conv2d(32, 16, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(16, 8, 1, 1, 0),
# 			nn.ReLU(),
# 			#
# 			nn.Conv2d(8, 3, 1, 1, 0),
# 			nn.Sigmoid()
# 		)
# 		#
#
# 	def forward(self, t1, t2, t3):   # 1, 3, 256, 256
# 		t11 = self.convt1(t1)
# 		t12 = self.convt2(t2)
# 		t13 = self.convt3(t3)
# 		x1 = torch.cat([t11, t12, t13], dim=1)		# torch.Size([1, 9, 256, 256])
# 		#
# 		w0 = self.conv1(x1)  # 1, 3, 1, 1
# 		w111, w112, w113 = torch.split(w0, [1, 1, 1], dim=1)
# 		#
# 		w11 = w111 / (w111 + w112 + w113)
# 		w12 = w112 / (w111 + w112 + w113)
# 		w13 = w113 / (w111 + w112 + w113)
# 		#
# 		output = torch.cat([t11 * w11, t12 * w12, t13 * w13], dim=1)
# 		#
# 		return output
#
#
# class NLEDN_IN_32_16_32_Add_A_Class_Edge_In_En_De(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_Class_Edge_In_En_De, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_6_5()
# 		self.prior_3_guided_weight = Multi_3Conv_3_1_2_Prior_Norm()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y, y_class, y_Edge):   # 1, 3, 256, 256
# 		t_prior = self.prior_3_guided_weight(y, y_class, y_Edge)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_32_16_32_Add_A_Class_Edge_In_En_De_add(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_Class_Edge_In_En_De_add, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_6_5()
# 		self.prior_3_guided_weight = Multi_3Conv_3_1_2_Prior_Norm()
# 		#
# 		self.Prior_Guide1 = Prior_Guide_add(64, 64)
# 		self.Prior_Guide2 = Prior_Guide_add(64, 64)
# 		self.Prior_Guide3 = Prior_Guide_add(64, 64)
# 		self.Prior_Guide4 = Prior_Guide_add(64, 64)
# 		self.Prior_Guide5 = Prior_Guide_add(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide_add(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide_add(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide_add(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide_add(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y, y_class, y_Edge):   # 1, 3, 256, 256
# 		t_prior = self.prior_3_guided_weight(y, y_class, y_Edge)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs
#
#
# class NLEDN_IN_Prior_A_6_5_Concat(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_Prior_A_6_5_Concat, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3+1+2, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 16 -> 8
#
# 	def forward(self, x):  # 1, 3, 256, 256
# 		#
# 		feature_neg_1 = self.conv1(x)  # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)  # # 1, 64, 256, 256
# 		#######################################################
# 		up_11 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_1 = self.trans_down1(up_11)  # 1, 64, 128, 128
#
# 		up_21 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_2 = self.trans_down2(up_21)  # 1, 64, 64, 64
#
# 		up_31 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_3 = self.trans_down3(up_31)  # 1, 64, 32, 32
#
# 		up_41 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_4 = self.trans_down4(up_41)  # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		#######################################################
# 		return Latent, up_41, up_31, up_21, up_11  # 16  32 64 128 256  channel:64
#
#
# class Multi_3Conv_3_1_2_Prior_Norm_Concat(nn.Module):
# 	def __init__(self):
# 		super(Multi_3Conv_3_1_2_Prior_Norm_Concat, self).__init__()
# 		#
# 		self.convt1 = nn.Sequential(
# 			nn.Conv2d(3, 16, 1, 1, 0),
# 		)
#
# 	def forward(self, t1, t2, t3):   # 1, 3, 256, 256
# 		x1 = torch.cat([t1, t2, t3], dim=1)		# torch.Size([1, 9, 256, 256])
# 		return x1
#
#
# class NLEDN_IN_32_16_32_Add_A_Class_Edge_In_En_De_Concat(nn.Module):
# 	def __init__(self):
# 		super(NLEDN_IN_32_16_32_Add_A_Class_Edge_In_En_De_Concat, self).__init__()
#
# 		self.conv1 = nn.Sequential(
# 			nn.Conv2d(3, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.conv2 = nn.Sequential(
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		# 几个 conv, 中间 channel, 输入 channel
# 		self.up_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)  # 256 -> 128
# 		self.up_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 128 -> 64
# 		self.up_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 64 -> 32
# 		self.up_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 32 -> 16
# 		#
# 		self.Latent = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)   # 16
# 		#
# 		self.down_4 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 16
# 		self.down_3 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 32
# 		self.down_2 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 64
# 		self.down_1 = Dense_Block_IN(block_num=4, inter_channel=32, channel=64)     # 128
# 		#
# 		self.CALayer4 = CALayer(128)
# 		self.CALayer3 = CALayer(128)
# 		self.CALayer2 = CALayer(128)
# 		self.CALayer1 = CALayer(128)
# 		#
# 		self.trans_down1 = Trans_Down(64, 64)
# 		self.trans_down2 = Trans_Down(64, 64)
# 		self.trans_down3 = Trans_Down(64, 64)
# 		self.trans_down4 = Trans_Down(64, 64)
# 		#
# 		self.trans_up4 = Trans_Up(64, 64)
# 		self.trans_up3 = Trans_Up(64, 64)
# 		self.trans_up2 = Trans_Up(64, 64)
# 		self.trans_up1 = Trans_Up(64, 64)
# 		#
# 		self.prior = NLEDN_IN_Prior_A_6_5_Concat()
# 		self.prior_3_guided_weight = Multi_3Conv_3_1_2_Prior_Norm_Concat()
# 		#
# 		self.Prior_Guide1 = Prior_Guide(64, 64)
# 		self.Prior_Guide2 = Prior_Guide(64, 64)
# 		self.Prior_Guide3 = Prior_Guide(64, 64)
# 		self.Prior_Guide4 = Prior_Guide(64, 64)
# 		self.Prior_Guide5 = Prior_Guide(64, 64)
# 		#
# 		self.Prior_Guide2_de = Prior_Guide(64, 64)
# 		self.Prior_Guide3_de = Prior_Guide(64, 64)
# 		self.Prior_Guide4_de = Prior_Guide(64, 64)
# 		self.Prior_Guide5_de = Prior_Guide(64, 64)
# 		#
# 		self.down_4_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_3_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_2_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		self.down_1_fusion = nn.Sequential(
# 			nn.Conv2d(64 + 64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 		)
# 		#
# 		self.fusion = nn.Sequential(
# 			nn.Conv2d(64, 64, 1, 1, 0),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU(),
# 			nn.Conv2d(64, 64, 3, 1, 1),
# 			nn.InstanceNorm2d(64, affine=True),
# 			nn.ReLU()
# 		)
# 		self.fusion2 = nn.Sequential(
# 			nn.Conv2d(64, 3, 3, 1, 1),
# 			nn.Tanh(),
# 		)
# 		#
#
# 	def forward(self, x, y, y_class, y_Edge):   # 1, 3, 256, 256
# 		t_prior = self.prior_3_guided_weight(y, y_class, y_Edge)
# 		#
# 		y_1, y_2, y_3, y_4, y_5 = self.prior(t_prior)  # 16,32,64,128,256
# 		#
# 		feature_neg_1 = self.conv1(x)   # 1, 64, 256, 256
# 		feature_0 = self.conv2(feature_neg_1)   # # 1, 64, 256, 256
# 		#######################################################
# 		up_111 = self.up_1(feature_0)  # 1, 64, 256, 256
# 		up_11 = self.Prior_Guide1(up_111, y_5)
# 		up_1 = self.trans_down1(up_11)   # 1, 64, 128, 128
#
# 		up_211 = self.up_2(up_1)  # 1, 64, 128, 128
# 		up_21 = self.Prior_Guide2(up_211, y_4)
# 		up_2 = self.trans_down2(up_21)    # 1, 64, 64, 64
#
# 		up_311 = self.up_3(up_2)  # 1, 64, 64, 64
# 		up_31 = self.Prior_Guide3(up_311, y_3)
# 		up_3 = self.trans_down3(up_31)    # 1, 64, 32, 32
#
# 		up_411 = self.up_4(up_3)  # 1, 64, 32, 32
# 		up_41 = self.Prior_Guide4(up_411, y_2)
# 		up_4 = self.trans_down4(up_41)    # 1, 64, 16, 16
#
# 		#######################################################
# 		Latent = self.Latent(up_4)  # 1, 64, 16, 16
# 		Latent = self.Prior_Guide5(Latent, y_1)
# 		#######################################################
#
# 		down_4 = self.trans_up4(Latent)  # 1, 64, 32, 32
# 		down_4 = torch.cat([up_411, down_4], dim=1)  # 1, 128, 32, 32
# 		down_41 = self.CALayer4(down_4)     # 1, 128, 32, 32
# 		down_4 = self.down_4_fusion(down_41)     # 1, 64, 32, 32
# 		down_4 = self.Prior_Guide2_de(down_4, y_2)
# 		down_4 = self.down_4(down_4)       # 1, 64, 32, 32
#
# 		down_3 = self.trans_up3(down_4)  # 1, 64, 64, 64
# 		down_3 = torch.cat([up_311, down_3], dim=1)  # 1, 128, 64, 64
# 		down_31 = self.CALayer3(down_3)     # 1, 128, 64, 64
# 		down_3 = self.down_3_fusion(down_31)     # 1, 64, 64, 64
# 		down_3 = self.Prior_Guide3_de(down_3, y_3)
# 		down_3 = self.down_3(down_3)       # 1, 64, 64, 64
#
# 		down_2 = self.trans_up2(down_3)  # 1, 64, 128,128
# 		down_2 = torch.cat([up_211, down_2], dim=1)  # 1, 128, 128,128
# 		down_21 = self.CALayer2(down_2)     # 1, 128, 128,128
# 		down_2 = self.down_2_fusion(down_21)     # 1, 64, 128,128
# 		down_2 = self.Prior_Guide4_de(down_2, y_4)
# 		down_2 = self.down_2(down_2)       # 1, 64, 128,128
#
# 		down_1 = self.trans_up1(down_2)  # 1, 64, 256, 256
# 		down_1 = torch.cat([up_111, down_1], dim=1)  # 1, 128, 256, 256
# 		down_11 = self.CALayer1(down_1)     # 1, 128, 256, 256
# 		down_1 = self.down_1_fusion(down_11)     # 1, 64, 256, 256
# 		down_1 = self.Prior_Guide5_de(down_1, y_5)
# 		down_1 = self.down_1(down_1)       # 1, 64, 256, 256
# 		#
# 		feature = self.fusion(down_1)  # 1, 64, 256, 256
# 		#
# 		feature = feature + feature_neg_1  # 1, 64, 256, 256
# 		#
# 		outputs = self.fusion2(feature)
# 		return outputs


if __name__ == "__main__":
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
	device_ids = [0, 1]
	#
	net = Prior_Guide(64, 64).cuda()
	#
	if torch.cuda.is_available():
		net = torch.nn.DataParallel(net, device_ids=device_ids)
		net.cuda()
	# net = Dense_rain()
	print("Params:", sum(param.numel() for param in net.parameters()))
	input0 = torch.FloatTensor(4, 64, 256, 256).cuda()
	input1 = torch.FloatTensor(4, 64, 256, 256).cuda()
	# input2 = torch.FloatTensor(1, 1, 256, 256).cuda()
	# input3 = torch.FloatTensor(1, 2, 256, 256).cuda()
	t = net(input0, input1)
	print("ok~")
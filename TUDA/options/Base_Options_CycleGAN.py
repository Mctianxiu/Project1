import argparse
from util import util
import models
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]


class BaseOptions_CycleGAN():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_inter_intra',
							help='models are saved here')
		parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
		#
		parser.add_argument('--dataset_Name', default='')
		parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
		#
		parser.add_argument('--imageSize', type=int, default=256, help='then crop to this size')
		#
		parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
		parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
		# 一般生成器和鉴别器的第一层都是64通道
		parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
		parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
		#
		parser.add_argument('--which_model_netG_A', type=str, default='unet',
							help='selects model to use for netG_A')
		parser.add_argument('--which_model_netG_B', type=str, default='unet',
							help='selects model to use for netG_B')
		# ############################ gan_type为wgan的话，norm要为 instance ############################
		parser.add_argument('--gan_type', type=str, default='wgan-gp',
							help='wgan-gp, gan, hinge-gp')
		parser.add_argument('--norm', type=str, default='instance',
							help='batch normalization or instance normalization')
		# ############################ gan_type为wgan的话，norm要为 instance ############################
		parser.add_argument('--no_dropout', type=str, default=False)
		parser.add_argument('--use_parallel', type=str, default=True)
		parser.add_argument('--gpu_ids', type=str, default='0,1',
							help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		# 鉴别器的model, 生成器的model, 鉴别器层数
		parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model for netD')
		parser.add_argument('--n_layers_D', type=int, default=5, help='only used if which_model_netD==n_layers')
		#
		parser.add_argument('--name', type=str, default='run_singlegan',
							help='name of the experiment. It decides where to store samples and models')
		parser.add_argument('--dataset_mode', type=str, default='aligned',
							help='chooses how datasets are loaded. [unaligned | aligned | single]')
		parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
		parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
		#
		parser.add_argument('--serial_batches', action='store_true',
							help='if true, takes images in order to make batches, otherwise takes them randomly')
		#
		parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
		parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
		parser.add_argument('--display_single_pane_ncols', type=int, default=0,
							help='if positive, display all images in a single visdom web panel with certain number '
								 'of images per row.')
		parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
							help='Maximum number of samples allowed per dataset. If the dataset directory contains '
								 'more than max_dataset_size, only a subset is loaded.')
		self.initialized = True
		return parser

	def gather_options(self):
		# initialize parser with basic options
		if not self.initialized:
			parser = argparse.ArgumentParser(
				formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		# get the basic options
		opt, _ = parser.parse_known_args()

		# modify model-related parser options
		model_name = opt.model  # cyclegan  后期的 self.model_names 是两回事
		model_option_setter = models.get_option_setter(model_name)
		parser = model_option_setter(parser, self.isTrain)
		opt, _ = parser.parse_known_args()  # parse again with the new defaults
		self.parser = parser
		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)
		# save to the disk
		if self.isTrain:
			expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
			util.mkdirs(expr_dir)
			file_name = os.path.join(expr_dir, 'opt.txt')
			with open(file_name, 'wt') as opt_file:
				opt_file.write(message)
				opt_file.write('\n')

	def parse(self):
		opt = self.gather_options()
		opt.isTrain = self.isTrain
		#
		self.print_options(opt)
		#
		gpu_num = torch.cuda.device_count()
		print('GPU NUM: {:2d}'.format(gpu_num))
		# set gpu ids
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		if len(opt.gpu_ids) > 0:
			torch.cuda.set_device(opt.gpu_ids[0])
		#
		self.opt = opt
		return self.opt


from misc import *
from util import html
from tqdm import tqdm
from models import create_model
from util.visualizer import Visualizer
from Test_global import test_num_id, num
from options.Base_Options_CycleGAN import BaseOptions_CycleGAN


def main(test_num, num):
    for test_epoch in num:
        class TestOptions(BaseOptions_CycleGAN):
            def initialize(self, parser):
                parser = BaseOptions_CycleGAN.initialize(self, parser)
                #
                parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
                #
                parser.add_argument('--model', type=str, default='testSingleGAN')
                #
                parser.add_argument('--valBatchSize', type=int, default=1, help='input batch size')
                ################################################
                # groups_supply_singles
                if test_num == 'groups_supply_singles':
                    parser.add_argument('--test_root', type=str,
                                        default=R'C:\Users\YuanBao\Desktop\1\groups_supply_singles')
                    parser.add_argument('--results_name', type=str, default='Images_groups_supply_singles')
                    parser.add_argument('--results_dir', type=str, default='./Images_groups_supply_singles/',
                    help='saves results here.')
                ################################################
                parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
                parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
                #
                parser.add_argument('--which_epoch', type=str, default=test_epoch,
                                    help='which epoch to load? set to latest to use latest cached model')
                #
                self.isTrain = False
                return parser
        #
        opt = TestOptions().parse()
        #
        # opt.manualSeed = random.randint(1, 10000)
        opt.manualSeed = 101
        #
        np.random.seed(opt.manualSeed)
        np.random.seed(opt.manualSeed)  # Numpy module.
        # torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
        # torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子；
        # 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
        torch.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed(opt.manualSeed)
        torch.cuda.manual_seed_all(opt.manualSeed)
        #
        torch.backends.cudnn.benchmark = False  # 寻找此结构下最优算法 默认False
        torch.backends.cudnn.deterministic = True  # 保证输入相同   默认False
        #
        print("Random Seed: ", opt.manualSeed)
        gpu_num = torch.cuda.device_count()
        print('GPU NUM: {:2d}'.format(gpu_num))
        #############################################
        opt.dataset_Name = 'CycleGAN_Dewater_test'
        #############################################
        valDataloader = getLoader(opt.dataset_Name,
                                  opt.test_root,
                                  opt.imageSize,
                                  opt.imageSize,
                                  opt.valBatchSize,
                                  opt.nThreads,
                                  mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                                  split='val',
                                  shuffle=False,
                                  seed=opt.manualSeed)

        model = create_model(opt)
        visualizer = Visualizer(opt)
        #
        dataset_size = len(valDataloader)  # 3141
        print('#training images = %d' % dataset_size)
        # create website
        web_dir = os.path.join(opt.results_dir, opt.name,
                               '%s_%s' % (opt.phase, opt.which_epoch))
        webpage = html.HTML(
            web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase,opt.which_epoch))
        #
        counter = 0
        #
        for i, data in enumerate(tqdm(valDataloader)):
            #
            model.set_input(data)
            model.test()
            #
            visuals = model.get_current_visuals()
            #
            img_path = model.get_image_paths()   # ['D:/Image_Real/Deal_With/Domain/De_water/test']
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, img_path, counter)
            counter = counter + 1
        ###########


if __name__ == '__main__':
    main(test_num_id, num)




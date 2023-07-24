import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='placeholder', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--easy_label', type=str, default='experiment_name', help='Interpretable name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='dcl', help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'], help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'stylegan2', 'smallstylegan2', 'resnet_cat'], help='specify generator architecture')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True,
                            help='no dropout for the generator')
        parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--random_scale_max', type=float, default=3.0,
                            help='(used for single image translation) Randomly scale the image by the specified factor as data augmentation.')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # parameters related to StyleGAN2-based network
        parser.add_argument('--stylegan2_G_num_downsampling',
                            default=1, type=int,
                            help='Number of downsampling layers used by StyleGAN2Generator')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    # def print_options(self, opt):
    #     """Print and save options

    #     It will print both current options and default values(if different).
    #     It will save options into a text file / [checkpoints_dir] / opt.txt
    #     """
    #     message = ''
    #     message += '----------------- Options ---------------\n'
    #     for k, v in sorted(vars(opt).items()):
    #         comment = ''
    #         default = self.parser.get_default(k)
    #         if v != default:
    #             comment = '\t[default: %s]' % str(default)
    #         message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    #     message += '----------------- End -------------------'
    #     print(message)

    #     # save to the disk
    #     expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    #     util.mkdirs(expr_dir)
    #     file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
    #     try:
    #         with open(file_name, 'wt') as opt_file:
    #             opt_file.write(message)
    #             opt_file.write('\n')
    #     except PermissionError as error:
    #         print("permission error {}".format(error))
    #         pass
    def print_options(self, opt):#hmg修改
        """Print and save options
        
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        import datetime
        notprint=['beta1','beta2','crop_size','display_env','display_freq',
                  'display_winsize','evaluation_freq','init_gain','init_type','input_nc',
                  'load_size','max_dataset_size','nce_T','ndf',
                #   'nce_includes_all_negatives_from_minibatch',
                  'no_antialias','no_antialias_up','no_dropout','no_flip',
                  'no_html','num_threads','output_nc','pool_size',
                  'print_freq','random_scale_max','sample_way',
                  'save_epoch_freq','save_latest_freq','update_html_freq',
                  'suffix','verbose',
                  '',''                  
                  'dataset_mode','display_id','display_ncols',
                  'display_port','display_server','easy_label','epoch_count',
                  'flip_equivariance','gan_mode','lr','lr_decay_iters',
                  'lr_policy','n_layers_D','nce_idt','netD','netF','netF_nc','netG','ngf','normD','normG',
                  'preprocess','save_by_iter','serial_batches',
                  'stylegan2_G_num_downsampling'
                  ]
        dt=datetime.datetime.now()
        message = '\n'
        message += '----------------- Options ---%d.%d %d:%d:%d------------\n'%(dt.month,dt.day,dt.hour,dt.minute,dt.second)
        messageprint='\n'
        messageprint += '----------------- Options ---%d.%d %d:%d:%d------------\n'%(dt.month,dt.day,dt.hour,dt.minute,dt.second)
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            if k not in notprint:messageprint+='{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End ------------------------------------------------------------------------'
        messageprint += '----------------- End -------------------'
        if opt.phase=='train':#只输出训练阶段的opt
            print(messageprint)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        if opt.phase=='train':#只保存训练阶段的opt
            file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
            try:
                with open(file_name, 'at') as opt_file:#hmg注意wt改成at
                    opt_file.write(message)
                    opt_file.write('\n')
            except PermissionError as error:
                print("permission error {}".format(error))
                pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        opt,printattr=self.hmggai(opt)#hmg添加

        self.print_options(opt)

        for p in printattr:
            print(p+'----'+str(getattr(opt, p)))


        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt

    def hmggai(self,opt):# hmg添加
        opt.display_id=-1
        if opt.isTrain:
            # opt.dataroot='datasets/horse2zebra'
            opt.dataroot='/media/cvlab/data/dataset/cityscapes'
            # opt.dataroot='/media/cvlab/data/dataset/afhq'
            opt.lr=0.0001 #论文里说DCL的lr是0.0001，SIMDCL是0.0002

            opt.name='city_Glo100_K4_dclw' #训练结果保存的路径
            # opt.name='test_forgaidaima'
            if opt.name.startswith('checkpoints/'):opt.name=opt.name[12:]
            opt.seed=1209

            opt.continue_train=True
            if opt.continue_train:
                opt.pretrained_name='city_Glo100_K4_dclw'
                # opt.pretrained_name='save_K_4_onepatch'
                # opt.pretrained_name='save_H2Z_DCL_1209'
                if opt.pretrained_name.startswith('checkpoints/'):opt.pretrained_name=opt.pretrained_name[12:]
                opt.epoch='250'

            opt.batch_size=2
            # opt.batch_size=1
            opt.no_html=True
            opt.save_epoch_freq=5
            opt.save_epoch_freq=1

            # opt.gpu_ids='-1'
            opt.global_num=20                   #在第几层GAP
            opt.usehmgmodification=True# use   hmg 
            if opt.usehmgmodification:              #井号 在这里是使用
                opt.global_similar_alph=100         #将特征图GAP后real和fake接近
                            # opt.Dreal_global_similar=1          #配合全局损失使用的，把Dreal里较大的像素置0后GAP
                # opt.ganloss_beta=0.2                #hmg注意，要使用pool，就不能使用ganloss_beta了..但是用pool的时候不是替换现有的图，而是拼在现有图的后面可以吗
                opt.K_num_patches=4                 #要不要加上负样本生成器，来生成更hard的样本？
                            # opt.nce_T_L=[0.19,0.15,0.11,0.07]   #配合K_num_patches来用的，网络越浅(感受野越小)像素间越相似的概率越大吧？应该,DCLW没用它
                # opt.perceptual_alph=0.1             #刚开始就加上还是先训练一定次数后再加上？arXiv:2111.14813v2
                opt.dclw_nce=True                   #是否使用解耦的对比损失
            opt.annotation='感觉gan02拖了大后腿'
            # opt.annotation='用官方的训练10轮后在16层全局(20层没弄到好的初始化F)，先继续训练25+25轮试试'
            
            opt.nce_includes_all_negatives_from_minibatch=False
            # opt.pool_size=50
            # opt.lr*=opt.batch_size
            
            # opt.n_epochs=5
            # opt.n_epochs_decay=201

            

            printattr=['name','seed','lr','n_epochs','n_epochs_decay'] #hmg添加
            if opt.continue_train:printattr+=['continue_train','pretrained_name','epoch']
            printattr+=['batch_size']
            if opt.usehmgmodification:
                printattr+=['usehmgmodification',
                            'global_similar_alph','Dreal_global_similar','global_num',
                            'ganloss_beta','K_num_patches',
                            'nce_T_L',
                            'perceptual_alph',
                            'dclw_nce']
            printattr+=['nce_includes_all_negatives_from_minibatch','pool_size']
        else:            
            # opt.dataroot='datasets/horse2zebra'
            opt.dataroot='/media/cvlab/data/dataset/afhq'
            # opt.dataroot='/media/cvlab/data/dataset/cityscapes'
            # opt.load_size=256
            
            # opt.name='H2Z_DCL'
            # opt.epoch=400
            # opt.model='dcl'
            # opt.gpu_ids=-1
            printattr=[]

        return opt,printattr
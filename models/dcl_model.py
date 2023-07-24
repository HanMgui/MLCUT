import itertools
import torch
from .base_model import BaseModel
from . import networks
# from .patchnce import PatchNCELoss2 as PatchNCELoss  #hmg注意，之前的实验都是PatchNCELoss，
#  但是PatchNCELoss只有使用外部负样本，论文里说不使用外部负样本，所以我改过来，在下面import了，所以这里去掉
import util.util as util
from util.image_pool import ImagePool

import os


class DCLModel(BaseModel):
    """ This class implements DCLGAN model.
    This code is inspired by CUT and CycleGAN.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for DCLGAN """
        parser.add_argument('--DCL_mode', type=str, default="DCL", choices='DCL')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=2.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--lambda_IDT', type=float, default=1.0, help='weight for l1 identical loss: (G(X),X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,  # hmg注意 论文里说的是False，代码里本来是True，我改成了False
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'],
                            help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization.")
        
        #hmg添加
        parser.add_argument('--usehmgmodification', type=bool, default=False,help='if use modification of hmg')
        parser.add_argument('--global_similar_alph', type=float, default=0,help='')
        parser.add_argument('--Dreal_global_similar', type=int, default=0,help='D(real) as mask to help global,higher is 0,lower is 1,means global of background.0means no use,>0 use the 1th,2th...method')
        parser.add_argument('--ganloss_beta', type=float, default=0,help='the param of ganloss2 need to in [0,1]')
        parser.add_argument('--K_num_patches', type=int, default=0,help='sample K*num_patches and sort,choose high num_patches for negative samples')
        parser.add_argument('--nce_T_L', type=list, default=[0.07,0.07,0.07,0.07],help='sample K*num_patches and sort,choose high num_patches for negative samples')
        parser.add_argument('--perceptual_alph', type=float, default=0,help='perceptual loss,0 means no use')
        parser.add_argument('--dclw_nce', type=bool, default=False,help='if use dclw loss(in 2110.06848)')

        parser.set_defaults(pool_size=0)  # no image pooling #hmg注意，要使用pool，就不能使用ganloss_beta了

        opt, _ = parser.parse_known_args()

        # Set default parameters for DCLGAN.
        if opt.DCL_mode.lower() == "dcl":
            parser.set_defaults(nce_idt=True, lambda_NCE=2.0)
        else:
            raise ValueError(opt.DCL_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'NCE1', 'D_B', 'G_B', 'NCE2', 'G']
        visual_names_A = ['real_A', 'fake_B']
        visual_names_B = ['real_B', 'fake_A']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['idt_B', 'idt_A']
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B

        if self.isTrain:
            self.model_names = ['G_A', 'F1', 'D_A', 'G_B', 'F2', 'D_B']
        else:  # during test time, only load G
            self.model_names = ['G_A', 'G_B']
        
        if self.opt.usehmgmodification:
            if self.opt.global_similar_alph>0: # hmg添加
                self.loss_names += ['glo_A', 'glo_B']
            if self.opt.perceptual_alph>0:
                self.loss_names += ['per_A', 'per_B']

        # define networks (both generator and discriminator)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netG_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                        opt.no_antialias_up, self.gpu_ids, opt)
        self.netF1 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)
        self.netF2 = networks.define_F(opt.input_nc, opt.netF, opt.normG,
                                       not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids,
                                       opt)

        if self.isTrain:
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            if opt.usehmgmodification and opt.ganloss_beta>0:#hmg修改
                self.criterionGAN = networks.GANLoss2(opt.gan_mode,ganloss_beta=opt.ganloss_beta).to(self.device)
            else:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            if self.opt.usehmgmodification and self.opt.dclw_nce:#hmg添加 K_n的判断在PatchNCELoss2_all里面，因此只在这里判断dclw
                from .patchnce_dclw import PatchNCELoss2_all
            else:
                from .patchnce import PatchNCELoss2_all
            for nce_layer in self.nce_layers:                
                self.criterionNCE.append(PatchNCELoss2_all(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionSim = torch.nn.L1Loss('sum').to(self.device) #DCL里好像没有用到，只是在SIMDCL里用了
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            if self.opt.usehmgmodification:
                if self.opt.global_similar_alph>0: # hmg添加
                    self.criterionGlo = torch.nn.L1Loss().to(self.device)
                    print('self.opt.global_num'+str(self.opt.global_num))
                if self.opt.K_num_patches>0: #hmg修改
                    nce_T_L=dict()
                    for i in range(len(self.opt.nce_T_L)):
                        nce_T_L[self.nce_layers[i]]=self.opt.nce_T_L[i]
                    self.nce_T_L=nce_T_L
                if self.opt.perceptual_alph>0:
                    from torchvision.models import vgg16
                    from .perceptual import LossNetwork
                    vgg_model = vgg16(pretrained=True).features[:16]
                    vgg_model = vgg_model.to(self.device)
                    # vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
                    for param in vgg_model.parameters():
                        param.requires_grad = False
                    self.perceptual_net=LossNetwork(vgg_model)
                    self.perceptual_net.eval()

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_G_loss().backward()  # calculate graidents for G
            if self.opt.usehmgmodification and self.opt.ganloss_beta>0:#hmg修改
                self.backward_D_AB_ganloss # calculate gradients for D_AB ganloss
            else:
                self.backward_D_A()  # calculate gradients for D_A
                self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()))
            # self.optimizer_F = torch.optim.Adam(itertools.chain(self.netF1.parameters(), self.netF2.parameters()),
            #                                     lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))  #hmg修改，根据论文来改的，注意注意注意！！！！！！！！！！！！！
            self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad() # 将梯度初始化为0
        if self.opt.usehmgmodification and self.opt.ganloss_beta>0:#hmg修改
            self.backward_D_AB_ganloss()
        else:
            self.backward_D_A()  # calculate gradients for D_A， 更新D
            self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step() # 更新参数

        # update G
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss() #更新G和F
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)

        if self.opt.nce_idt:
            self.idt_A = self.netG_A(self.real_B)
            self.idt_B = self.netG_B(self.real_A)

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B) #得到pool中所有的fake_B
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B) * self.opt.lambda_GAN

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A) * self.opt.lambda_GAN

    def backward_D_AB_ganloss(self):#hmg添加,注意，这里就没有使用pool了
        pred_realB = self.netD_A(self.real_B)
        pred_fakeB = self.netD_A(self.fake_B.detach())
        pred_realA = self.netD_B(self.real_A)
        pred_fakeA = self.netD_B(self.fake_A.detach())
        loss_D_realA = self.criterionGAN(pred_realB, True)
        loss_D_realB = self.criterionGAN(pred_realA, True)
        loss_D_fakeA = self.criterionGAN(pred_fakeB, False,pred_realA.detach())        
        loss_D_fakeB = self.criterionGAN(pred_fakeA, False,pred_realB.detach())
        # Combined loss and calculate gradients
        self.loss_D_A=(loss_D_realA + loss_D_fakeA)*0.5
        self.loss_D_B=(loss_D_realB + loss_D_fakeB) * 0.5
        loss_D = self.loss_D_A+self.loss_D_B       
        loss_D.backward()
        return loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fakeB = self.fake_B
        fakeA = self.fake_A

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:            
            if self.opt.usehmgmodification and self.opt.ganloss_beta>0:#hmg修改
                pred_fakeB = self.netD_A(fakeB)
                pred_fakeA = self.netD_B(fakeA)
                pred_realB = self.netD_A(self.real_B)
                pred_realA = self.netD_B(self.real_A)
                self.loss_G_A = self.criterionGAN(pred_fakeB, True,pred_realA.detach()).mean() * self.opt.lambda_GAN
                self.loss_G_B = self.criterionGAN(pred_fakeA, True,pred_realB.detach()).mean() * self.opt.lambda_GAN       

                # if not os.path.isdir('checkpoints/'+os.path.join(self.opt.pretrained_name,'D_out')):#hmg测试 保存判别器的结果
                #     os.mkdir('checkpoints/'+os.path.join(self.opt.pretrained_name,'D_out'))
                # save_path_real=os.path.join('checkpoints/'+os.path.join(self.opt.pretrained_name,'real'),self.image_paths[0].split('/')[-1])
                # util.save_image(util.tensor2im(self.real_B), save_path_real, aspect_ratio=1.0)#hmg测试 保存判别器的结果
                # save_path=os.path.join('checkpoints/'+os.path.join(self.opt.pretrained_name,'D_out'),self.image_paths[0].split('/')[-1]) #hmg测试 保存判别器的结果
                # a=pred_realB.detach()
                # a[a<0]/=(a.min()*-1)
                # a[a>0]/=a.max()
                # a=torch.nn.functional.interpolate(a,size=[256,256],mode='bilinear')
                # util.save_image(util.tensor2im(a), save_path, aspect_ratio=1.0)#hmg测试 保存判别器的结果
            else:
                pred_fakeB = self.netD_A(fakeB)
                pred_fakeA = self.netD_B(fakeA)
                self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN
                self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_A = 0.0
            self.loss_G_B = 0.0

        if self.opt.lambda_NCE > 0.0:
            if self.opt.usehmgmodification and self.opt.global_similar_alph>0: #hmg添加
                if self.opt.ganloss_beta==0:
                    pred_realB = self.netD_A(self.real_B)
                    pred_realA = self.netD_B(self.real_A)
                self.loss_NCE1,self.loss_glo_A = self.calculate_NCE_loss1(self.real_A, self.fake_B,pred_realA.detach())#.detach()好吗
                self.loss_NCE2,self.loss_glo_B = self.calculate_NCE_loss2(self.real_B, self.fake_A,pred_realB.detach())
                self.loss_NCE1*=self.opt.lambda_NCE
                self.loss_NCE2*=self.opt.lambda_NCE
                '''# a=torch.nn.functional.interpolate(self.netD_B(self.real_A),size=[256,256],mode='bilinear')#hmg添加，为了测试
                # b=torch.nn.functional.interpolate(self.netD_A(self.fake_B),size=[256,256],mode='bilinear')
                # c=abs(a+b)/2
                # self.savefeat(a.squeeze(dim=0).squeeze(dim=0),a,0)
                # self.savefeat(b.squeeze(dim=0).squeeze(dim=0),a,1)
                # self.savefeat(c.squeeze(dim=0).squeeze(dim=0),a,2)'''
            else:
                self.loss_NCE1 = self.calculate_NCE_loss1(self.real_A, self.fake_B) * self.opt.lambda_NCE
                self.loss_NCE2 = self.calculate_NCE_loss2(self.real_B, self.fake_A) * self.opt.lambda_NCE
        else:
            self.loss_NCE1, self.loss_NCE_bd, self.loss_NCE2 = 0.0, 0.0, 0.0
        if self.opt.lambda_NCE > 0.0:
            # self.loss_NCE1-=int(self.loss_NCE1) #因为加上dclw后损失变化的部分在千分位往后，只是乘上100的话又会产生崩溃解，所以把整数减掉试试
            # self.loss_NCE2-=int(self.loss_NCE2)
            # L1 IDENTICAL Loss
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * self.opt.lambda_IDT # self.idt_A=self.netG_A(self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * self.opt.lambda_IDT
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5 + (self.loss_idt_A + self.loss_idt_B) * 0.5
        else:
            loss_NCE_both = (self.loss_NCE1 + self.loss_NCE2) * 0.5

        self.loss_G = (self.loss_G_A + self.loss_G_B) * 0.5 + loss_NCE_both
        if self.opt.usehmgmodification:
            if self.opt.global_similar_alph>0: #hmg添加
                self.loss_glo_A*=self.opt.global_similar_alph
                self.loss_glo_B*=self.opt.global_similar_alph
                self.loss_G+=(self.loss_glo_A+self.loss_glo_B)*0.5
            if self.opt.perceptual_alph>0: #hmg添加
                self.loss_per_A= self.perceptual_net(self.fake_B, self.real_A)*self.opt.perceptual_alph
                self.loss_per_B= self.perceptual_net(self.fake_A, self.real_B)*self.opt.perceptual_alph
                self.loss_G+=(self.loss_per_A+self.loss_per_B)
        
        return self.loss_G

    def calculate_NCE_loss1(self, src, tgt,Dreal=None): #A->B 方向  #hmg修改
        n_layers = len(self.nce_layers)
        if self.opt.usehmgmodification and self.opt.global_similar_alph>0: #hmg添加
            if self.opt.global_num==20:
                feat_q = self.netG_B(tgt, self.nce_layers+[20], encode_only=True)#hmg注意 在第20层上
                feat_k = self.netG_A(src, self.nce_layers+[20], encode_only=True)#hmg注意 在第20层上
            elif self.opt.global_num==16:
                feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)#hmg注意 在第16层上
                feat_k = self.netG_A(src, self.nce_layers, encode_only=True)#hmg注意 在第16层上
            if self.opt.Dreal_global_similar>0 and Dreal is not None:#!!!别忘了改calculate_NCE_loss2
                if self.opt.Dreal_global_similar==1: #把较大的一半置0，这样的话global_similar_alph是不是应该大一点
                    Dreal=torch.nn.functional.interpolate(Dreal,size=(feat_q[-1].shape[2],feat_q[-1].shape[3]),mode='bilinear',align_corners=True)
                    Dreal2D=Dreal.view(Dreal.shape[0],-1)
                    maxnum,maxid=Dreal2D.topk(int(feat_q[-1].shape[2]*feat_q[-1].shape[3]/2),dim=1,sorted=False)
                    mask=torch.ones_like(Dreal)
                    mask2D=mask.view(Dreal.shape[0],-1)
                    for i in range(maxnum.shape[0]):
                        mask2D[i][maxid[i]]=0
                    feat_q[-1]*=mask
                    feat_k[-1]*=mask
            if self.opt.global_num==20:
                feat_q[-1]=feat_q[-1].sum(dim=[2,3],keepdim=True)#hmg注意 在第20层上
                feat_k[-1]=feat_k[-1].sum(dim=[2,3],keepdim=True)#hmg注意 在第20层上
            elif self.opt.global_num==16:
                feat_q.append(feat_q[-1].sum(dim=[2,3],keepdim=True))#hmg注意 在第16层上
                feat_k.append(feat_k[-1].sum(dim=[2,3],keepdim=True))#hmg注意 在第16层上
        else:
            feat_q = self.netG_B(tgt, self.nce_layers, encode_only=True)
            feat_k = self.netG_A(src, self.nce_layers, encode_only=True)
        if self.opt.usehmgmodification and self.opt.K_num_patches>0:  #hmg修改
            feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None,self.opt.K_num_patches)#torch.save(self.netF1(feat_k, 256*256, None),'visualization_for_hmg/'+self.image_paths[0].split('/')[-1].split('.')[0]+'.pth')
        else:
            feat_k_pool, sample_ids = self.netF1(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF2(feat_q, self.opt.num_patches, sample_ids)# self.savefeature(feat_q,feat_k)#hmg添加，用于测试
        if self.opt.usehmgmodification and self.opt.global_similar_alph>0: #hmg添加
            feat_q_global=feat_q_pool.pop()
            feat_k_global=feat_k_pool.pop()
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            if self.opt.usehmgmodification and self.opt.K_num_patches>0: #hmg修改
                loss = crit(f_q, f_k,self.nce_T_L[nce_layer])#,saveneg=True)#hmg测试，保存负样本的相似度
            else:
                loss = crit(f_q, f_k)#,saveneg=True)#hmg测试，保存负样本的相似度
            total_nce_loss += loss.mean()
        if self.opt.usehmgmodification and self.opt.global_similar_alph>0: #hmg添加
            return total_nce_loss / n_layers,self.criterionGlo(feat_q_global,feat_k_global)*self.opt.global_similar_alph
        else:
            return total_nce_loss / n_layers

    def calculate_NCE_loss2(self, src, tgt,Dreal=None): #B->A 方向  #hmg修改
        n_layers = len(self.nce_layers)
        if self.opt.usehmgmodification and self.opt.global_similar_alph>0: #hmg添加
            if self.opt.global_num==20:
                feat_q = self.netG_A(tgt, self.nce_layers+[20], encode_only=True)#hmg注意 在第20层上
                feat_k = self.netG_B(src, self.nce_layers+[20], encode_only=True)#hmg注意 在第20层上
            elif self.opt.global_num==16:
                feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)#hmg注意 在第16层上
                feat_k = self.netG_B(src, self.nce_layers, encode_only=True)#hmg注意 在第16层上
            if self.opt.Dreal_global_similar>0 and Dreal is not None:#!!!别忘了改calculate_NCE_loss2
                if self.opt.Dreal_global_similar==1:#把较大的一半置0
                    Dreal=torch.nn.functional.interpolate(Dreal,size=(feat_q[-1].shape[2],feat_q[-1].shape[3]),mode='bilinear',align_corners=True)
                    Dreal2D=Dreal.view(Dreal.shape[0],-1)
                    maxnum,maxid=Dreal2D.topk(int(feat_q[-1].shape[2]*feat_q[-1].shape[3]/2),dim=1,sorted=False)
                    mask=torch.ones_like(Dreal)
                    mask2D=mask.view(Dreal.shape[0],-1)
                    for i in range(maxnum.shape[0]):
                        mask2D[i][maxid[i]]=0
                    feat_q[-1]*=mask
                    feat_k[-1]*=mask
            if self.opt.global_num==20:
                feat_q[-1]=feat_q[-1].sum(dim=[2,3],keepdim=True)#hmg注意 在第20层上
                feat_k[-1]=feat_k[-1].sum(dim=[2,3],keepdim=True)#hmg注意 在第20层上
            elif self.opt.global_num==16:
                feat_q.append(feat_q[-1].sum(dim=[2,3],keepdim=True))#hmg注意 在第16层上
                feat_k.append(feat_k[-1].sum(dim=[2,3],keepdim=True))#hmg注意 在第16层上
        else:        
            feat_q = self.netG_A(tgt, self.nce_layers, encode_only=True)
            feat_k = self.netG_B(src, self.nce_layers, encode_only=True)
        if self.opt.usehmgmodification and self.opt.K_num_patches>0:  #hmg修改
            feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None,self.opt.K_num_patches)
        else:
            feat_k_pool, sample_ids = self.netF2(feat_k, self.opt.num_patches, None)

        feat_q_pool, _ = self.netF1(feat_q, self.opt.num_patches, sample_ids)
        if self.opt.usehmgmodification and self.opt.global_similar_alph>0: #hmg添加
            feat_q_global=feat_q_pool.pop()
            feat_k_global=feat_k_pool.pop()
        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            if self.opt.usehmgmodification and self.opt.K_num_patches>0: #hmg修改
                loss = crit(f_q, f_k,self.nce_T_L[nce_layer])
            else:
                loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        if self.opt.usehmgmodification and self.opt.global_similar_alph>0: #hmg添加
            return total_nce_loss / n_layers,self.criterionGlo(feat_q_global,feat_k_global)*self.opt.global_similar_alph
        else:
            return total_nce_loss / n_layers

    def generate_visuals_for_evaluation(self, data, mode): # 这个是不是也没用过
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG_A
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals

    def savefeat(self,featq,featk,num):#hmg添加，用于测试，下载图片
        import cv2
        import numpy
        # feat=abs(featk-featq)
        # featq=featq.sum(dim=1).squeeze()
        featq=(featq-featq.min())/(featq.max()-featq.min())
        # featk=featk.sum(dim=1).squeeze()
        # featk=(featk-featk.min())/(featk.max()-featk.min())
        # feat=feat.sum(dim=1).squeeze()
        # feat=(feat-feat.min())/(feat.max()-feat.min())
        # # feat=((feat.detach().to('cpu')+0.5)*255).numpy().astype('uint8')
        # f=torch.cat([featq,featk],dim=1)
        # f=torch.cat([f,feat],dim=1)
        f=featq.squeeze(dim=0).permute(1,2,0)
        f=numpy.array((f*255).detach().to('cpu'),dtype=numpy.uint8)
        name='checkpoints/test/'+self.image_paths[0].split('/')[-1].split('_')[-1][:-4]+'_%d.png'%num
        # name='checkpoints/test/'+'0_%d.png'%num
        cv2.imwrite(name,f)# H W C
        # print('save aa_%d.jpg'%num)
    
    def savefeature(self,feat_q,feat_k):
        feat_q_all_pool, _ = self.netF1(feat_q, 0, None)
        feat_k_all_pool, _ = self.netF2(feat_k, 0, None)
        qs=feat_q_all_pool
        ks=feat_k_all_pool
        cos = torch.nn.CosineSimilarity(dim=1)
        sim=[]
        for n in range(len(qs)):
            q=qs[n]
            k=ks[n]
            sim.append([])
            sim[-1]=cos(q,k).unsqueeze(dim=0)
            # sim[-1]=torch.nn.functional.interpolate(sim[-1],scale_factor=[256/q.shape[-1],256/q.shape[-1]],mode='bilinear')
            self.savefeat(sim[-1].squeeze(dim=0).squeeze(dim=0),sim[-1],n)
        a=0
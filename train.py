import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import os #hmg添加

def seed_torch(seed=1029):
    import random
    import os
    # import np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_gpu_info(use_index=(0,)):
    import pynvml
    # 初始化管理工具
    pynvml.nvmlInit()
    # device = torch.cuda.current_device()  # int
    gpu_count = pynvml.nvmlDeviceGetCount()  # int
    for index in range(gpu_count):
        # 不是使用的gpu，就剔除
        if index not in use_index:
            continue
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / 1024 ** 2  # 已用显存大小
    # 关闭管理工具
    pynvml.nvmlShutdown()
    return used  #MB

'''unaligned_dataset.py,base_options.py,train_options.py,fid_score.py'''
if __name__ == '__main__':
    # time.sleep(530*(50-38))
    # while get_gpu_info() > 400:# nohup python train.py --dataroot /media/cvlab/data/dataset/afhq &    datasets/horse2zebra &
    #     time.sleep(60*5)
    ifsuixunsuiting=True #随训随停，每个epoch都保存.pth，然后每%5的epoch都清理一下不是%5的.pth
    if ifsuixunsuiting:
        import datetime
        

    print()
    opt = TrainOptions().parse()   # get training options
    if opt.seed>0:seed_torch(opt.seed)

    # opt.phase='test'#hmg测试 保存判别器的结果
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    # opt.phase='train'#hmg测试 保存判别器的结果
    
    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    if opt.continue_train:#hmg添加
        if opt.epoch.isdigit() and opt.epoch_count==1:
            opt.epoch_count=int(opt.epoch)+1
        total_iters+=opt.epoch_count*(dataset_size-dataset_size%opt.batch_size)
        print('continue_train hmg and opt.epoch_count is !! %d !! ,total_iters is %d'%(opt.epoch_count,total_iters))
    t_data=0
    optimize_time = 0.1    
    times = []
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        #hmg添加,用于随机种子
        if epoch == opt.epoch_count and opt.epoch_count>1 and opt.seed>0:
            print('load hmg range(1,%d) for seed'%(opt.epoch_count))
            t_data=0
            for i in range(1,opt.epoch_count):
                for _, _ in enumerate(dataset):
                    break

        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                if len(opt.gpu_ids) > 0:model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            visuals = model.get_current_visuals()#hmg测试 保存判别器的结果

            # if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            #     save_result = total_iters % opt.update_html_freq == 0
            #     model.compute_visuals()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq < opt.batch_size:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data,ifprint=False)
                # if opt.display_id is None or opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
            #     print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
            #     print(opt.name)  # it's useful to occasionally show the experiment name on console
            #     save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
            #     model.save_networks(save_suffix)

            iter_data_time = time.time()
        
        if ifsuixunsuiting:#随训随停
            if epoch>280:ifsuixunsuiting=False
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                # model.save_networks('latest')
                model.save_networks(epoch)
                if epoch % 5 == 0:
                    fileroot=model.save_dir
                    filelist=os.listdir(fileroot)#hmg添加
                    for filename in filelist:
                        fileint=filename.split('_')[0]
                        if fileint.isdigit():
                            if int(fileint)%5 >0:
                                os.remove(os.path.join(fileroot,filename))
            dt=datetime.datetime.now()
            print('End of epoch %d / %d \t Time Taken: %ds|%.2fh,now %d.%d %d:%d:%d' % 
                (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time,
                (time.time() - epoch_start_time)/3600,dt.month,dt.day,dt.hour,dt.minute,dt.second))
        else:#正常
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                # model.save_networks('latest')
                model.save_networks(epoch)
            print('End of epoch %d / %d \t Time: %d s' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    # import subprocess
    # subprocess.Popen('nohup python train.py &',shell=True,stdout=None,stderr=None).wait()

# nohup python train.py &

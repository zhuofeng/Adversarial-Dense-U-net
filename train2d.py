# -*- coding: utf8 -*-
import os, time, pickle, random, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import csv
import tensorflow as tf
import tensorlayer as tl
from model import *
from utils import *
from config import config, log_config
import nibabel as nib
import datetime
from skimage.measure import block_reduce
###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  #now is 4
patch_size = config.PATCH_SIZE
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
logdir = config.VALID.logdir

ni = int(np.sqrt(batch_size))

def medicaltrain(): #需要改写成64*64的patch
    startTime = datetime.datetime.now()
    print(startTime)
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_INIT".format(tl.global_flag['mode']) #tl.global_flag['mode'] = args.mode
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    
    #define model
    t_image = tf.placeholder('float32', [batch_size, 64, 64, 1], name='t_image_input_to_UNET')
    t_target_image = tf.placeholder('float32', [batch_size, 64, 64, 1], name='t_target_image')
    
    
    dense_unet = DenseUNET(t_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d(dense_unet.outputs, is_train=True, reuse=True)
    
    dense_unet.print_params(False)
    net_d.print_params(False)
    
   
    '''
    #vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(dense_unet.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    #_, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)
    '''
    ## test inference
    dense_unet_test = DenseUNET(t_image, is_train=False, reuse=True)
    
    # ###========================== DEFINE TRAIN OPS ==========================###
    print(logits_real.get_shape().as_list())
    print(tf.ones_like(logits_real).get_shape().as_list())
    
    with tf.name_scope('d_losses'):
        with tf.name_scope('d_loss1'):
            d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')  #tf.ones_like: Creates a tensor with all elements set to 1. logits_real is output tensor
        with tf.name_scope('d_loss2'):
            d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
        with tf.name_scope('d_loss'):
            d_loss = d_loss1 + d_loss2
            d_loss = d_loss * 0.001
    tf.summary.scalar('d_loss', d_loss)
    print("defined d_loss")
    # Wasserstein GAN Loss
    #Discriminator's error
    with tf.name_scope('G_losses'):
        with tf.name_scope('g_gan_loss'):
            g_gan_loss = - 1e-3 * tf.reduce_mean(logits_fake) #Computes the mean of elements across dimensions of a tensor. (deprecated arguments)
        with tf.name_scope('mse_loss'):
            mse_loss = tl.cost.mean_squared_error(dense_unet.outputs, t_target_image, is_mean=True) 
            #mse_loss = tl.cost.mean_squared_error(dense_unet.outputs , t_target_image, is_mean=True) #(4,L,W,H,1)
        with tf.name_scope('g_loss'):
            g_loss = mse_loss + g_gan_loss
    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('g_loss', g_loss)
    print('defined g_loss')
    
    g_vars = tl.layers.get_variables_with_name('DenseUNET', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    #learning rate
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    #choose of learning method
    # g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    #print(type(dense_unet.outputs)) #<class 'tensorflow.python.framework.ops.Tensor'>
    #print(type(t_target_image)) #<class 'tensorflow.python.framework.ops.Tensor'>
    #print(type(mse_loss))  #<class 'tensorflow.python.framework.ops.Tensor'>
    g_optim_init = tf.train.RMSPropOptimizer(lr_v).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.RMSPropOptimizer(lr_v).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.RMSPropOptimizer(lr_v).minimize(d_loss, var_list=d_vars)
    
    # clip op
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars] #Clips tensor values to a specified min and max.

    ###========================== RESTORE MODEL =============================###
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    gpuconfig.allow_soft_placement=True
    gpuconfig.log_device_placement=False
    sess = tf.Session(config=gpuconfig)
    loss_writer = tf.summary.FileWriter(logdir, sess.graph)
    merged = tf.summary.merge_all()
    '''
    with sess.as_default():
        tf.global_variables_initializer().run()
    '''
    tl.layers.initialize_global_variables(sess)
    if os.path.exists(checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode'])):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=dense_unet)
        print("read the gan")
    elif os.path.exists(checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode'])):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=dense_unet)
        print("read the ganinit")
    else:
        print("g_init is new")
    if os.path.exists(checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode'])):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)
        print("read the D")
    else:
        print("d is new")    
    
    #save the sample data
    #读取图片并保存样本
    #读取图片.一次将所有训练集读进来
    ## If your machine have enough memory, please pre-load the whole train set.
    
    train_this_imgsarray, train_img_header = readnii(config.VALID.medical_path)
    print(train_this_imgsarray.shape)
    
    train_this_imgsarray = np.float32(train_this_imgsarray)
    train_hr_imgsarray_slices = train_this_imgsarray.shape[2]
    train_this_imgsarray = train_this_imgsarray[150:854, 150:854, int(train_hr_imgsarray_slices*0.1/8)*8:int(train_hr_imgsarray_slices*0.9/8)*8]
    train_this_imgsarray = normalizationmin0max1(train_this_imgsarray)
    
    print("this is the min and max:")
    print(np.min(train_this_imgsarray))
    print(np.max(train_this_imgsarray))
    #train_hr_imgs_array 训练的时候slice, height, width]
    sample_imgs = train_this_imgsarray[:,:,100:108]
    del train_this_imgsarray
    #print(np.min(sample_imgslist[0])) #-1.0
    #print(np.max(sample_imgslist[0])) #-0.073367156
    
    sample_imgs_SMALL = block_reduce(sample_imgs, block_size = (8,8,8), func=np.mean)
    
    print("the size of the sample images: ")
    print(sample_imgs.shape) 
    print(sample_imgs_SMALL.shape)
    print('the maximum and minimum ')
    print(np.min(sample_imgs))
    print(np.max(sample_imgs))
    print(np.min(sample_imgs_SMALL))
    print(np.max(sample_imgs_SMALL))
    
    #nib.save(sample_imgs)  save as nifti image
    imgbig = np.arange(sample_imgs.shape[0]*sample_imgs.shape[1]*sample_imgs.shape[2], dtype='float32').reshape(sample_imgs.shape[0], sample_imgs.shape[1], sample_imgs.shape[2])
    imgsmall = np.arange(sample_imgs_SMALL.shape[0]*sample_imgs_SMALL.shape[1]*sample_imgs_SMALL.shape[2], dtype='float32').reshape(sample_imgs_SMALL.shape[0], sample_imgs_SMALL.shape[1], sample_imgs_SMALL.shape[2]) 
    print("the shape")
    print(imgsmall.shape)
    print(imgbig.shape)
    imgbig = sample_imgs
    imgsmall = sample_imgs_SMALL
    print("the minmax")
    print(np.min(imgsmall))
    print(np.max(imgsmall))
    print(np.min(imgbig))
    print(np.max(imgbig))
    
    imgsmall = nib.Nifti1Image(imgsmall, np.eye(4))
    imgbig = nib.Nifti1Image(imgbig, np.eye(4))
    print(save_dir_ginit+'/_train_sample_small.nii.gz')
    print(save_dir_gan+'/_train_sample_big.nii.gz')
    nib.save(imgsmall, save_dir_ginit+'/_train_sample_small.nii.gz')
    nib.save(imgbig, save_dir_gan+'/_train_sample_big.nii.gz')
    
    del imgsmall
    del imgbig
    del sample_imgs
    
    #prepare training data  准备训练数据
    all_img_patchs = np.arange(config.TRAIN.batchnum*len(config.TRAIN.all_medical_path)*patch_size*patch_size*2, dtype='float32').reshape(config.TRAIN.batchnum*len(config.TRAIN.all_medical_path), patch_size, patch_size, 2)
    for j in range (0, len(config.TRAIN.all_medical_path)): 
        train_this_imgsarray, _ = readnii(config.TRAIN.all_medical_path[j])
        train_this_imgsarray = np.float32(train_this_imgsarray)
        train_hr_imgsarray_slices = train_this_imgsarray.shape[2]
        train_this_imgsarray = train_this_imgsarray[150:854, 150:854, int(train_hr_imgsarray_slices*0.1/8)*8:int(train_hr_imgsarray_slices*0.9/8)*8] #掐头去尾
        train_this_imgsarray = normalizationmin0max1(train_this_imgsarray)
        train_this_lr_imgsarray = block_reduce(train_this_imgsarray, block_size = (8,8,1), func=np.mean)
        train_this_lr_imgsarray = zoom(train_this_lr_imgsarray, (8.,8.,1.))
        print("the shape of both img arrays")
        print(train_this_imgsarray.shape)
        print(train_this_lr_imgsarray.shape)
        for i in range (0 ,config.TRAIN.batchnum):
            imgs_big_patch, imgs_small_patch = train_crop_both_imgs_fn_andsmall(train_this_imgsarray, train_this_lr_imgsarray,  patch_size, is_random=True) #patch size is 64*64
            all_img_patchs[j*config.TRAIN.batchnum+i, :, :, 0] = imgs_small_patch
            all_img_patchs[j*config.TRAIN.batchnum+i, :, :, 1] = imgs_big_patch
            print(i)
        del train_this_imgsarray
        
    print(all_img_patchs.shape)
    np.random.shuffle(all_img_patchs)
    print("the shape of all_img_patchs")
    print(all_img_patchs.shape)
    
    #准备测试数据
    test_hr_imgsarray, _ = readnii(config.VALID.medical_path)
    test_hr_imgsarray = test_hr_imgsarray[150:854, 150:854, int(train_hr_imgsarray_slices*0.1/8)*8:int(train_hr_imgsarray_slices*0.9/8)*8]
    test_hr_imgsarray = normalizationmin0max1(test_hr_imgsarray)
    print("this is the min and max:")
    print(np.min(test_hr_imgsarray))
    print(np.max(test_hr_imgsarray))
    test_lr_imgsarray = block_reduce(test_hr_imgsarray, block_size = (8,8,1), func=np.mean)
    test_lr_imgsarray = zoom(test_lr_imgsarray, (8.,8.,1.))
    
    test_img_patchs = np.arange(100*patch_size*patch_size*2, dtype='float32').reshape(100, patch_size, patch_size, 2)
   
    for i in range (0, 100):
            imgs_test_big_patch, imgs_test_small_patch = train_crop_both_imgs_fn_andsmall(test_hr_imgsarray, test_lr_imgsarray, patch_size, is_random=True)
            test_img_patchs[i, :, :, 0] = imgs_test_small_patch
            test_img_patchs[i, :, :, 1] = imgs_test_big_patch
    
    print("the shape of test_img_patchs")
    print(test_img_patchs.shape)
    print(np.min(all_img_patchs))
    print(np.max(all_img_patchs))
    print(np.min(test_img_patchs))
    print(np.max(test_img_patchs))
    
    test_hr_img_examples = test_img_patchs[0, :, :, 1]
    test_lr_img_examples = test_img_patchs[0, :, :, 0]
    outnii1 = nib.Nifti1Image(test_hr_img_examples, np.eye(4))
    nib.save(outnii1,'loss/testhr.nii.gz')
    outnii2 = nib.Nifti1Image(test_lr_img_examples, np.eye(4))
    nib.save(outnii2, 'loss/testlr.nii.gz')
    
    train_hr_img_examples = all_img_patchs[0, :, :, 1]
    train_lr_img_examples = all_img_patchs[0, :, :, 0]
    outnii3 = nib.Nifti1Image(train_hr_img_examples, np.eye(4))
    nib.save(outnii3,'loss/trainhr.nii.gz')
    outnii4 = nib.Nifti1Image(train_lr_img_examples, np.eye(4))
    nib.save(outnii4, 'loss/trainlr.nii.gz')
    ###========================= initialize G ====================###
    ## fixed learning rate
    #pretrain DENSENET
    
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0
        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, int(all_img_patchs.shape[0] / batch_size) * batch_size, batch_size):
            step_time = time.time()
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: all_img_patchs[idx:idx+batch_size, :, :, 0:1], t_target_image: all_img_patchs[idx:idx+batch_size, :, :, 1:2]}) 
            #errD, summary, _, _ = sess.run([d_loss, merged, d_optim, clip_D], {t_image: b_imgs_small, t_target_image: b_imgs_big})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss = total_mse_loss + errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)
        if epoch == 0:
            csvFileinit = open('loss/init3d.csv', 'a') 
            writer1 = csv.writer(csvFileinit)
            writer1.writerow(["epoch", "train_loss", "test_loss"])
            csvFileinit.close()
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 1 == 0):
            total_test_loss = 0
            for idx in range(0, int(test_img_patchs.shape[0] / batch_size) * batch_size, batch_size):
                out = sess.run(dense_unet.outputs, {t_image: test_img_patchs[idx:idx+batch_size,:,:,0:1]})
                mse = ((out - test_img_patchs[idx:idx+batch_size,:,:,1:2]) ** 2).mean(axis=None)
                total_test_loss = total_test_loss + mse
            
            total_test_loss = total_test_loss / (int(test_img_patchs.shape[0] / batch_size) * batch_size)
            out = sess.run(dense_unet.outputs, {t_image: test_img_patchs[0:batch_size, :, :, 0:1]})
            
            print("the test mse is:{}".format(total_test_loss))
            print("[*] save images")
            print("print some message of the out:{}")
            print("1. type of the out:{}".format(type(out)))
            print("2. shape of the out:{}".format(out.shape))
            print("3. max and min of the out:{}".format(np.max(out)))
            print(np.min(out))
            #tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)
            outnii = np.arange(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3], dtype='float32').reshape(out.shape[1], out.shape[2], out.shape[0]*out.shape[3])
            
            for i in range(0, out.shape[0]):
                outnii[:,:,i] = out[i,:,:,0]
            outnii = nib.Nifti1Image(outnii, np.eye(4))
            
            nib.save(outnii, save_dir_ginit+'/train_%d.nii.gz' % epoch)
            csvFileinit = open('loss/init2d.csv', 'a') 
            writer1 = csv.writer(csvFileinit)
            writer1.writerow([epoch, total_mse_loss/n_iter, total_test_loss])
            csvFileinit.close()
        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(dense_unet.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
            #tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_medicallungsrgan.npz', network=net_g)
    #正式开始训练
    ###========================= train GAN (SRGAN) =========================###
    
    # clipping method
    # clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, self.clip_values[0], self.clip_values[1])) for
    #                                      var in self.discriminator_variables]
    for epoch in range(0, n_epoch+1):
        ## update learning rate
        if epoch !=0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every) #config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_big = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_small = tl.prepro.threading_data(b_imgs_big, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, int(all_img_patchs.shape[0] / batch_size) * batch_size, batch_size):
            print()
            step_time = time.time()
            ## update D
            errD, _, _ = sess.run([d_loss, d_optim, clip_D], {t_image: all_img_patchs[idx:idx+batch_size, :, :, 0:1], t_target_image: all_img_patchs[idx:idx+batch_size, :, :, 1:2]})
            ## update G
            errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim],{t_image: all_img_patchs[idx:idx+batch_size, :, :, 0:1], t_target_image: all_img_patchs[idx:idx+batch_size, :, :, 1:2]})
            print("Epoch [%2d/%2d] time: %4.4fs, W_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)"
                  % (epoch, n_epoch, time.time() - step_time, errD, errG, errM, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)
        if epoch == 0:
            csvFilegan = open('loss/gan3d.csv', 'a') 
            writer2 = csv.writer(csvFilegan)
            writer2.writerow(["epoch", "test_loss", "errD", "errG", "errM", "errA"])
            csvFilegan.close()
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 1 == 0):
            total_test_loss = 0
            for idx in range(0, int(test_img_patchs.shape[0] / batch_size) * batch_size, batch_size):
                out = sess.run(dense_unet.outputs, {t_image: test_img_patchs[idx:idx+batch_size,:,:,0:1]})
                mse = ((out - test_img_patchs[idx:idx+batch_size,:,:,1:2]) ** 2).mean(axis=None)
                total_test_loss = total_test_loss + mse
            total_test_loss = total_test_loss / (int(test_img_patchs.shape[0] / batch_size) * batch_size)
            out = sess.run(dense_unet.outputs, {t_image: test_img_patchs[0:batch_size, :, :, 0:1]})
            print("the test mse is:{}".format(total_test_loss))
            print("[*] save images")
            print("print some message of the out:{}")
            print("1. type of the out:{}".format(type(out)))
            print("2. shape of the out:{}".format(out.shape))
            print("3. max and min of the out:{}".format(np.max(out)))
            print(np.min(out))
            outnii = np.arange(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3], dtype='float32').reshape(out.shape[1], out.shape[2], out.shape[0]*out.shape[3])
            for i in range(0, out.shape[0]):
                outnii[:,:,i] = out[i,:,:,0]
            outnii = nib.Nifti1Image(outnii, np.eye(4))
            nib.save(outnii, save_dir_gan+'/train_%d.nii.gz' % epoch)
            csvFileinit = open('loss/gan2d.csv', 'a') 
            writer2 = csv.writer(csvFileinit)
            writer2.writerow([epoch, total_test_loss, errD, errG, errM, errA])
            csvFileinit.close()
        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            #tl.files.save_npz(dense_unet.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(dense_unet.all_params, name=checkpoint_dir+'/g_{}.npz'.format(epoch), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
    overTime = datetime.datetime.now()
    print(overTime)
#对训练好的desenet进行测试
def medicalevaluateDenseUNET():
    checkpoint_dir = "checkpoint"
    valid_hr_imgsarray, valid_hr_img_header = readnii(config.VALID.medical_path) 
    valid_hr_slices = valid_hr_imgsarray.shape[2]
    print(valid_hr_imgsarray.shape)
    print("successfuly read the image!") 
    valid_hr_imgsarray = valid_hr_imgsarray[150:854, 150:854, int(valid_hr_slices*0.1/8)*8:int(valid_hr_slices*0.9/8)*8]
    valid_hr_imgsarray = normalizationmin0max1(valid_hr_imgsarray)
    #print(np.max(valid_hr_imgsarray))
    #print(np.min(valid_hr_imgsarray))
    
    valid_lr_imgsarray = block_reduce(valid_hr_imgsarray, block_size = (8,8,1), func=np.mean)
    valid_lr_imgsarray = zoom(valid_lr_imgsarray, (8.,8.,1))
    
    test_sample_small = nib.Nifti1Image(valid_lr_imgsarray, np.eye(4))
    test_sample_big = nib.Nifti1Image(valid_hr_imgsarray, np.eye(4))
    nib.save(test_sample_small, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/_test_sample_small.nii.gz')
    nib.save(test_sample_big, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/_test_sample_big.nii.gz')
    
    valid_hr_img = np.arange(valid_hr_imgsarray.shape[0]*valid_hr_imgsarray.shape[1]*valid_hr_imgsarray.shape[2], dtype = 'float32').reshape(valid_hr_imgsarray.shape[2], valid_hr_imgsarray.shape[0], valid_hr_imgsarray.shape[1], 1)
    valid_lr_img = np.arange(valid_hr_imgsarray.shape[0]*valid_hr_imgsarray.shape[1]*valid_hr_imgsarray.shape[2], dtype = 'float32').reshape(valid_hr_imgsarray.shape[2], valid_hr_imgsarray.shape[0], valid_hr_imgsarray.shape[1], 1)
    
    for i in range(0, valid_hr_imgsarray.shape[2]):
        valid_hr_img[i,:,:,0] = valid_hr_imgsarray[:,:,i]
        valid_lr_img[i,:,:,0] = valid_lr_imgsarray[:,:,i] 
    
    print('=============Some informaation of HR and LR img=================')
    print(np.min(valid_lr_img))
    print(np.max(valid_lr_img))
    print(np.min(valid_hr_img))
    print(np.max(valid_hr_img))
    
    t_image = tf.placeholder('float32', [valid_lr_img.shape[0], patch_size, patch_size, 1], name='input_image')
    
    print("Define the net work and read parameters")
    densenet = DenseUNET(t_image, is_train=False, reuse=False)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))
    tl.layers.initialize_global_variables(sess)
    load_params = tl.files.load_npz(name=checkpoint_dir+'/g_medicaltrain_init.npz')
    tl.files.assign_params(sess, load_params, network=densenet)
    
    
    valid_sr_img = np.zeros(valid_hr_img.shape[0]*valid_hr_img.shape[1]*valid_hr_img.shape[2], dtype = 'float32').reshape(valid_hr_img.shape[0], valid_hr_img.shape[1], valid_hr_img.shape[2], 1)
    
    '''
    for k in range(0, valid_lr_img.shape[1]/patch_size):
        for j in range(0, valid_lr_img.shape[1]/patch_size):
            valid_lr_patch = valid_lr_img[:, k*patch_size:(k+1)*patch_size, j*patch_size:(j+1)*patch_size, :]
            out = sess.run(densenet.outputs, {t_image: valid_lr_patch})
            valid_sr_img[:, k*patch_size:(k+1)*patch_size, j*patch_size:(j+1)*patch_size, 0] = out[:,:,:,0] #(50,64,64,1)
    '''
    print(valid_sr_img.shape)
    print(valid_hr_img.shape) #(440, 704, 704, 1)
    for k in range(0,2*valid_lr_img.shape[1]/patch_size-1):
        for j in range(0,2*valid_lr_img.shape[1]/patch_size-1):
            floatk = float(k)
            floatj = float(j)  
            valid_lr_patch = valid_lr_img[:, int(floatk *patch_size/2):int(floatk *patch_size/2 + patch_size), int(floatj*patch_size/2):int(floatj*patch_size/2 + patch_size), :]
            #print(valid_lr_patch.shape) (440, 64, 64, 1)
            out = sess.run(densenet.outputs, {t_image: valid_lr_patch})
            #print(out.shape) (440, 64, 64, 1)
            if floatk  == 0 and floatj == 0:
                valid_sr_img[:, int(floatk *patch_size) : int((floatk +1)*patch_size), int(floatj*patch_size):int((floatj+1)*patch_size), 0] = out[:,:,:,0] #(slices,64,64,1)
            elif floatk  == 0:
                valid_sr_img[:, int(floatk *patch_size/2):int((floatk +2)*patch_size/2), int(floatj*patch_size/2):int((floatj+1)*patch_size/2), 0] = (out[:,:,0:patch_size/2,0] + valid_sr_img[:, int(floatk *patch_size/2):int((floatk +2)*patch_size/2), int(floatj*patch_size/2):int((floatj+1)*patch_size/2), 0]) /2
                valid_sr_img[:, int(floatk *patch_size/2):int((floatk +2)*patch_size/2), int((floatj+1)*patch_size/2):int((floatj+2)*patch_size/2), 0] = (out[:,:,patch_size/2:patch_size,0] + valid_sr_img[:, int(floatk *patch_size/2):int((floatk +2)*patch_size/2), int((floatj+1)*patch_size/2):int((floatj+2)*patch_size/2), 0]) /2
            elif floatj == 0:
                valid_sr_img[:, int(floatk *patch_size/2):int((floatk +1)*patch_size/2), int(floatj*patch_size/2):int((floatj+2)*patch_size/2), 0] = (out[:,0:patch_size/2,:,0] + valid_sr_img[:, int(floatk *patch_size/2):int((floatk +1)*patch_size/2), int(floatj*patch_size/2):int((floatj+2)*patch_size/2), 0]) /2
                valid_sr_img[:, int((floatk +1)*patch_size/2):int((floatk +2)*patch_size/2), int(floatj*patch_size/2):int((floatj+2)*patch_size/2), 0] = (out[:,patch_size/2:patch_size,:,0] + valid_sr_img[:, int((floatk +1)*patch_size/2):int((floatk +2)*patch_size/2), int(floatj*patch_size/2):int((floatj+2)*patch_size/2), 0]) /2
            else:
                valid_sr_img[:, int(floatk *patch_size/2):int((floatk +2)*patch_size/2), int(floatj*patch_size/2):int((floatj+1)*patch_size/2), 0] = (out[:,:,0:patch_size/2,0] + valid_sr_img[:, int(floatk *patch_size/2):int((floatk +2)*patch_size/2), int(floatj*patch_size/2):int((floatj+1)*patch_size/2), 0]) /2
                valid_sr_img[:, int(floatk *patch_size/2):int((floatk +1)*patch_size/2), int(floatj*patch_size/2):int((floatj+2)*patch_size/2), 0] = (out[:,0:patch_size/2,:,0] + valid_sr_img[:, int(floatk *patch_size/2):int((floatk +1)*patch_size/2), int(floatj*patch_size/2):int((floatj+2)*patch_size/2), 0]) /2
                valid_sr_img[:, int((floatk +1)*patch_size/2):int((floatk +2)*patch_size/2), int((floatj+1)*patch_size/2):int((floatj+2)*patch_size/2), 0] = out[:,patch_size/2:patch_size,patch_size/2:patch_size,0]
                
            
    print("this is the output and the hr")
    print(valid_sr_img.shape)
    print(valid_hr_img.shape)
    print(np.min(valid_sr_img))
    print(np.max(valid_sr_img))
    print(np.min(valid_hr_img))  #1.0
    print(np.max(valid_hr_img))  #1.0
    
    mse1 = ((valid_hr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :] - valid_sr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :]) ** 2).mean(axis=None)
    mse2 = ((valid_hr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :] - valid_lr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :]) ** 2).mean(axis=None)
    print("this is the sr and bicubic mse")
    print(mse1)  #0.09449510108286935
    print(mse2)
    psnr1 = my_psnr(valid_hr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :], valid_sr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :])
    psnr2 = my_psnr(valid_hr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :], valid_lr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :])
    print("this is the sr and bicubic psnr")
    print(psnr1)
    print(psnr2)
    ssim1 = my_ssim(valid_hr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :], valid_sr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :])
    ssim2 = my_ssim(valid_hr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :], valid_lr_img[patch_size:valid_hr_img.shape[0]-patch_size, patch_size:valid_hr_img.shape[1]-patch_size, patch_size:valid_hr_img.shape[2] - patch_size, :])
    print("this is the sr and bicubic ssim")
    print(ssim1)
    print(ssim2)
    data_out = tf.convert_to_tensor(valid_sr_img, valid_sr_img.dtype)
    data_valid_hr_img = tf.convert_to_tensor(valid_hr_img, valid_hr_img.dtype)
    mse_loss = tl.cost.mean_squared_error(data_out , data_valid_hr_img, is_mean=True)
    print(mse_loss)
    sess1 = tf.Session()
    # Evaluate the tensor `c`.
    print(sess1.run(mse_loss))  #0.09449510108286927
    #print(out.shape) #4,big,big,3
    imgout = np.zeros(valid_sr_img.shape[0]*valid_sr_img.shape[1]*valid_sr_img.shape[2]*valid_sr_img.shape[3], dtype = 'float32').reshape(valid_sr_img.shape[1], valid_sr_img.shape[2], valid_sr_img.shape[0]*valid_sr_img.shape[3])
    for i in range(0, valid_sr_img.shape[0]):
        imgout[:,:,i] = valid_sr_img[i,:,:,0]
    print(imgout.shape)
    print(np.min(imgout))
    print(np.max(imgout))
    imgout = nib.Nifti1Image(imgout, np.eye(4))
    nib.save(imgout, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/SRimage.nii.gz')


if __name__ == '__main__':
    '''
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
    '''
    import argparse
    parser = argparse.ArgumentParser()

    #define the mode as srgan
    parser.add_argument('--mode', type=str, default='medicaltrain3D', help='evaluate')
    
    args = parser.parse_args()
    args.mode = 'medicalevaluateDenseUNET'
    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'medicaltrain':
        medicaltrain()
    elif tl.global_flag['mode'] == 'medicalevaluate':
        medicalevaluate()
    elif tl.global_flag['mode'] == 'medicalevaluateDenseUNET':
        medicalevaluateDenseUNET()
    elif tl.global_flag['mode'] == 'medicaltrain3D':
        medicaltrain3D()
    elif tl.global_flag['mode'] == 'medicalevaluateDenseUNET3D':
        medicalevaluateDenseUNET3D()
    else:
        raise Exception("Unknow --mode")
    
    '''
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_hr_imgs = read_all_imgs(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    print(len(train_hr_imgs))
    sample_imgs = train_hr_imgs[0:batch_size]
    print(type(sample_imgs))
    sample_imgs_big = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print(type(sample_imgs_big))
    print(sample_imgs_big.shape)
    '''
# -*- coding: utf8 -*-
import os, time, pickle, random, time
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

def medicaltrain():
    startTime = datetime.datetime.now()
    print(startTime)
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_INIT".format(tl.global_flag['mode']) #tl.global_flag['mode'] = args.mode
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    
    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgsarray, train_hr_img_header = readnii(config.VALID.medical_path) 
    train_hr_imgsarray = np.float32(train_hr_imgsarray)
    train_hr_imgsarray = normalizationminmax1(train_hr_imgsarray)
    
    print("this is the min and max:")
    print(np.min(train_hr_imgsarray))
    print(np.max(train_hr_imgsarray))
    train_hr_imgs = [] #a list
    i = 0
    
    
    while i < train_hr_imgsarray.shape[2]:
        train_hr_imgs.append(train_hr_imgsarray[:,:,i:i+1])
        i = i+1
    del train_hr_imgsarray
    ###========================== DEFINE MODEL ============================###
    ## train inference.
    t_image = tf.placeholder('float32', [batch_size, 96, 96, 1], name='t_image_input_to_UNET')
    t_target_image = tf.placeholder('float32', [batch_size, 384, 384, 1], name='t_target_image')

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
            d_loss = d_loss * 0.01
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, log_device_placement=True)
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
    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    print(type(sample_imgs)) #list
    for i in range(len(sample_imgs)):
        temparray = sample_imgs[i]
        sample_imgs[i] = temparray[:,:,0]
    
    # sample_imgs = read_all_imgs(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    #print(sample_imgs_384.shape)  #4,384,384
    
    print('sample HR sub-image:',sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    
    sample_imgs_96 = zoom(gaussian_filter(sample_imgs_384, 1, order=0, output=None, mode='reflect'), (1,0.25,0.25))
    
    print("the size of the sample images: ")
    print(sample_imgs_384.shape) #(4,384,384)
    print(sample_imgs_96.shape) 
    
    #nib.save(sample_imgs)  save as nifti image
    img384 = np.arange(sample_imgs_384.shape[0]*sample_imgs_384.shape[1]*sample_imgs_384.shape[2]).reshape(sample_imgs_384.shape[1], sample_imgs_384.shape[2], sample_imgs_384.shape[0])
    img96 = np.arange(sample_imgs_96.shape[0]*sample_imgs_96.shape[1]*sample_imgs_96.shape[2]).reshape(sample_imgs_96.shape[1], sample_imgs_96.shape[2], sample_imgs_96.shape[0]) #(96,96,4,1)
    img384.dtype = sample_imgs_384.dtype
    img96.dtype = sample_imgs_384.dtype
    for i in range(1, sample_imgs_384.shape[0]+1):
        img384[:,:,i-1] = sample_imgs_384[i-1,:,:]
        img96[:,:,i-1] = sample_imgs_96[i-1,:,:]
    print("the shape and minmax")
    print(img96.shape)
    print(img384.shape)
    print(np.min(img96))
    print(np.max(img96))
    print(np.min(img384))
    print(np.max(img384))
    
    img96 = nib.Nifti1Image(img96, np.eye(4))
    img384 = nib.Nifti1Image(img384, np.eye(4))
    nib.save(img96, save_dir_ginit+'/_train_sample_96.nii.gz')
    nib.save(img384, save_dir_gan+'/_train_sample_384.nii.gz')
    
    del img96
    del img384
    sample_imgs_96 = sample_imgs_96[:,:,:,np.newaxis]
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
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            if len(train_hr_imgs[idx : idx + batch_size])<batch_size:
                break
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            '''
            print('-------------------------------')
            print(np.max(b_imgs_96))
            print(np.min(b_imgs_96))
            print(np.max(b_imgs_384))
            print(np.min(b_imgs_384))
            print(b_imgs_96.shape)
            print(b_imgs_384.shape)
            print('-------------------------------')
            '''
            #print(b_imgs_96.shape)
            #print(np.max(b_imgs_384)) #147.59923
            #print(np.min(b_imgs_384)) #0.0
            #print(np.min(b_imgs_96)) #-1.0916667e-06
            #print(np.max(b_imgs_96)) #146.53592
            ## update G
            #print(b_imgs_384.shape) #(4, 384, 384, 1)
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})  #都是4维的，第一维是batch数，第二三维是长宽，第四维是channel，也就是1
            #errD, summary, _, _ = sess.run([d_loss, merged, d_optim, clip_D], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        print(log)
        errM, summary1, _ = sess.run([mse_loss, merged, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
        loss_writer.add_summary(summary1, epoch)
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(dense_unet_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
            #out = normalizationtoimg(out)
            print("[*] save images")
            print("print some message of the out:{}")
            print("1. type of the out:{}".format(type(out)))
            print("2. shape of the out:{}".format(out.shape))
            print("3. max and min of the out:{}".format(np.max(out)))
            print(np.min(out))
            #tl.vis.save_images(out, [ni, ni], save_dir_ginit+'/train_%d.png' % epoch)
            outnii = np.arange(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]).reshape(out.shape[1], out.shape[2], out.shape[0]*out.shape[3])
            outnii.dtype = out.dtype
            
            for i in range(1, out.shape[0]+1):
                for j in range(1, out.shape[3]+1):
                    outnii[:,:,(i-1)*out.shape[3]+j-1] = out[i-1,:,:,j-1]
            outnii = nib.Nifti1Image(outnii, np.eye(4))
            
            nib.save(outnii, save_dir_ginit+'/train_%d.nii.gz' % epoch)
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
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        for idx in range(0, len(train_hr_imgs), batch_size):
            if len(train_hr_imgs[idx : idx + batch_size])<batch_size:
                break
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(
                    train_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
            ## update D
            errD, _, _ = sess.run([d_loss, d_optim, clip_D], {t_image: b_imgs_96, t_target_image: b_imgs_384})
            
            # d_vars = sess.run(clip_discriminator_var_op)
            ## update G
            errG, errM, errA, _ = sess.run([g_loss, mse_loss, g_gan_loss, g_optim],{t_image: b_imgs_96, t_target_image: b_imgs_384})

            print("Epoch [%2d/%2d] %4d time: %4.4fs, W_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)"
                  % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        print(log)
        errD, summary2, _, _ = sess.run([d_loss, merged, d_optim, clip_D], {t_image: b_imgs_96, t_target_image: b_imgs_384})
        errG, errM, errA, summary3, _ = sess.run([g_loss, mse_loss, g_gan_loss, merged, g_optim],{t_image: b_imgs_96, t_target_image: b_imgs_384})
        loss_writer.add_summary(summary2, epoch)
        loss_writer.add_summary(summary3, epoch)
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(dense_unet_test.outputs, {t_image: sample_imgs_96})#; print('gen sub-image:', out.shape, out.min(), out.max())
            #out = normalizationtoimg(out)
            print("[*] save images")
            #tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch))
            print("print some message of the out:")
            print("1. type of the out:{}".format(type(out)))
            print("2. shape of the out:{}".format(out.shape))
            print("3. max of the out:{}".format(np.max(out)))
            print("4. min of the out{}".format(np.min(out)))
            outnii = np.arange(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]).reshape(out.shape[1], out.shape[2], out.shape[0]*out.shape[3])
            outnii.dtype = out.dtype
            for i in range(1, out.shape[0]+1):
                for j in range(1, out.shape[3]+1):
                    outnii[:,:,(i-1)*out.shape[3]+j-1] = out[i-1,:,:,j-1]
            outnii = nib.Nifti1Image(outnii, np.eye(4))
            nib.save(outnii, save_dir_gan+'/train_%d.nii.gz' % epoch)
        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(dense_unet.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
    overTime = datetime.datetime.now()
    print(overTime)

def medicaltrain3D():
    startTime = datetime.datetime.now()
    print(startTime)
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_INIT".format(tl.global_flag['mode']) #tl.global_flag['mode'] = args.mode  
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint3D"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    
    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgsarray, train_hr_img_header = readnii(config.TRAIN.medical_path) 
    train_hr_imgsarray = np.float32(train_hr_imgsarray)
    train_hr_imgsarray = train_hr_imgsarray[150:850, 150:850, :]
    train_hr_imgsarray = normalizationmin0max1(train_hr_imgsarray)
    
    print("this is the min and max:")
    print(np.min(train_hr_imgsarray))
    print(np.max(train_hr_imgsarray))
    train_hr_imgs = [] #a list
    i = 0
    
    
    while i < train_hr_imgsarray.shape[2]:
        train_hr_imgs.append(train_hr_imgsarray[:,:,i:i+1])
        i = i+1
    del train_hr_imgsarray
    
    t_image = tf.placeholder('float32', [batch_size, int(patch_size), int(patch_size), int(patch_size), 1], name='t_image_input_to_UNET')
    t_target_image = tf.placeholder('float32', [batch_size, patch_size, patch_size, patch_size, 1], name='t_target_image')

    dense_unet3D = DenseUNET3D(t_image, is_train=True, reuse=False)
    
    # Generate high res. version from low res.
    
    net_d, logits_real = SRGAN_d3D(t_target_image, is_train=True, reuse=False)
    _,     logits_fake = SRGAN_d3D(dense_unet3D.outputs, is_train=True, reuse=True)

    dense_unet3D.print_params(False)
    net_d.print_params(False)
    
    '''
    #vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False) # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(dense_unet.outputs, size=[224, 224], method=0, align_corners=False) # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224+1)/2, reuse=False)
    #_, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224+1)/2, reuse=True)
    '''
    ## test inference
    dense_unet3D_test = DenseUNET3D(t_image, is_train=False, reuse=True)
    
    # ###========================== DEFINE TRAIN OPS ==========================###
    print(logits_real.get_shape().as_list())
    print(tf.ones_like(logits_real).get_shape().as_list())
    
    with tf.name_scope('d_losses'):
        with tf.name_scope('d_loss1'):
            d_loss1 = tf.losses.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))  #tf.ones_like: Creates a tensor with all elements set to 1. logits_real is output tensor
        with tf.name_scope('d_loss2'):
            d_loss2 = tf.losses.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
        with tf.name_scope('d_loss'):
            d_loss = d_loss1 + d_loss2
            d_loss = d_loss * 0.001
    tf.summary.scalar('d_loss', d_loss)
    print("defined d_loss")
    # Wasserstein GAN Loss
    #Discriminator's error
    print(dense_unet3D.outputs.shape)  #(4, 128, 128, 128, 1)
    
    with tf.name_scope('G_losses'):
        with tf.name_scope('g_gan_loss'):
            g_gan_loss = - 1e-3 * tf.reduce_mean(logits_fake) #Computes the mean of elements across dimensions of a tensor. (deprecated arguments)
            print('gan_loss')
        with tf.name_scope('mse_loss'):
            mse_loss = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(dense_unet3D.outputs, t_target_image), [1, 2, 3, 4]))
            #mse_loss = tl.cost.mean_squared_error(dense_unet3D.outputs, t_target_image, is_mean=True)
            #mse_loss = tl.cost.sigmoid_cross_entropy(dense_unet3D.outputs, t_target_image)
        with tf.name_scope('g_loss'):
            g_loss = mse_loss + g_gan_loss
    tf.summary.scalar('mse_loss', mse_loss)
    tf.summary.scalar('g_loss', g_loss)
    print('defined g_loss')
    
    #print(type(mse_loss)) #<class 'tensorflow.python.framework.ops.Tensor'>
    #print(type(dense_unet3D.outputs)) #<class 'tensorflow.python.framework.ops.Tensor'>
    #print(type(t_target_image)) #<class 'tensorflow.python.framework.ops.Tensor'>
    
    g_vars = tl.layers.get_variables_with_name('DenseUNET3D', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d3D', True, True)

    #learning rate
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    #choose of learning method
    #g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    g_optim_init = tf.train.RMSPropOptimizer(lr_v).minimize(mse_loss, var_list=g_vars)

    ## SRGAN
    g_optim = tf.train.RMSPropOptimizer(lr_v).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.RMSPropOptimizer(lr_v).minimize(d_loss, var_list=d_vars)
    
    # clip op
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars] #Clips tensor values to a specified min and max.

    ###========================== RESTORE MODEL =============================###
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, log_device_placement=True)
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
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), network=dense_unet3D)
        print("read the gan")
    elif os.path.exists(checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode'])):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), network=dense_unet3D)
        print("read the ganinit")
    else:
        print("g_init is new")
    if os.path.exists(checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode'])):
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)
        print("read the D")
    else:
        print("d is new")  
    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgslist = train_hr_imgs[0:8]
    #print(np.min(sample_imgslist[0])) #-1.0
    #print(np.max(sample_imgslist[0])) #-0.073367156
    
    print(type(sample_imgslist)) #list
    for i in range(len(sample_imgslist)):
        temparray = sample_imgslist[i]
        sample_imgslist[i] = temparray[:,:,0]
    
    print(len(sample_imgslist))  #8
    print(type(sample_imgslist[0]))  #array
    print(sample_imgslist[0].shape[0])
    sample_imgs = np.arange(len(sample_imgslist) * sample_imgslist[0].shape[0] * sample_imgslist[0].shape[1]).reshape(len(sample_imgslist), sample_imgslist[0].shape[0], sample_imgslist[0].shape[1])
    sample_imgs = np.float32(sample_imgs)
    print(sample_imgs.shape)
    
    for i in range(0, len(sample_imgslist)):
        sample_imgs[i,:,:] = sample_imgslist[i]
    
    sample_imgs_SMALL = block_reduce(sample_imgs, block_size = (8,8,8), func=np.mean)
    
    print("the size of the sample images: ")
    print(sample_imgs.shape) #(8, 1024, 1024)
    print(sample_imgs_SMALL.shape) #(1, 128, 128)
    print('the maximum and minimum ')
    print(np.min(sample_imgs))
    print(np.max(sample_imgs))
    print(np.min(sample_imgs_SMALL))
    print(np.max(sample_imgs_SMALL))
    
    #nib.save(sample_imgs)  save as nifti image
    imgbig = np.arange(sample_imgs.shape[0]*sample_imgs.shape[1]*sample_imgs.shape[2]).reshape(sample_imgs.shape[1], sample_imgs.shape[2], sample_imgs.shape[0])
    imgsmall = np.arange(sample_imgs_SMALL.shape[0]*sample_imgs_SMALL.shape[1]*sample_imgs_SMALL.shape[2]).reshape(sample_imgs_SMALL.shape[1], sample_imgs_SMALL.shape[2], sample_imgs_SMALL.shape[0]) 
    imgbig.dtype = sample_imgs.dtype
    imgsmall.dtype = sample_imgs.dtype
    print("the shape")
    print(imgsmall.shape)
    print(imgbig.shape)
    for i in range(0, sample_imgs.shape[0]):
        imgbig[:,:,i] = sample_imgs[i,:,:]
    for i in range(0, sample_imgs_SMALL.shape[0]):
        imgsmall[:,:,0] = sample_imgs_SMALL[i,:,:]
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
    
    sample_imgs_SMALL = sample_imgs_SMALL[:,:,:,np.newaxis]
    
    train_hr_imgs_array = np.arange(len(train_hr_imgs)*train_hr_imgs[0].shape[0]*train_hr_imgs[0].shape[1], dtype = 'float32').reshape(len(train_hr_imgs), train_hr_imgs[0].shape[0], train_hr_imgs[0].shape[1])
    #print(train_hr_imgs_array.shape) #(545, 1024, 1024)
    for i in range(0, len(train_hr_imgs)):
        train_hr_imgs_array[i, :, :] = train_hr_imgs[i][:,:,0]
    del train_hr_imgs
    small_patch_size = int(patch_size / 8)
    test_img_big, test_img_small = train_crop_sub_imgs_fn_andsmall3D(train_hr_imgs_array, batch_size, patch_size, small_patch_size, is_random=False)
    #testimgbig = nib.Nifti1Image(test_img_big, np.eye(4))
    #testimgsmall = nib.Nifti1Image(test_img_small, np.eye(4))
    test_sample_img_big = test_img_big[0,:,:,:,0]
    test_sample_img_small = test_img_small[0,:,:,:,0]
    print(test_sample_img_big.shape)
    print(test_sample_img_small.shape)
    test_sample_img_big = nib.Nifti1Image(test_sample_img_big, np.eye(4))
    test_sample_img_small = nib.Nifti1Image(test_sample_img_small, np.eye(4))
    nib.save(test_sample_img_big, save_dir_ginit+'/_test_img_big.nii.gz')
    nib.save(test_sample_img_small, save_dir_gan+'/_test_img_small.nii.gz')
    
    '''
        将学习次数和损失保存成为csv
    '''
    ###========================= initialize G ====================### 
    #对generator进行预训练
    ## fixed learning rate
    #pretrain DENSENET
    
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init+1):
        epoch_time = time.time()

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        step_time = time.time()
        '''
        b_imgs_big = tl.prepro.threading_data(
                train_hr_imgs[idx : idx + batch_size],
                fn=crop_sub_imgs_fn3D, is_random=True)
        b_imgs_small = tl.prepro.threading_data(b_imgs_big, fn=downsample_fn)
        '''
        
        b_imgs_big, b_imgs_small = train_crop_sub_imgs_fn_andsmall3D(train_hr_imgs_array, batch_size, patch_size, small_patch_size, is_random=True)
        if np.max(b_imgs_big)==-1. or np.max(b_imgs_small)==-1.:
            continue
        
        #事实证明没有问题,但是CT图象没有用的空气部分太多了，导致训练不好
        '''
        test_save_small = b_imgs_small[0,:,:,:,0]
        test_save_big = b_imgs_big[0,:,:,:,0]
        test_save_small = nib.Nifti1Image(test_save_small, np.eye(4))
        test_save_big = nib.Nifti1Image(test_save_big, np.eye(4))
        nib.save(test_save_small, save_dir_gan+'/_test_save_small.nii.gz')
        nib.save(test_save_big, save_dir_gan+'/_test_save_big.nii.gz')
        os._exit(0)
        '''
        '''
        print('-------------------------------')
        print(np.max(b_imgs_96))
        print(np.min(b_imgs_96))
        print(np.max(b_imgs_384))
        print(np.min(b_imgs_384))
        print(b_imgs_96.shape)
        print(b_imgs_384.shape)
        print('-------------------------------')
        '''
        #print(b_imgs_96.shape)
        #print(np.max(b_imgs_384)) #147.59923
        #print(np.min(b_imgs_384)) #0.0
        #print(np.min(b_imgs_96)) #-1.0916667e-06
        #print(np.max(b_imgs_96)) #146.53592
        ## update G
        #print(b_imgs_384.shape) #(4, 384, 384, 1)
        errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_small, t_target_image: b_imgs_big})  #都是4维的，第一维是batch数，第二三维是长宽，第四维是channel，也就是1
        #errD, summary, _, _ = sess.run([d_loss, merged, d_optim, clip_D], {t_image: b_imgs_96, t_target_image: b_imgs_384})
        #print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
        #total_mse_loss += errM
        #n_iter += 1
        #log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss/n_iter)
        #print(log)
        errM, summary1, _ = sess.run([mse_loss, merged, g_optim_init], {t_image: b_imgs_small, t_target_image: b_imgs_big})
        loss_writer.add_summary(summary1, epoch)
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, errM)
        print(log)
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            print(np.max(test_img_small))
            print(np.min(test_img_small))
            out = sess.run(dense_unet3D.outputs, {t_image: test_img_small})#; print('gen sub-image:', out.shape, out.min(), out.max())
            #out = normalizationtoimg(out)
            print("[*] save images")
            print("print some message of the out:{}")
            print("1. type of the out:{}".format(type(out)))
            print("2. shape of the out:{}".format(out.shape))
            print("3. max and min of the out:{}".format(np.max(out))) #0.0
            print(np.min(out)) #0.0
            outnii = np.arange(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]*out.shape[4], dtype = 'float32').reshape(out.shape[1], out.shape[2], out.shape[3])
            outnii = out[0,:,:,:,0]
            print(outnii.shape)
            outnii = nib.Nifti1Image(outnii, np.eye(4))
            nib.save(outnii, save_dir_ginit+'/train_%d.nii.gz' % epoch)
            #save loss
            csvFileinit = open('loss/init.csv', 'a') 
            writer1 = csv.writer(csvFileinit)
            writer1.writerow([epoch, errM])
            csvFileinit.close()
        ## save model
        if (epoch != 0) and (epoch % 100 == 0): #每迭代100次保存1次
            loss_writer.add_summary(summary1, epoch)
            tl.files.save_npz(dense_unet3D.all_params, name=checkpoint_dir+'/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
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
        #total_d_loss, total_g_loss = 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.
        step_time = time.time()
        '''
        b_imgs_384 = tl.prepro.threading_data(
                train_hr_imgs[idx : idx + batch_size],
                fn=crop_sub_imgs_fn3D, is_random=True)
        b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
        '''
        b_imgs_big, b_imgs_small = train_crop_sub_imgs_fn_andsmall3D(train_hr_imgs_array, batch_size, patch_size, small_patch_size, is_random=True)
        ## update D
        '''
        print("this is some messages of the gan")
        print(b_imgs_big.shape)
        print(b_imgs_small.shape)
        print(np.max(b_imgs_big))
        print(np.min(b_imgs_big))
        print(np.max(b_imgs_small))
        print(np.min(b_imgs_small))
        '''
        errD, summary2, _, _ = sess.run([d_loss, merged, d_optim, clip_D], {t_image: b_imgs_small, t_target_image: b_imgs_big})
        
        # d_vars = sess.run(clip_discriminator_var_op)
        ## update G
        errG, errM, errA, summary3, _ = sess.run([g_loss, mse_loss, g_gan_loss, merged, g_optim],{t_image: b_imgs_small, t_target_image: b_imgs_big})
        
        print("Epoch [%2d/%2d] time: %4.4fs, W_loss: %.8f g_loss: %.8f (mse: %.6f adv: %.6f)"
              % (epoch, n_epoch, time.time() - step_time, errD, errG, errM, errA))
        #total_d_loss += errD
        #total_g_loss += errG
        loss_writer.add_summary(summary2, epoch)
        loss_writer.add_summary(summary3, epoch)
        #log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss/n_iter, total_g_loss/n_iter)
        #print(log)
        '''
        errD, summary2, _, _ = sess.run([d_loss, merged, d_optim, clip_D], {t_image: b_imgs_small, t_target_image: b_imgs_big})
        errG, errM, errA, summary3, _ = sess.run([g_loss, mse_loss, g_gan_loss, merged, g_optim],{t_image: b_imgs_small, t_target_image: b_imgs_big})
        loss_writer.add_summary(summary2, epoch)
        loss_writer.add_summary(summary3, epoch)
        '''
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 100 == 0):
            out = sess.run(dense_unet3D.outputs, {t_image: test_img_small})#; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            #tl.vis.save_images(out, [ni, ni], save_dir_gan+'/train_%d.png' % epoch))
            print("print some message of the out:")
            print("1. type of the out:{}".format(type(out)))
            print("2. shape of the out:{}".format(out.shape))
            print("3. max of the out:{}".format(np.max(out)))
            print("4. min of the out{}".format(np.min(out)))
            
            outnii = np.arange(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]*out.shape[4], dtype = 'float32').reshape(out.shape[1], out.shape[2], out.shape[3], out.shape[0]*out.shape[4])
            outnii = out[0,:,:,:,0]
            print(outnii.shape)
            outnii = nib.Nifti1Image(outnii, np.eye(4))
            nib.save(outnii, save_dir_gan+'/train_%d.nii.gz' % epoch)
            csvFilegan = open('loss/gan.csv', 'a') 
            writer2 = csv.writer(csvFilegan)
            writer2.writerow([epoch, errD, errG, errM, errA])
            csvFilegan.close()
        ## save model
        if (epoch != 0) and (epoch % 100 == 0):
            tl.files.save_npz(dense_unet3D.all_params, name=checkpoint_dir+'/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
    overTime = datetime.datetime.now()
    print(overTime)
    
def medicalevaluateDenseUNET3D():
    
    checkpoint_dir = "checkpoint3D"
    valid_hr_imgsarray, valid_hr_img_header = readnii(config.VALID.medical_path)
    print(valid_hr_imgsarray.shape)
    print("successfuly read the image!")
    valid_hr_imgsarray = np.float32(valid_hr_imgsarray)
    valid_hr_imgsarray = normalizationmin0max1(valid_hr_imgsarray)
    valid_hr_imgsarray = valid_hr_imgsarray[192:832,192:832,256:320]
    
    t_image =  tf.placeholder('float32', [1, 64, 64, 64, 1], name='input_image')
    
    densenet3D = DenseUNET3D(t_image, is_train=False, reuse=False)
    
    gpuconfig = tf.ConfigProto()
    gpuconfig.gpu_options.allow_growth = True
    gpuconfig.allow_soft_placement=True
    sess = tf.Session(config=gpuconfig)
    tl.layers.initialize_global_variables(sess)
    
    load_params = tl.files.load_npz(name=checkpoint_dir+'/iteration47610/g_medicaltrain3D_init.npz')
    tl.files.assign_params(sess, load_params, network=densenet3D)
    imgshape = valid_hr_imgsarray.shape
    #print(imgshape) #(1024, 1024, 64)
    SR_img = np.zeros((imgshape[0],imgshape[1],imgshape[2]),dtype='float32')
    input_img_array = np.arange(patch_size * patch_size * patch_size, dtype = 'float32').reshape(1, patch_size, patch_size, patch_size, 1)
    valid_lr_imgsarray = block_reduce(valid_hr_imgsarray, block_size = (8,8,8), func=np.mean)
    valid_lr_imgsarray = zoom(valid_lr_imgsarray, (8.,8.,8.))
    print(valid_lr_imgsarray.shape) #(1024, 1024, 64)
    k = 0
    x = 10
    stepsize = 50 #设置步长
    
    #2621440
    while x <= (valid_lr_imgsarray.shape[0]-patch_size):
        y = 10
        while y <= valid_lr_imgsarray.shape[1]-patch_size:
            input_img_array[0,:,:,:,0] = valid_lr_imgsarray[x-7:x+patch_size-7, y-7:y+patch_size-7,:]
            valid_sr_imgsarray = sess.run(densenet3D.outputs, {t_image: input_img_array})
            print(np.min(valid_sr_imgsarray))
            print(np.max(valid_sr_imgsarray))
            SR_img[x:x+50, y:y+50,:] = valid_sr_imgsarray[0,7:57,7:57,:,0]
            print(np.min(SR_img))
            print(np.max(SR_img))
            print(4) #这个最多
            k = k + 1
            print("{} iteration success".format(k))
            print("this is x and y")
            print(x)
            print(y)
            y = y + 50
        x = x + 50
    
    
    for x in range(0, 10):
        for y in range(0, 10):
            input_img_array[0,:,:,:,0] = valid_lr_imgsarray[x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size,:]
            print(np.min(input_img_array))
            print(np.max(input_img_array))
            valid_sr_imgsarray = sess.run(densenet3D.outputs, {t_image: input_img_array})
            #print(valid_sr_imgsarray.shape) (1, 64, 64, 64, 1)
            SR_img[x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size,:] = valid_sr_imgsarray[0,:,:,:,0]
            k = k + 1
            print("{} iteration success".format(k))
            y = y + 1
        x = x + 1
    
        
    SR_img = nib.Nifti1Image(SR_img, np.eye(4))
    valid_lr_imgsarray = nib.Nifti1Image(valid_lr_imgsarray, np.eye(4))
    nib.save(SR_img, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest3D/_out.nii.gz')
    nib.save(valid_lr_imgsarray, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest3D/_lr.nii.gz')

#对训练好的desenet进行测试
def medicalevaluateDenseUNET():
    checkpoint_dir = "checkpoint"
    valid_hr_imgsarray, valid_hr_img_header = readnii(config.TRAIN.medical_path) 
    print(valid_hr_imgsarray.shape)
    print("successfuly read the image!") 
    valid_hr_imgsarray = normalizationminmax1(valid_hr_imgsarray)
    #print(np.max(valid_hr_imgsarray))
    #print(np.min(valid_hr_imgsarray))
    valid_hr_imgs = []
    i = 0
    
    
    while i < valid_hr_imgsarray.shape[2]:
        valid_hr_imgs.append(valid_hr_imgsarray[:,:,i:i+1])
        i = i+1
    
    idx = 300
    '''
    b_imgs_384 = tl.prepro.threading_data(      #使用了python threading， 多线程处理
                    valid_hr_imgs[idx : idx + batch_size],
                    fn=crop_sub_imgs_fn, is_random=False)
    '''
    
    b_imgs_384 = tl.prepro.threading_data(      #使用了python threading， 多线程处理
                    valid_hr_imgs[idx : idx + 50],
                    fn=crop_sub_imgs_fn, is_random=False)
    b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
    del valid_hr_imgs
    print("this is the maximum and minmum")
    print(np.min(b_imgs_384))  #-1.0
    print(np.max(b_imgs_384))  #-0.2238000000000001
    print(np.min(b_imgs_96))  #-1.0047235679672717
    print(np.max(b_imgs_96))  #-0.2571468398329928
    print(b_imgs_384.shape)   #(50, 384, 384, 1)
    
    
    img384 = np.arange(b_imgs_384.shape[0]*b_imgs_384.shape[1]*b_imgs_384.shape[2]).reshape(b_imgs_384.shape[1], b_imgs_384.shape[2], b_imgs_384.shape[0])
    img96 = np.arange(b_imgs_96.shape[0]*b_imgs_96.shape[1]*b_imgs_96.shape[2]).reshape(b_imgs_96.shape[1], b_imgs_96.shape[2], b_imgs_96.shape[0])
    
    img384.dtype = b_imgs_384.dtype
    img96.dtype = b_imgs_96.dtype
    
    print(img384.shape) #(384, 384, 100)
    print(img96.shape) #(96,96,100)
    
    for i in range(1, b_imgs_384.shape[0]+1):
        img384[:,:,i-1] = b_imgs_384[i-1,:,:,0]
        img96[:,:,i-1] = b_imgs_96[i-1,:,:,0]
        
    img96 = nib.Nifti1Image(img96, np.eye(4))
    img384 = nib.Nifti1Image(img384, np.eye(4))
    nib.save(img96, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/_test_sample_96.nii.gz')
    nib.save(img384, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/_test_sample_384.nii.gz')
    
    valid_hr_img = b_imgs_384
    valid_lr_img = b_imgs_96
    
    print('==============================')
    print(np.min(valid_lr_img))
    print(np.max(valid_lr_img))
    
    size = valid_lr_img.shape 
    
    
    t_image = tf.placeholder('float32', [size[0], size[1], size[2], size[3]], name='input_image')
    densenet = DenseUNET(t_image, is_train=False, reuse=False)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))
    tl.layers.initialize_global_variables(sess)
    
    load_params = tl.files.load_npz(name=checkpoint_dir+'/backup/init500gan2000/g_medicaltrain_init.npz')
    tl.files.assign_params(sess, load_params, network=densenet)
    #tl.files.load_and_assign_npz(sess=sess, name='checkpoint/backup/init500gan2000/g_medicaltrain.npz', network=DenseUNET)
    
    start_time = time.time()
    out = sess.run(densenet.outputs, {t_image: valid_lr_img})
    out = np.float32(out)
    valid_hr_img = np.float32(valid_hr_img)
    
    ##################################
    print("this is the output and the hr")
    print(out.shape)
    print(valid_hr_img.shape)
    print(np.min(out))
    print(np.max(out))
    print(np.min(valid_hr_img))  #1.0
    print(np.max(valid_hr_img))  #1.0
    
    mse = ((valid_hr_img - out) ** 2).mean(axis=None)
    print("this is mse")
    print(mse)  #0.09449510108286935
    psnr = my_psnr(valid_hr_img, out)
    print("this is psnr")
    print(psnr)
    data_out = tf.convert_to_tensor(out, out.dtype)
    data_valid_hr_img = tf.convert_to_tensor(valid_hr_img, valid_hr_img.dtype)
    mse_loss = tl.cost.mean_squared_error(data_out , data_valid_hr_img, is_mean=True)
    print("this is the mse loss:")
    print(mse_loss)
    sess1 = tf.Session()
    # Evaluate the tensor `c`.
    print(sess1.run(mse_loss))  #0.09449510108286927
    
    #print(out.shape) #4,384,384,3
    imgout = np.arange(out.shape[0]*out.shape[1]*out.shape[2]*out.shape[3]).reshape(out.shape[1], out.shape[2], out.shape[0]*out.shape[3])
    imgout.dtype = out.dtype
    for i in range(1, out.shape[0]+1):
        for j in range(1, out.shape[3]+1):
            imgout[:,:,(i-1)*out.shape[3]+j-1] = out[i-1,:,:,j-1]
    imgout = nib.Nifti1Image(imgout, np.eye(4))
    nib.save(imgout, '/homes/tzheng/Mypythonfiles/densunetdiscirminator/samples/medicaltest/_out.nii.gz')
    
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape)) # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)

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
    args.mode = 'medicalevaluate'
    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'medicaltrainDenseUNET':
        medicaltrain()
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
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print(type(sample_imgs_384))
    print(sample_imgs_384.shape)
    '''
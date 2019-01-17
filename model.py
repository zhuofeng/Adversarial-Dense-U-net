# -*- coding: utf8 -*-
import tensorflow as tf 
import tensorlayer as tl
from tensorlayer.layers import *
import time
from time import localtime, strftime
from config import config, log_config


BASE = 32  #now is 32

def SRGAN_d(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv2d(net_in, df_dim, (4, 4), (2, 2), act=lrelu, padding='SAME', W_init=w_init, name='h0/c')
        
        #假如：原来是（1，5，5，5），经过Conv2d之后结果是：（4，4，5，64）
        net_h1 = Conv2d(net_h0, df_dim * 2, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv2d(net_h1, df_dim * 4, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h2/c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv2d(net_h2, df_dim * 8, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv2d(net_h3, df_dim * 16, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv2d(net_h4, df_dim * 32, (4, 4), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv2d(net_h5, df_dim * 16, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv2d(net_h6, df_dim * 8, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, is_train=is_train, gamma_init=gamma_init, name='h7/bn')

        net = Conv2d(net_h7, df_dim * 2, (1, 1), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
        net = Conv2d(net, df_dim * 2, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
        net = Conv2d(net, df_dim * 8, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)

        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return net_ho, logits
'''
def DenseUNET(input_images, is_train=True, reuse=False):
    #how to merge the layers?
    #deconv3_2 = tl.layers.ConcatLayer([conv4, deconv3], concat_dim=3, name='concat3_2')
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("DenseUNET", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        #preout2 = SubpixelConv2d(preout1, scale=2, n_out_channel=None, act=tf.nn.relu, name='upsampling2/2')
        
        #now start the denseunet
        
        conv11 = Conv2d(net_in, BASE, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv111')
        conv11 = BatchNormLayer(conv11, is_train=is_train, gamma_init = gamma_init, name='conv112')
        conc11 = ConcatLayer([net_in, conv11], 3, name='concat11')
        #print(tf.shape(conc11))  #[4, 96, 96, 33]
        conv12 = Conv2d(conc11, BASE, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv121')
        conv12 = BatchNormLayer(conv12, is_train=is_train, gamma_init = gamma_init, name='conv122')
        conc12 = ConcatLayer([net_in, conv12], 3, name='concat12')
        pool1 = MaxPool2d(conc12, (3,3), (2,2), padding='SAME', name='pool1')
        
        conv21 = Conv2d(pool1, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv211')
        conv21 = BatchNormLayer(conv21, is_train=is_train, gamma_init = gamma_init, name = 'conv212')
        conc21 = ConcatLayer([pool1, conv21], 3, name='concat21')
        conv22 = Conv2d(conc21, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv221')
        conv22 = BatchNormLayer(conv22, is_train=is_train, gamma_init = gamma_init, name = 'conv222')
        conc22 = ConcatLayer([pool1, conv22], 3, name='concat22')
        pool2 = MaxPool2d(conc22, (3,3), (2,2), padding='SAME', name='pool2')
        
        conv31 = Conv2d(pool2, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv311')
        conv31 = BatchNormLayer(conv31, is_train=is_train, gamma_init = gamma_init, name = 'conv312')
        conc31 = ConcatLayer([pool2, conv31], 3, name='concat31')
        conv32 = Conv2d(conc31, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv321')
        conv32 = BatchNormLayer(conv32, is_train=is_train, gamma_init = gamma_init, name = 'conv322')
        conc32 = ConcatLayer([pool2, conv32], 3, name='concat32')
        pool3 = MaxPool2d(conc32, (3,3), (2,2), padding='SAME', name='pool3')
        
        conv41 = Conv2d(pool3, BASE*8, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv411')
        conv41 = BatchNormLayer(conv41, is_train=is_train, gamma_init = gamma_init, name = 'conv412')
        conc41 = ConcatLayer([pool3, conv41], 3, name='concat41')
        conv42 = Conv2d(conc41, BASE*8, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv421')
        conv42 = BatchNormLayer(conv42, is_train=is_train, gamma_init = gamma_init, name = 'conv422')
        conc42 = ConcatLayer([pool3, conv42], 3, name='concat42')
        #pool4 = MaxPool2d(conv42, (3,3), (2,2), padding='SAME', data_format='channels_last', name='pool4')
        
        #the right-side of the u-net
        #DeConv2d
        up7 = DeConv2d(conc42, 128, (3,3), (2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up711')
        #up7 = SubpixelConv2d(conc42, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        up7 = ConcatLayer([up7, conv32], 3, name='up712')
        conv71 = Conv2d(up7, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv711')
        conv71 = BatchNormLayer(conv71, is_train=is_train, gamma_init = gamma_init, name = 'conv712')
        conc71 = ConcatLayer([up7, conv71], 3, name='concat71')
        conv72 = Conv2d(conc71, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv721')
        conv72 = BatchNormLayer(conv72, is_train=is_train, gamma_init = gamma_init, name = 'conv722')
        conc72 = ConcatLayer([up7, conv72], 3, name='concat72')
        
        up8 = DeConv2d(conc72, 64, (3,3), (2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up811')
        #up8 = SubpixelConv2d(conc72, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
        up8 = ConcatLayer([up8, conv22], 3, name='up812')
        conv81 = Conv2d(up8, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv811')
        conv81 = BatchNormLayer(conv81, is_train=is_train, gamma_init = gamma_init, name = 'conv812')
        conc81 = ConcatLayer([up8, conv81], 3, name='concat81')
        conv82 = Conv2d(conc81, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv821')
        conv82 = BatchNormLayer(conv82, is_train=is_train, gamma_init = gamma_init, name = 'conv822')
        conc82 = ConcatLayer([up8, conv82], 3, name='concat82')
        
        up9 = DeConv2d(conc82, 32, (3,3), (2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up911')
        up9 = ConcatLayer([up9, conv12], 3, name='up912')
        conv91 = Conv2d(up9, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv911')
        conv91 = BatchNormLayer(conv91, is_train=is_train, gamma_init = gamma_init, name = 'conv912')
        conc91 = ConcatLayer([up9, conv91], 3, name='concat91')
        conv92 = Conv2d(conc91, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv921')
        conv92 = BatchNormLayer(conv92, is_train=is_train, gamma_init = gamma_init, name = 'conv922')
        conc92 = ConcatLayer([up9, conv92], 3, name='concat92')
        
        #4*upsampling
        preout = SubpixelConv2d(conc92, scale=2, n_out_channel=None, act=tf.nn.relu, name='up1')
        preout = Conv2d(preout, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up1/conv2d')  # <-- may need to increase n_filter
        preout = BatchNormLayer(preout, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up1/batch_norm')
        
        preout = SubpixelConv2d(preout, scale=2, n_out_channel=None, act=tf.nn.relu, name='up2')
        preout = Conv2d(preout, 64, (3, 3), (1, 1), padding='SAME', W_init=w_init, b_init=b_init, name='up2/conv2d')  # <-- may need to increase n_filter
        preout = BatchNormLayer(preout, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up2/batch_norm')
        
        out = Conv2d(preout, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')  #shape:[1, 1, 64, 3]
        return out
'''

def DenseUNET(input_images, is_train=True, reuse=False):
    #how to merge the layers?
    #deconv3_2 = tl.layers.ConcatLayer([conv4, deconv3], concat_dim=3, name='concat3_2')
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("DenseUNET", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        #preout2 = SubpixelConv2d(preout1, scale=2, n_out_channel=None, act=tf.nn.relu, name='upsampling2/2')
        
        #now start the denseunet
        
        conv11 = Conv2d(net_in, BASE, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv111')
        conv11 = BatchNormLayer(conv11, is_train=is_train, gamma_init = gamma_init, name='conv112')
        conc11 = ConcatLayer([net_in, conv11], 3, name='concat11')
        #print(tf.shape(conc11))  #[4, 96, 96, 33]
        conv12 = Conv2d(conc11, BASE, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv121')
        conv12 = BatchNormLayer(conv12, is_train=is_train, gamma_init = gamma_init, name='conv122')
        conc12 = ConcatLayer([net_in, conv12], 3, name='concat12')
        pool1 = MaxPool2d(conc12, (3,3), (2,2), padding='SAME', name='pool1')
        
        conv21 = Conv2d(pool1, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv211')
        conv21 = BatchNormLayer(conv21, is_train=is_train, gamma_init = gamma_init, name = 'conv212')
        conc21 = ConcatLayer([pool1, conv21], 3, name='concat21')
        conv22 = Conv2d(conc21, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv221')
        conv22 = BatchNormLayer(conv22, is_train=is_train, gamma_init = gamma_init, name = 'conv222')
        conc22 = ConcatLayer([pool1, conv22], 3, name='concat22')
        pool2 = MaxPool2d(conc22, (3,3), (2,2), padding='SAME', name='pool2')
        
        conv31 = Conv2d(pool2, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv311')
        conv31 = BatchNormLayer(conv31, is_train=is_train, gamma_init = gamma_init, name = 'conv312')
        conc31 = ConcatLayer([pool2, conv31], 3, name='concat31')
        conv32 = Conv2d(conc31, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv321')
        conv32 = BatchNormLayer(conv32, is_train=is_train, gamma_init = gamma_init, name = 'conv322')
        conc32 = ConcatLayer([pool2, conv32], 3, name='concat32')
        pool3 = MaxPool2d(conc32, (3,3), (2,2), padding='SAME', name='pool3')
        
        conv41 = Conv2d(pool3, BASE*8, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv411')
        conv41 = BatchNormLayer(conv41, is_train=is_train, gamma_init = gamma_init, name = 'conv412')
        conc41 = ConcatLayer([pool3, conv41], 3, name='concat41')
        conv42 = Conv2d(conc41, BASE*8, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv421')
        conv42 = BatchNormLayer(conv42, is_train=is_train, gamma_init = gamma_init, name = 'conv422')
        conc42 = ConcatLayer([pool3, conv42], 3, name='concat42')
        #pool4 = MaxPool2d(conv42, (3,3), (2,2), padding='SAME', data_format='channels_last', name='pool4')
        
        #the right-side of the u-net
        #DeConv2d
        up7 = DeConv2d(conc42, 128, (3,3), (2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up711')
        #up7 = SubpixelConv2d(conc42, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        up7 = ConcatLayer([up7, conv32], 3, name='up712')
        conv71 = Conv2d(up7, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv711')
        conv71 = BatchNormLayer(conv71, is_train=is_train, gamma_init = gamma_init, name = 'conv712')
        conc71 = ConcatLayer([up7, conv71], 3, name='concat71')
        conv72 = Conv2d(conc71, BASE*4, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv721')
        conv72 = BatchNormLayer(conv72, is_train=is_train, gamma_init = gamma_init, name = 'conv722')
        conc72 = ConcatLayer([up7, conv72], 3, name='concat72')
        
        up8 = DeConv2d(conc72, 64, (3,3), (2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up811')
        #up8 = SubpixelConv2d(conc72, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')
        up8 = ConcatLayer([up8, conv22], 3, name='up812')
        conv81 = Conv2d(up8, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv811')
        conv81 = BatchNormLayer(conv81, is_train=is_train, gamma_init = gamma_init, name = 'conv812')
        conc81 = ConcatLayer([up8, conv81], 3, name='concat81')
        conv82 = Conv2d(conc81, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv821')
        conv82 = BatchNormLayer(conv82, is_train=is_train, gamma_init = gamma_init, name = 'conv822')
        conc82 = ConcatLayer([up8, conv82], 3, name='concat82')
        
        up9 = DeConv2d(conc82, 32, (3,3), (2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up911')
        up9 = ConcatLayer([up9, conv12], 3, name='up912')
        conv91 = Conv2d(up9, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv911')
        conv91 = BatchNormLayer(conv91, is_train=is_train, gamma_init = gamma_init, name = 'conv912')
        conc91 = ConcatLayer([up9, conv91], 3, name='concat91')
        conv92 = Conv2d(conc91, BASE*2, (3,3), (1,1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv921')
        conv92 = BatchNormLayer(conv92, is_train=is_train, gamma_init = gamma_init, name = 'conv922')
        conc92 = ConcatLayer([up9, conv92], 3, name='concat92')
        
        out = Conv2d(conc92, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')  #shape:[1, 1, 64, 3]
        return out
def DenseUNET3D(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    with tf.variable_scope("DenseUNET3D", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        '''
        up1 = DeConv3d(net_in, 1, (3,3,3), (2,2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up1')
        up1 = BatchNormLayer(up1, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up1/batch_norm')
        
        up2 = DeConv3d(up1, 1, (3,3,3), (2,2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up2')
        up2 = BatchNormLayer(up2, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up2/batch_norm')
        
        up3 = DeConv3d(up2, 1, (3,3,3), (2,2,2), padding='SAME', act=None, W_init=w_init, b_init=b_init, name='up3')
        up3 = BatchNormLayer(up3, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='up3/batch_norm')
        '''
        conv11 = Conv3dLayer(net_in, act=tf.nn.relu, shape=[3,3,3,1,BASE], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv111')
        conv11 = BatchNormLayer(conv11, is_train=is_train, gamma_init = gamma_init, name='conv112')
        conc11 = ConcatLayer([net_in, conv11], 4, name='concat11')
        conv12 = Conv3dLayer(conc11, act=tf.nn.relu, shape=[3,3,3,BASE+1,BASE], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv121')
        conv12 = BatchNormLayer(conv11, is_train=is_train, gamma_init = gamma_init, name='conv122')
        conc12 = ConcatLayer([net_in, conv12], 4, name='concat12')
        pool1 = MaxPool3d(conc12, (3,3,3), (2,2,2), padding='SAME', name='pool1')
        
        conv21 = Conv3dLayer(pool1, act=tf.nn.relu, shape=[3,3,3,BASE+1,BASE*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv211')
        conv21 = BatchNormLayer(conv21, is_train=is_train, gamma_init = gamma_init, name='conv212')
        conc21 = ConcatLayer([pool1, conv21], 4, name='concat21')
        conv22 = Conv3dLayer(conc21, act=tf.nn.relu, shape=[3,3,3,BASE*3+1,BASE*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv221')
        conv22 = BatchNormLayer(conv22, is_train=is_train, gamma_init = gamma_init, name='conv222')
        conc22 = ConcatLayer([pool1, conv22], 4, name='concat22')
        pool2 = MaxPool3d(conc22, (3,3,3), (2,2,2), padding='SAME', name='pool2')
        
        conv31 = Conv3dLayer(pool2, act=tf.nn.relu, shape=[3,3,3,BASE*3+1,BASE*4], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv311')
        conv31 = BatchNormLayer(conv31, is_train=is_train, gamma_init = gamma_init, name='conv312')
        conc31 = ConcatLayer([pool2, conv31], 4, name='concat31')
        conv32 = Conv3dLayer(conc31, act=tf.nn.relu, shape=[3,3,3,BASE*7+1,BASE*4], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv321')
        conv32 = BatchNormLayer(conv32, is_train=is_train, gamma_init = gamma_init, name='conv322')
        conc32 = ConcatLayer([pool2, conv32], 4, name='concat32')
        pool3 = MaxPool3d(conc32, (3,3,3), (2,2,2), padding='SAME', name='pool3')
        
        conv41 = Conv3dLayer(pool3, act=tf.nn.relu, shape=[3,3,3,BASE*7+1,BASE*8], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv411')
        conv41 = BatchNormLayer(conv41, is_train=is_train, gamma_init = gamma_init, name='conv412')
        conc41 = ConcatLayer([pool3, conv41], 4, name='concat41')
        conv42 = Conv3dLayer(conc41, act=tf.nn.relu, shape=[3,3,3,BASE*15+1,BASE*8], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv421')
        conv42 = BatchNormLayer(conv42, is_train=is_train, gamma_init = gamma_init, name='conv422')
        conc42 = ConcatLayer([pool3, conv42], 4, name='concat42')  #BASE*15 + 1
        
        up7 = DeConv3d(conc42, 128, (3,3,3), (2,2,2), padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='up711')
        up7 = ConcatLayer([up7, conv32], 4, name='up712')  
        conv71 = Conv3dLayer(up7, act=tf.nn.relu, shape=[3,3,3,BASE*8,BASE*4], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv711')
        conv71 = BatchNormLayer(conv71, is_train=is_train, gamma_init = gamma_init, name = 'conv712')
        conc71 = ConcatLayer([up7, conv71], 4, name='concat71')
        conv72 = Conv3dLayer(conc71, act=tf.nn.relu, shape=[3,3,3,BASE*12,BASE*4], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv721')
        conv72 = BatchNormLayer(conv72, is_train=is_train, gamma_init = gamma_init, name = 'conv722')
        conc72 = ConcatLayer([up7, conv72], 4, name='concat72')
        
        up8 = DeConv3d(conc72, 64, (3,3,3), (2,2,2), padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='up811')
        up8 = ConcatLayer([up8, conv22], 4, name='up812')
        conv81 = Conv3dLayer(up8, act=tf.nn.relu, shape=[3,3,3,BASE*4,BASE*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv811')
        conv81 = BatchNormLayer(conv81, is_train=is_train, gamma_init = gamma_init, name = 'conv812')
        conc81 = ConcatLayer([up8, conv81], 4, name='concat81')
        conv82 = Conv3dLayer(conc81, act=tf.nn.relu, shape=[3,3,3,BASE*6,BASE*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv821')
        conv82 = BatchNormLayer(conv82, is_train=is_train, gamma_init = gamma_init, name = 'conv822')
        conc82 = ConcatLayer([up8, conv82], 4, name='concat82')
        
        up9 = DeConv3d(conc82, 32, (3,3,3), (2,2,2), padding='SAME', act=tf.nn.relu, W_init=w_init, b_init=b_init, name='up911')
        up9 = ConcatLayer([up9, conv12], 4, name='up912')
        conv91 = Conv3dLayer(up9, act=tf.nn.relu, shape=[3,3,3,BASE*2,BASE], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv911')
        conv91 = BatchNormLayer(conv91, is_train=is_train, gamma_init = gamma_init, name = 'conv912')
        conc91 = ConcatLayer([up9, conv91], 4, name='concat91')
        conv92 = Conv3dLayer(conc91, act=tf.nn.relu, shape=[3,3,3,BASE*3,BASE], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='conv921')
        conv92 = BatchNormLayer(conv92, is_train=is_train, gamma_init = gamma_init, name = 'conv922')
        conc92 = ConcatLayer([up9, conv92], 4, name='concat92')
        
        out = Conv3dLayer(conc92, act=tf.nn.relu, shape=[1,1,1,BASE*3,1], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='out') #(4,128,128,128,1)
        #out = tf.slice(out, [0, 0, 0, 0, 0], [config.BATCH_SIZE, config.PATCH_SIZE, config.PATCH_SIZE, config.PATCH_SIZE, 1])
        return out
    
def DenseUNET3Dkeras():
    inputs = Input(shape=self.lr_shape)
    conv11 = Conv3D(BASE, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv11 = BatchNormalization()(conv11)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(BASE, (3, 3, 3), activation='relu', padding='same')(conc11)
    conv12 = BatchNormalization()(conv12)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(BASE*2, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv21 = BatchNormalization()(conv21)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(BASE*2, (3, 3, 3), activation='relu', padding='same')(conc21)
    conv22 = BatchNormalization()(conv22)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(BASE*4, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv31 = BatchNormalization()(conv31)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(BASE*4, (3, 3, 3), activation='relu', padding='same')(conc31)
    conv32 = BatchNormalization()(conv32)
    conc32 = concatenate([pool2, conv32], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

    conv41 = Conv3D(BASE*8, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv41 = BatchNormalization()(conv41)
    conc41 = concatenate([pool3, conv41], axis=4)
    conv42 = Conv3D(BASE*8, (3, 3, 3), activation='relu', padding='same')(conc41)
    conv42 = BatchNormalization()(conv42)
    conc42 = concatenate([pool3, conv42], axis=4)
    '''
    pool4 = MaxPooling2D(pool_size=(2, 2))(conc42)

    conv51 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(pool4)
    conv51 = BatchNormalization()(conv51)
    conc51 = concatenate([pool4, conv51], axis=3)
    conv52 = Conv2D(BASE*16, (3, 3), activation='relu', padding='same')(conc51)
    conv52 = BatchNormalization()(conv52)
    conc52 = concatenate([pool4, conv52], axis=3)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conc52), conc42], axis=3)
    conv61 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(up6)
    conv61 = BatchNormalization()(conv61)
    conc61 = concatenate([up6, conv61], axis=3)
    conv62 = Conv2D(BASE*8, (3, 3), activation='relu', padding='same')(conc61)
    conv62 = BatchNormalization()(conv62)
    conc62 = concatenate([up6, conv62], axis=3)
    '''
    #反卷积，这一步的目的是将图片放大，相当于upsamling
    up7 = concatenate([Conv2DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc42), conv32], axis=4)
    conv71 = Conv3D(BASE*4, (3, 3, 3), activation='relu', padding='same')(up7)
    conv71 = BatchNormalization()(conv71)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(BASE*4, (3, 3 ,3), activation='relu', padding='same')(conc71)
    conv72 = BatchNormalization()(conv72)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv2DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
    conv81 = Conv3D(BASE*2, (3, 3, 3), activation='relu', padding='same')(up8)
    conv81 = BatchNormalization()(conv81)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(BASE*2, (3, 3, 3), activation='relu', padding='same')(conc81)
    conv82 = BatchNormalization()(conv82)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv2DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    conv91 = Conv3D(BASE, (3, 3, 3), activation='relu', padding='same')(up9)
    conv91 = BatchNormalization()(conv91)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(BASE, (3, 3, 3), activation='relu', padding='same')(conc91)
    conv92 = BatchNormalization()(conv92)
    conc92 = concatenate([up9, conv92], axis=4)
    
    
    outputs = Conv3D(1, (3, 3, 3), activation='relu', padding='same')(conc91)
    

    model = Model(inputs, outputs)
    return model
def SRGAN_d3D(input_images, is_train=True, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    
    with tf.variable_scope("SRGAN_d3D", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net_in = InputLayer(input_images, name='input/images')
        net_h0 = Conv3dLayer(net_in, act=None, shape=[4,4,4,1,df_dim], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h0/c')
        print()
        net_h1 = Conv3dLayer(net_h0, act=None, shape=[4,4,4,df_dim,df_dim*2], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h1/c')
        net_h1 = BatchNormLayer(net_h1, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h1/bn')
        net_h2 = Conv3dLayer(net_h1, act=None, shape=[4,4,4,df_dim*2,df_dim*4], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h2c')
        net_h2 = BatchNormLayer(net_h2, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h2/bn')
        net_h3 = Conv3dLayer(net_h2, act=None, shape=[4,4,4,df_dim*4,df_dim*8], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h3/c')
        net_h3 = BatchNormLayer(net_h3, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h3/bn')
        net_h4 = Conv3dLayer(net_h3, act=None, shape=[4,4,4,df_dim*8,df_dim*16], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h4/c')
        net_h4 = BatchNormLayer(net_h4, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h4/bn')
        net_h5 = Conv3dLayer(net_h4, act=None, shape=[4,4,4,df_dim*16,df_dim*32], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h5/c')
        net_h5 = BatchNormLayer(net_h5, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h5/bn')
        net_h6 = Conv3dLayer(net_h5, act=None, shape=[4,4,4,df_dim*32,df_dim*16], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
        net_h6 = BatchNormLayer(net_h6, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h6/bn')
        net_h7 = Conv3dLayer(net_h6, act=None, shape=[4,4,4,df_dim*16,df_dim*8], strides=[1,2,2,2,1], padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
        net_h7 = BatchNormLayer(net_h7, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='h7/bn')
        
        net = Conv3dLayer(net_h7, act=None, shape=[1,1,1,df_dim*8,df_dim*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='res/c')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn')
        net = Conv3dLayer(net, act=None, shape=[3,3,3,df_dim*2,df_dim*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
        net = BatchNormLayer(net, act=lrelu, is_train=is_train, gamma_init=gamma_init, name='res/bn2')
        net = Conv3dLayer(net, act=None, shape=[3,3,3,df_dim*2,df_dim*8], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
        net = BatchNormLayer(net, is_train=is_train, gamma_init=gamma_init, name='res/bn3')
        net_h8 = ElementwiseLayer([net_h7, net], combine_fn=tf.add, name='res/add')
        net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)
        
        net_ho = FlattenLayer(net_h8, name='ho/flatten')
        net_ho = DenseLayer(net_ho, n_units=1, act=tf.identity, W_init=w_init, name='ho/dense')
        logits = net_ho.outputs
        net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)
        
    return net_ho, logits

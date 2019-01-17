from easydict import EasyDict as edict
import json
import os
config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8
config.TRAIN.lr_init = 1e-5
config.TRAIN.beta1 = 0.9
config.PATCH_SIZE = 64
## initialize G
config.TRAIN.n_epoch_init = 10
#config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.batchnum = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'data2017/DIV2K_train_HR/'
config.TRAIN.lr_img_path = 'data2017/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = 'data2017/DIV2K_valid_HR/'
config.VALID.lr_img_path = 'data2017/DIV2K_valid_LR_bicubic/X4/'

config.VALID.logdir = 'log/'

#the path of the hoshisuna image and lung image
config.TRAIN.hoshisuna_path = '/homes/tzheng/CTdata/hoshisunanii/hoshi000005.nii.gz'
config.TRAIN.medical_path1 = '/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung030_dual_cb003_zf_ringRem_med3.nii.gz'
config.TRAIN.medical_path2 = '/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung025_cb_005_zf_ringRem_med3.nii.gz'
config.TRAIN.medical_path3 = '/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung018-a_cb_001_zf_ringRem_med3.nii.gz'
config.TRAIN.medical_path4 = '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung017/uCT/nulung017_052_000.nii.gz'
config.TRAIN.medical_path5 = '/homes/tzheng/CTdata/CTMicroNUrespsurg/nii/nulung014/uCT/nulung014_027_000.nii.gz'
config.VALID.medical_path =  '/homes/tzheng/CTdata/CTMicroNUrespsurg/converted/DICOM_nulung026_cb_003_zf_ringRem.nii.gz'
config.TRAIN.all_medical_path = []
config.TRAIN.all_medical_path.append(config.TRAIN.medical_path1)
config.TRAIN.all_medical_path.append(config.TRAIN.medical_path2)
config.TRAIN.all_medical_path.append(config.TRAIN.medical_path3)
config.TRAIN.all_medical_path.append(config.TRAIN.medical_path4)
config.TRAIN.all_medical_path.append(config.TRAIN.medical_path5)
 

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")

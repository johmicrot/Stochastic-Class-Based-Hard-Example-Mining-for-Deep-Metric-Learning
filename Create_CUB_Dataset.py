import numpy as np
import glob
import cv2
import os

# Output image size
im_size = 256

# Location of the CUB dataset
source_Dir = 'CUB_200_2011/images/*'
out_dir = 'CUB_as_npy'
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
all_classes = []
cls_l = []
images = []
tstx = []
tsty = []
trnx = []
trny = []
trny_OH = []
num_train_classes = 100
for cls_folder in glob.glob(source_Dir):
    print(cls_folder)
    cls = cls_folder.split('/')[-1]
    cls_number = int(cls.split('.')[0]) - 1

    for img_fl in glob.glob(cls_folder + '/*'):
        img_name = img_fl.split('/')[-1]
        img = cv2.imread(img_fl)
        img = cv2.resize(img, (im_size, im_size))
        if cls_number < num_train_classes:
            trnx.append(img)
            cls_OH = [0] * num_train_classes
            cls_OH[cls_number] = 1
            trny_OH.append(cls_OH)
            trny.append(cls)
        elif cls_number < num_train_classes * 2:
            tstx.append(img)
            tsty.append(cls)
trnx = np.array(trnx)
trny = np.array(trny)
trny_OH = np.array(trny_OH)
tstx = np.array(tstx)
tsty = np.array(tsty)

np.save('%s/trnx_CUB(%i)_%s' % (out_dir, num_train_classes, im_size), trnx)
np.save('%s/trny_CUB(%i)_%s' % (out_dir, num_train_classes, im_size), trny)
# Also save the train dataset as One hot encoded
np.save('%s/trny_OH_CUB(%i)_%s' % (out_dir, num_train_classes, im_size), trny_OH)
np.save('%s/tstx_CUB(%i)_%s' % (out_dir, num_train_classes, im_size), tstx)
np.save('%s/tsty_CUB(%i)_%s' % (out_dir, num_train_classes, im_size), tsty)


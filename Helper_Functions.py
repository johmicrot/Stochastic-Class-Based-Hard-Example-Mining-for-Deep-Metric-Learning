import numpy as np
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def process_input(sys):
    parser = argparse.ArgumentParser(description='Arguments for implementation of SCHEM paper')
    parser.add_argument('-end_lyr', default=4,  help='Layers to remove from MobilenetV2')
    parser.add_argument('-sclCS', default=0, type=int,help='Paper specifies the CS weights should have a unit norm')
    parser.add_argument('-l2FE', default=0, help='Boolean option to l2 normalize feature map G, as specified in the paper',type=int)
    parser.add_argument('-out_features', default=320, type=int, help='number of features for feature map G, '
                                                                     'is correlated to the end_lyr value '
                                                                     'and MobilenetV2')
    parser.add_argument('-runname', required=True, type=str)
    parser.add_argument('-dataset', default='CUB')
    parser.add_argument('-epocs', default=100, type=int)
    parser.add_argument('-img_load_size', default=110, type=int, help='Size of the image to be loaded in, paper uses 256')
    parser.add_argument('-img_crop_size', default=96, type=int, help='Crop size before feeing into model, paper uses 224')
    parser.add_argument('-inital_weights', default='imagenet', type=str)
    parser.add_argument('-img_location', default='CUB_as_npy', type=str, help='Location where the .npy file is located')
    parser.add_argument('-saveW', default=0, help='Boolean to save model weights')

    args = vars(parser.parse_args())
    return args


def random_flip_QPN(q, p, n):
    if np.random.random() > 0.5:
      q = np.fliplr(q)
    if np.random.random() > 0.5:
      p = np.fliplr(p)
    if np.random.random() > 0.5:
      n = np.fliplr(n)
    return q, p, n


def distance_calc(args):
    x, y, z = args
    pos_mag = tf.math.sqrt(tf.reduce_sum(tf.math.square(x - y)))
    neg_mag = tf.math.sqrt(tf.reduce_sum(tf.math.square(x - z)))

    return neg_mag - pos_mag


def make_dirs(dirs_list):
    for directory in dirs_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
    return
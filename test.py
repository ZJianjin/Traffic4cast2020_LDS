import random
from random import shuffle
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import datetime
import time
import queue
import threading
import logging
from PIL import Image
import itertools
import yaml
import re
import os
import glob
import shutil
import sys
import copy
import h5py
from net_all import *
from trainer_all import *

season = None
use_mask = True
use_flip = False
use_time = True
model_name = 'neta'

train_winter = ['-01-', '-02-', '-03-']
train_summer = ['-05-', '-04-', '-06-']
test_winter = ['-11-', '-12-']
test_summer = ['-07-', '-08-', '-09-', '-10-']

SEED = 0
num_train_file = 285
num_frame_per_day = 288
num_frame_before = 12
num_frame_sequence = 24
target_frames = [0, 1, 2, 5, 8, 11]
num_sequence_per_day = num_frame_per_day - num_frame_sequence + 1
height = 495
width = 436
num_channel = 9
num_channel_discretized = 8  # 4 * 2
visual_input_channels = 115  # 12 * 8
visual_output_channels = 6 * 8  # 6 * 8
vector_input_channels = 1  # start time point
import json
#
n = 1
s = 255
e = 85
w = 170
tv = 16

##############################Set the path##############################################
data_root = './data'
model_root = './jianjzhmodelstest'
log_root = './output'
##############################Set the path##############################################
#

target_city = 'ISTANBUL'  # ['BERLIN', 'MOSCOW', 'ISTANBUL']
# test_start_index_list = np.array([ 18,  57, 114, 174, 222], np.int32)    # 'BERLIN' 
# test_start_index_list = np.array([ 45, 102, 162, 210, 246], np.int32)   # 'Moscow' # 'Istanbul'
input_static_data_path = data_root + '/' + target_city + '/' + target_city + '_static_2019.h5'
input_mask_data_path = data_root + '/maskdata/'
input_train_data_folder_path = data_root + '/' + target_city + '/training'
input_val_data_folder_path = data_root + '/' + target_city + '/validation'
input_test_data_folder_path = data_root + '/' + target_city + '/testing'
save_model_path = model_root + '/' + target_city + str(season) + str(use_flip) + str(use_mask)
summary_path = log_root + '/' + target_city + str(season) + str(use_flip) + str(use_mask)
#
batch_size_test = 5
learning_rate = 3e-4
load_model_path = model_root + '/' + 'ISTANBULneta'
# load_model_path = ''
is_training = False
# premodel = os.path.join(model_root, 'BERLINneta', 'model-58000.cptk')
global_step = 60000


def write_data(data, filename):
    f = h5py.File(filename, 'w', libver='latest')
    dset = f.create_dataset('array', shape=(data.shape), data=data, compression='gzip', compression_opts=9)
    f.close()


def get_data_filepath_list(input_data_folder_path):
    data_filepath_list = []
    for filename in os.listdir(input_data_folder_path):
        if filename.split('.')[-1] != 'h5':
            continue
        data_filepath_list.append(os.path.join(input_data_folder_path, filename))
    data_filepath_list = sorted(data_filepath_list)

    return data_filepath_list


def get_static_data(input_static_data_path):
    fr = h5py.File(input_static_data_path, 'r')
    data = fr['array'].value / 255.0
    return data


def get_mask_data(input_mask_data_path, city):
    map_0 = np.load(input_mask_data_path + city + 'map_0.npy')
    map_1 = np.load(input_mask_data_path + city + 'map_1.npy')
    map_2 = np.load(input_mask_data_path + city + 'map_2.npy')
    map_3 = np.load(input_mask_data_path + city + 'map_3.npy')
    result = np.concatenate([map_0, map_0, map_1, map_1, map_2, map_2, map_3, map_3], axis=-1)
    return result


if __name__ == '__main__':

    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    trainer = Trainer(height, width, visual_input_channels, visual_output_channels, vector_input_channels,
                      learning_rate,
                      save_model_path, load_model_path, summary_path, is_training, use_mask, model_name)
    tf.reset_default_graph()

    test_data_filepath_list = get_data_filepath_list(input_test_data_folder_path)
    if season == 'winter':
        tmp = []
        for i in test_data_filepath_list:
            if any([j in i for j in test_winter]):
                tmp.append(i)
        data_filepath_list = tmp
    elif season == 'summer':
        tmp = []
        for i in test_data_filepath_list:
            if any([j in i for j in test_summer]):
                tmp.append(i)
        data_filepath_list = tmp
    print('test_data_filepath_list\t', len(test_data_filepath_list), )

    test_output_filepath_list = list()
    for test_data_filepath in test_data_filepath_list:
        filename = test_data_filepath.split('/')[-1]
        test_output_filepath_list.append('output/' + target_city + '/' + target_city + '_test' + '/' + filename)
    static_data = get_static_data(input_static_data_path)
    mask_data = get_mask_data(input_mask_data_path, target_city)

    try:
        if not os.path.exists('output'):
            os.makedirs('output')
        if not os.path.exists('output/' + target_city):
            os.makedirs('output/' + target_city)
        if not os.path.exists('output/' + target_city + '/' + target_city + '_test'):
            os.makedirs('output/' + target_city + '/' + target_city + '_test')
    except Exception:
        print('output path not made')
        exit(-1)

    with open('test_data.json') as f:
        test_json = json.load(f)

    for i in range(len(test_data_filepath_list)):
        file_path = test_data_filepath_list[i]
        out_file_path = test_output_filepath_list[i]

        fr = h5py.File(file_path, 'r')
        a_group_key = list(fr.keys())[0]
        data = fr[a_group_key]
        # assert data.shape[0] == num_frame_per_day
        data = np.array(data, np.uint8)

        test_data_batch_list = []
        test_data_time_list = []
        test_data_mask_list = []

        batch_size_test = data.shape[0]
        for j in range(batch_size_test):
            test_data_time_list.append(float(j) / float(num_frame_per_day))

        data_sliced = data[:, :, :, :, :num_channel]
        if use_time:
            for time_dict in test_json:
                time_data = list(time_dict.keys())[0]
                if time_data in file_path:
                    time_data = time_dict[time_data]
                    break
            time_id = np.ones_like(data_sliced)[:, :, :, :, :1]
            for m in range(len(time_data)):
                for n in range(num_frame_before):
                    time_id[m, n] = time_id[m, n] * (time_data[m] + n) / 288.0 * 255.0
            data_sliced = np.concatenate([data_sliced, time_id], axis=-1)
        data_mask = (np.max(data_sliced, axis=4) == 0)
        test_data_mask_list = data_mask[:, :, :, :]
        test_data_batch_list.append(data_sliced)

        test_data_time_list = np.asarray(test_data_time_list, np.float32)
        input_time = np.reshape(test_data_time_list, (batch_size_test, 1))

        test_data_mask = test_data_mask_list

        input_data = np.concatenate(test_data_batch_list, axis=0).astype(np.float32)
        input_data[:, :, :, :, :] = input_data[:, :, :, :, :] / 255.0
        input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size_test, height, width, -1))
        static_data_tmp = np.tile(static_data, [batch_size_test, 1, 1, 1])
        input_data = np.concatenate([input_data, static_data_tmp], axis=-1)

        # input_data_mask = np.zeros((batch_size_test, num_frame_before, height, width, num_channel_discretized), np.bool)
        # input_data_mask[test_data_mask[:, :num_frame_before, :, :], :] = True
        # input_data_mask = np.moveaxis(input_data_mask, 1, -1).reshape((batch_size_test, height, width, -1))
        # input_data[input_data_mask] = -1.0

        true_label_mask = np.ones((batch_size_test, height, width, visual_output_channels), dtype=np.float32)
        if use_mask:
          orig_label_mask = np.tile(mask_data, [1, 1, 1, len(target_frames)])
        else:
          orig_label_mask = np.ones((batch_size_test, height, width, visual_output_channels), dtype=np.float32)

        prediction_list = []
        # print(input_data.shape)
        # assert 0
        import scipy.misc as misc

        # trainer.load_model(premodel)
        # print('load model')
        for b in range(batch_size_test):
            run_out_one = trainer.infer(input_data[b, :, :, :][np.newaxis, :, :, :],
                                        input_time[b, :][np.newaxis, :],
                                        true_label_mask[b, :, :, :][np.newaxis, :, :, :], global_step)
            prediction_one = run_out_one['predict']
            prediction_list.append(prediction_one)
            # print(input_data[b,:,:,:].shape)
            # for t in range(3):
            #   misc.imsave('output_'+str(b)+'_'+str(t)+'.png', np.reshape(prediction_one, [495, 436, 3, 8])[:, :, t, 0])
        # assert 0

        prediction = np.concatenate(prediction_list, axis=0)
        prediction = np.moveaxis(np.reshape(prediction, (
        batch_size_test, height, width, num_channel_discretized, len(target_frames),)), -1, 1)
        prediction = prediction.astype(np.float32) * 255.0
        prediction = np.rint(prediction)
        prediction = np.clip(prediction, 0.0, 255.0).astype(np.uint8)

        assert prediction.shape == (batch_size_test, len(target_frames), height, width, num_channel_discretized)

        write_data(prediction, out_file_path)

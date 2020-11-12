#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/12/2020 1:07 PM
# @Author  : Jianjin Zhang
# @File    : validate.py

import random
import h5py
from trainer_all import *

############set path#####################
target_city = 'ISTANBUL'  # city name: ['BERLIN', 'MOSCOW', 'ISTANBUL']
data_root = './data' ## data path
load_model_path = './jianjzhmodelstest6' + '/' + target_city + 'neta' ## model path
#########################################

finetune_on_validation = False
use_mask = True

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
num_channel_target = 8
visual_input_channels = 115
visual_output_channels = 48
vector_input_channels = 1
model_name = 'neta'
model_root = './jianjzhmodels'
log_root = './output'

learning_rate = 3e-4

input_static_data_path = data_root + '/' + target_city + '/' + target_city + '_static_2019.h5'
input_train_data_folder_path = data_root + '/' + target_city + '/training'
if finetune_on_validation:
    input_train_data_folder_path = data_root + '/' + target_city + '/validation'
input_val_data_folder_path = data_root + '/' + target_city + '/validation'
input_test_data_folder_path = data_root + '/' + target_city + '/testing'
save_model_path = model_root + '/' + target_city + model_name + str(finetune_on_validation) + str(learning_rate)
summary_path = log_root + '/' + target_city + model_name + str(finetune_on_validation) + str(learning_rate)

batch_size_val = 1


def get_data_filepath_list(input_data_folder_path):
    data_filepath_list = []
    for filename in os.listdir(input_data_folder_path):
        if filename.split('.')[-1] != 'h5':
            continue
        data_filepath_list.append(os.path.join(input_data_folder_path, filename))
    data_filepath_list = sorted(data_filepath_list)

    return data_filepath_list


def get_data(input_data_folder_path):
    data_filepath_list = []
    for filename in os.listdir(input_data_folder_path):
        if filename.split('.')[-1] != 'h5':
            continue
        data_filepath_list.append(os.path.join(input_data_folder_path, filename))
    data_filepath_list = sorted(data_filepath_list)

    data_np_list = []
    for file_path in data_filepath_list:
        fr = h5py.File(file_path, 'r')
        a_group_key = list(fr.keys())[0]
        data = fr[a_group_key]
        assert data.shape[0] == num_frame_per_day
        data_np_list.append(data[:, :, :, :num_channel])

    return data_np_list


def get_static_data(input_static_data_path):
    fr = h5py.File(input_static_data_path, 'r')
    data = fr['array'].value / 255.0
    return data


def get_mask_data(city):
    map_0 = np.load(city + 'map_0.npy')
    map_1 = np.load(city + 'map_1.npy')
    map_2 = np.load(city + 'map_2.npy')
    map_3 = np.load(city + 'map_3.npy')
    result = np.concatenate([map_0, map_0, map_1, map_1, map_2, map_2, map_3, map_3], axis=-1)
    return result


if __name__ == '__main__':

    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    trainer = Trainer(height, width, visual_input_channels, visual_output_channels, vector_input_channels,
                      learning_rate, save_model_path, load_model_path, summary_path, False, use_mask, model_name)
    tf.reset_default_graph()

    val_data = get_data(input_val_data_folder_path)
    print('val_data\t', len(val_data), '\t', val_data[0].shape, '\tsize:', sys.getsizeof(val_data))

    static_data = get_static_data(input_static_data_path)
    mask_data = get_mask_data(target_city)

    val_set = []
    #
    for i in range(len(val_data)):
        for j in range(0, num_sequence_per_day, num_frame_sequence):
            val_set.append((i, j))
    num_val_iteration_per_epoch = int(len(val_set) / batch_size_val)
    print('num_val_iteration_per_epoch:', num_val_iteration_per_epoch)

    train_input_queue = queue.Queue()
    train_output_queue = queue.Queue()


    eval_loss_list = list()
    for a in range(num_val_iteration_per_epoch):

        val_orig_data_batch_list = []
        val_data_batch_list = []
        val_data_mask_list = []
        val_data_time_list = []
        val_stat_batch_list = []
        for i_j in val_set[a * batch_size_val: (a + 1) * batch_size_val]:
            (i, j) = i_j
            val_data_time_list.append(float(j) / float(num_frame_per_day))
            val_orig_data_batch_list.append(val_data[i][j:j + num_frame_sequence, :, :, :][np.newaxis, :, :, :, :])
            data_sliced = val_data[i][j:j + num_frame_sequence, :, :, :]
            data_mask = (np.max(data_sliced, axis=3) == 0)
            val_data_mask_list.append(data_mask[np.newaxis, :, :, :])

            val_data_batch_list.append(data_sliced[:, :, :, :][np.newaxis, :, :, :, :])

        val_data_time_list = np.asarray(val_data_time_list)
        input_time = np.reshape(val_data_time_list, (batch_size_val, 1))

        val_orig_data_batch = np.concatenate(val_orig_data_batch_list, axis=0)
        val_data_batch = np.concatenate(val_data_batch_list, axis=0)
        val_data_mask = np.concatenate(val_data_mask_list, axis=0)

        input_data = val_data_batch[:, :num_frame_before, :, :, :]
        true_label = val_data_batch[:, num_frame_before:, :, :, :]
        orig_label = val_orig_data_batch[:, num_frame_before:, :, :, :]

        input_data = input_data.astype(np.float32)
        true_label = true_label.astype(np.float32)
        orig_label = orig_label.astype(np.float32)

        input_data[:, :, :, :, :] = input_data[:, :, :, :, :] / 255.0
        true_label[:, :, :, :, :] = true_label[:, :, :, :, :] / 255.0
        orig_label = orig_label / 255.0
        orig_label = orig_label[:, target_frames]
        orig_label = orig_label[:, :, :, :, :num_channel_target]
        true_label = true_label[:, :, :, :, :num_channel_target]

        input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size_val, height, width, -1))
        static_data_tmp = np.tile(static_data, [batch_size_val, 1, 1, 1])
        input_data = np.concatenate([input_data, static_data_tmp], axis=-1)
        true_label = np.moveaxis(true_label, 1, -1).reshape((batch_size_val, height, width, -1))
        orig_label = np.moveaxis(orig_label, 1, -1).reshape((batch_size_val, height, width, -1))

        if use_mask:
            orig_label_mask = np.tile(mask_data, [1, 1, 1, len(target_frames)])
        else:
            orig_label_mask = np.ones_like(orig_label)
        run_out = trainer.evaluate(input_data, orig_label, orig_label_mask, input_time, 55000)
        eval_loss_list.append(run_out['loss'])
    print('time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\t', 'eval_loss:', np.mean(eval_loss_list))
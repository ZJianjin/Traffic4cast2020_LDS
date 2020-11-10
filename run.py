import random
import h5py
from trainer_all import *

train_winter = ['-01-', '-02-', '-03-']
train_summer = ['-05-', '-04-', '-06-']
test_winter = ['-11-', '-12-']
test_summer = ['-07-', '-08-', '-09-', '-10-']

season = None
use_flip = False
use_mask = False
use_time = False
finetune_on_validation = False
# skip_names = ['conv_1_0', 'conv_1_1', 'conv_1_2', 'conv_1_3', 'conv1x1_1_4']
skip_names = []

SEED = 0
num_train_file = 285
num_frame_per_day = 288
num_frame_before = 12
num_frame_sequence = 24
target_frames = [0, 1, 2, 5, 8, 11]
num_sequence_per_day = num_frame_per_day - num_frame_sequence + 1  # 288-15+1=274
height = 495
width = 436
num_channel = 9
num_channel_target = 8
# num_channel_discretized = 9  # 4 * 2
visual_input_channels = 115  # 12 * 8
if use_time:
    visual_input_channels += 12
visual_output_channels = 48  # 6 * 8
vector_input_channels = 1  # start time point
model_name = 'neta'
#
n = 1
s = 255
e = 85
w = 170
tv = 16
#
##############################Set the path##############################################
data_root = './data'
model_root = './jianjzhmodels'
# pre_model_root = os.path.join('./jianjzhmodelstest2', 'MOSCOWneta', 'model-62000.cptk')
pre_model_root = None
# load_model_path = './jianjzhmodelstest2' + '/' + 'ISTANBULneta'
load_model_path = None
log_root = './output'
##############################Set the path##############################################
#

learning_rate = 3e-4

target_city = 'ISTANBUL'  # ['BERLIN', 'MOSCOW', 'ISTANBUL']
# test_start_index_list = np.array([ 18,  57, 114, 174, 222], np.int32)    # 'BERLIN' 
# test_start_index_list = np.array([ 45, 102, 162, 210, 246], np.int32)   # 'Moscow' # 'Istanbul'
input_static_data_path = data_root + '/' + target_city + '/' + target_city + '_static_2019.h5'
input_mask_data_path = data_root + '/maskdata/'
input_train_data_folder_path = data_root + '/' + target_city + '/training'
if finetune_on_validation:
    input_train_data_folder_path = data_root + '/' + target_city + '/validation'
input_val_data_folder_path = data_root + '/' + target_city + '/validation'
input_test_data_folder_path = data_root + '/' + target_city + '/testing'
save_model_path = model_root + '/' + target_city + model_name + str(season) + str(use_flip) + str(use_mask) + str(use_time) + str(finetune_on_validation) + str(learning_rate)
summary_path = log_root + '/' + target_city + model_name + str(season) + str(use_flip) + str(use_mask) + str(use_time) + str(finetune_on_validation) + str(learning_rate)
#
batch_size = 2
batch_size_val = 1
is_training = True
num_epoch_to_train = 100000000
save_per_iteration = 2000
#
num_thread = 16


def return_date(file_name):
    match = re.search(r'\d{4}\d{2}\d{2}', file_name)
    date = datetime.datetime.strptime(match.group(), '%Y%m%d').date()
    return date


def list_filenames(directory, excluded_dates):
    filenames = os.listdir(directory)
    np.random.shuffle(filenames)
    excluded_dates = [datetime.datetime.strptime(x, '%Y-%m-%d').date() for x in excluded_dates]
    filenames = [x for x in filenames if return_date(x) not in excluded_dates]
    return filenames


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
    if season == 'winter':
        tmp = []
        for i in data_filepath_list:
            if any([j in i for j in test_winter]):
                tmp.append(i)
        data_filepath_list = tmp
    elif season == 'summer':
        tmp = []
        for i in data_filepath_list:
            if any([j in i for j in test_summer]):
                tmp.append(i)
        data_filepath_list = tmp
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


def get_mask_data(input_mask_data_path, city):
    map_0 = np.load(input_mask_data_path + city + 'map_0.npy')
    map_1 = np.load(input_mask_data_path + city + 'map_1.npy')
    map_2 = np.load(input_mask_data_path + city + 'map_2.npy')
    map_3 = np.load(input_mask_data_path + city + 'map_3.npy')
    result = np.concatenate([map_0, map_0, map_1, map_1, map_2, map_2, map_3, map_3], axis=-1)
    return result


def get_assignment_map(tvars, init_checkpoint, skip_names):
    model_var_names = []
    for var in tvars:
        model_var_names.append(var.name)
    init_vars = tf.train.list_variables(init_checkpoint)
    assignment_map = dict()
    for ckpt_var in init_vars:
        ckpt_var_name = ckpt_var[0]
        if all([skip not in ckpt_var_name for skip in skip_names]) and ckpt_var_name + ':0' in model_var_names:
            print(ckpt_var_name + ' -> ' + ckpt_var_name + ':0')
            assignment_map[ckpt_var_name] = ckpt_var_name
    return assignment_map


if __name__ == '__main__':

    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)

    trainer = Trainer(height, width, visual_input_channels, visual_output_channels, vector_input_channels,
                      learning_rate, save_model_path, load_model_path, summary_path, is_training, use_mask, model_name)
    tf.reset_default_graph()

    try:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
    except Exception:
        print('save_model_path not made')
        exit(-1)

    val_data = get_data(input_val_data_folder_path)
    print('val_data\t', len(val_data), '\t', val_data[0].shape, '\tsize:', sys.getsizeof(val_data))

    train_data_filepath_list = get_data_filepath_list(input_train_data_folder_path)
    if season == 'winter':
        tmp = []
        for i in train_data_filepath_list:
            if any([j in i for j in train_winter]):
                tmp.append(i)
        train_data_filepath_list = tmp
    elif season == 'summer':
        tmp = []
        for i in train_data_filepath_list:
            if any([j in i for j in train_summer]):
                tmp.append(i)
        train_data_filepath_list = tmp
    print('train_data_filepath_list\t', len(train_data_filepath_list), )
    static_data = get_static_data(input_static_data_path)
    mask_data = get_mask_data(input_mask_data_path, target_city)

    train_set = []
    #
    for i in range(len(train_data_filepath_list)):
        for j in range(num_sequence_per_day):
            train_set.append((i, j))
    num_iteration_per_epoch = int(len(train_set) / batch_size)
    print('num_iteration_per_epoch:', num_iteration_per_epoch)

    val_set = []
    #
    for i in range(len(val_data)):
        for j in range(0, num_sequence_per_day, num_frame_sequence):
            val_set.append((i, j))
    num_val_iteration_per_epoch = int(len(val_set) / batch_size_val)
    print('num_val_iteration_per_epoch:', num_val_iteration_per_epoch)

    train_input_queue = queue.Queue()
    train_output_queue = queue.Queue()


    def load_train_multithread():

        while True:
            if train_input_queue.empty() or train_output_queue.qsize() > 8:
                time.sleep(0.1)
                continue
            i_j_list = train_input_queue.get()

            train_orig_data_batch_list = []
            train_data_batch_list = []
            train_data_mask_list = []
            train_data_time_list = []
            train_stat_batch_list = []
            for train_i_j in i_j_list:
                (i, j) = train_i_j

                file_path = train_data_filepath_list[i]

                fr = h5py.File(file_path, 'r')
                a_group_key = list(fr.keys())[0]
                data = fr[a_group_key]
                assert data.shape[0] == num_frame_per_day

                train_data_time_list.append(float(j) / float(num_frame_per_day))

                train_orig_data_batch_list.append(
                    data[j:j + num_frame_sequence, :, :, :num_channel][np.newaxis, :, :, :, :])

                data_sliced = data[j:j + num_frame_sequence, :, :, :num_channel]
                if use_time:
                    time_id = np.ones_like(data_sliced)[:, :, :, :1]
                    for m in range(num_frame_sequence):
                        time_id[m] = time_id[m] * (j + m) / 288.0 * 255.0
                    data_sliced = np.concatenate([data_sliced, time_id], axis=-1)
                #
                data_mask = (np.max(data_sliced, axis=3) == 0)
                train_data_mask_list.append(data_mask[np.newaxis, :, :, :])

                train_data_batch_list.append(data_sliced[:, :, :, :][np.newaxis, :, :, :, :])

            train_data_time_list = np.asarray(train_data_time_list)
            input_time = np.reshape(train_data_time_list, (batch_size, 1))

            train_orig_data_batch = np.concatenate(train_orig_data_batch_list, axis=0)
            train_data_batch = np.concatenate(train_data_batch_list, axis=0)
            train_data_mask = np.concatenate(train_data_mask_list, axis=0)

            input_data = train_data_batch[:, :num_frame_before, :, :, :]
            true_label = train_data_batch[:, num_frame_before:, :, :, :]
            orig_label = train_orig_data_batch[:, num_frame_before:, :, :, :]

            input_data = input_data.astype(np.float32)
            true_label = true_label.astype(np.float32)
            orig_label = orig_label.astype(np.float32)

            input_data[:, :, :, :, :] = input_data[:, :, :, :, :] / 255.0
            true_label[:, :, :, :, :] = true_label[:, :, :, :, :] / 255.0
            orig_label = orig_label / 255.0
            orig_label = orig_label[:, target_frames]
            orig_label = orig_label[:, :, :, :, :num_channel_target]
            true_label = true_label[:, :, :, :, :num_channel_target]

            input_data = np.moveaxis(input_data, 1, -1).reshape((batch_size, height, width, -1))
            static_data_tmp = np.tile(static_data, [batch_size, 1, 1, 1])
            input_data = np.concatenate([input_data, static_data_tmp], axis=-1)
            true_label = np.moveaxis(true_label, 1, -1).reshape((batch_size, height, width, -1))
            orig_label = np.moveaxis(orig_label, 1, -1).reshape((batch_size, height, width, -1))

            if use_mask:
                orig_label_mask = np.tile(mask_data, [1, 1, 1, len(target_frames)])
            else:
                orig_label_mask = np.ones_like(orig_label)
            # orig_label_mask = np.ones((batch_size, num_frame_sequence - num_frame_before, height, width, num_channel),
            #                           np.float32)
            # orig_label_mask[train_data_mask[:, num_frame_before:, :, :], :] = 0.0
            # orig_label_mask = np.moveaxis(orig_label_mask, 1, -1).reshape((batch_size, height, width, -1))
            #
            # input_data_mask = np.zeros((batch_size, num_frame_before, height, width, num_channel_discretized), np.bool)
            # input_data_mask[train_data_mask[:, :num_frame_before, :, :], :] = True
            # input_data_mask = np.moveaxis(input_data_mask, 1, -1).reshape((batch_size, height, width, -1))
            # input_data[input_data_mask] = -1.0

            train_output_queue.put((input_data, orig_label, orig_label_mask, input_time))


    thread_list = []
    assert num_thread > 0
    for i in range(num_thread):
        t = threading.Thread(target=load_train_multithread)
        t.start()

    global_step = 0
    if pre_model_root is not None:
        print('load' + pre_model_root)
        tvars = trainer.get_trainable_vars()
        assignmap = get_assignment_map(tvars, pre_model_root, skip_names)
        trainer.load_model_assignmap(pre_model_root, assignmap)
    if not is_training:
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
                if use_time:
                    time_id = np.ones_like(data_sliced)[:, :, :, :1]
                    for m in range(num_frame_sequence):
                        time_id[m] = time_id[m] * (j + m) / 288.0 * 255.0
                    data_sliced = np.concatenate([data_sliced, time_id], axis=-1)
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
            # orig_label_mask = np.ones(
            #     (batch_size_val, num_frame_sequence - num_frame_before, height, width, num_channel),
            #     np.float32)
            # orig_label_mask[val_data_mask[:, num_frame_before:, :, :], :] = 0.0
            # orig_label_mask = np.moveaxis(orig_label_mask, 1, -1).reshape((batch_size_val, height, width, -1))
            #
            # input_data_mask = np.zeros((batch_size_val, num_frame_before, height, width, num_channel_discretized),
            #                            np.bool)
            # input_data_mask[val_data_mask[:, :num_frame_before, :, :], :] = True
            # input_data_mask = np.moveaxis(input_data_mask, 1, -1).reshape((batch_size_val, height, width, -1))
            # input_data[input_data_mask] = -1.0
            run_out = trainer.evaluate(input_data, orig_label, orig_label_mask, input_time, 55000)
            eval_loss_list.append(run_out['loss'])
        print('time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'global_step:', global_step, '\t', 'eval_loss:', np.mean(eval_loss_list))
    else:
        for epoch in range(num_epoch_to_train):

            np.random.shuffle(train_set)

            for a in range(num_iteration_per_epoch):

                i_j_list = []
                for train_i_j in train_set[a * batch_size: (a + 1) * batch_size]:
                    i_j_list.append(train_i_j)
                train_input_queue.put(i_j_list)

            for a in range(num_iteration_per_epoch):
                # print(global_step)
                while train_output_queue.empty():
                    time.sleep(0.1)
                (input_data, true_label, true_label_mask, input_time) = train_output_queue.get()
                # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                run_out = trainer.update(input_data, true_label, true_label_mask, input_time, global_step)
                if use_flip:
                    run_out = trainer.update(input_data[:, :, ::-1], true_label[:, :, ::-1], true_label_mask[:, :, ::-1], input_time, global_step)
                global_step += 1
                # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

                if global_step % save_per_iteration == 0:

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
                            val_orig_data_batch_list.append(
                                val_data[i][j:j + num_frame_sequence, :, :, :][np.newaxis, :, :, :, :])
                            data_sliced = val_data[i][j:j + num_frame_sequence, :, :, :]
                            if use_time:
                                time_id = np.ones_like(data_sliced)[:, :, :, :1]
                                for m in range(num_frame_sequence):
                                    time_id[m] = time_id[m] * (j + m) / 288.0 * 255.0
                                data_sliced = np.concatenate([data_sliced, time_id], axis=-1)
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
                        # orig_label_mask = np.ones(
                        #     (batch_size_val, num_frame_sequence - num_frame_before, height, width, num_channel),
                        #     np.float32)
                        # orig_label_mask[val_data_mask[:, num_frame_before:, :, :], :] = 0.0
                        # orig_label_mask = np.moveaxis(orig_label_mask, 1, -1).reshape(
                        #     (batch_size_val, height, width, -1))
                        #
                        # input_data_mask = np.zeros(
                        #     (batch_size_val, num_frame_before, height, width, num_channel_discretized), np.bool)
                        # input_data_mask[val_data_mask[:, :num_frame_before, :, :], :] = True
                        # input_data_mask = np.moveaxis(input_data_mask, 1, -1).reshape(
                        #     (batch_size_val, height, width, -1))
                        # input_data[input_data_mask] = -1.0

                        run_out = trainer.evaluate(input_data, orig_label, orig_label_mask, input_time, global_step)
                        eval_loss_list.append(run_out['loss'])

                    print('time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'global_step:', global_step, '\t', 'epoch:', epoch, '\t', 'eval_loss:',
                          np.mean(eval_loss_list))

                    trainer.save_model(global_step)
                    trainer.write_summary(global_step)

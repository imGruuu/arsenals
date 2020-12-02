import numpy as np
import os
import cv2


def get_all_parent_path(root):#获取根目录下所有文件夹的路径
    folders = os.listdir(root)
    parent_path = []
    for folder in folders:
        if '_' in folder and folder[0] == 'R':
            parent_path.append("{}/{}".format(root, folder))
    return parent_path


def get_batch_paths(paths, seq_size_):#获取一个文件夹下所有文件的路径
    batch_paths = []
    for path in paths:
        files = os.listdir(path)
        i = 0
        file_paths = []
        for file in files:
            if i == seq_size_:
                break
            file_paths.append("{}/{}".format(path, file))
            i += 1
        batch_paths.append(file_paths)
    return batch_paths


def data_generator(images, batch_size_, seq_size_, image_size_):
    if seq_size_ % 2:#取偶数个图片，后面等分成两个 前一个做输入，后一个做标签
        raise ValueError("sequence size not oushu!")

    # minibatch input data and target data
    seq_size = int(seq_size_ / 2)

    input_data = np.zeros((batch_size_, seq_size, 1, image_size_, image_size_), dtype=np.float32)
    target_data = np.zeros((batch_size_, seq_size, 1, image_size_, image_size_), dtype=np.float32)

    for i in range(batch_size_):
        input_data_id = 0
        target_data_id = 0
        for j in range(seq_size_):
            img = cv2.imread(images[i][j], 0)
            img = cv2.resize(img, (image_size_, image_size_))#将img尺寸规范化

            if j % 2:#划分输入和目标数据
                input_data[i, input_data_id, 0, :, :] = img
                input_data_id += 1
            else:
                target_data[i, target_data_id, 0, :, :] = img
                target_data_id += 1

    input_data = input_data.reshape([seq_size, batch_size_, 1, image_size_, image_size_])
    target_data = target_data.reshape([seq_size, batch_size_, 1, image_size_, image_size_])
    return input_data, target_data

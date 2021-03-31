import json
import os
import random

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from datasets.transforms_policy import XGazePolicy

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_xgaze_train_val_loader(config):
    refer_list_file = os.path.join(config.data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)
    print(f'using train and val set, val ratio={config.split_ratio}')

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    if hasattr(config, "use_aa") and config.use_aa:
        print('use augment policy')
        trans = transforms.Compose([
            XGazePolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        trans = DEFAULT_TRANSFORM

    sub_folder_use = 'train'
    train_set = GazeDataset(dataset_path=config.data_dir, keys_to_use=datastore[sub_folder_use],
                            sub_folder=sub_folder_use, transform=trans, is_shuffle=True,
                            is_load_label=True, is_load_pose=config.is_load_pose)

    indices = list(range(len(train_set)))
    split = int(np.floor(len(train_set) * config.split_ratio))
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, num_workers=config.num_workers,
                              sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]))

    val_loader = DataLoader(train_set, batch_size=config.val_batch_size, num_workers=config.num_workers,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:len(train_set)]))

    return train_loader, val_loader


def get_xgaze_train_loader(config):
    # load dataset
    refer_list_file = os.path.join(config.data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)
    print('using train set')

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'train'

    if hasattr(config, "use_aa") and config.use_aa:
        trans = transforms.Compose([
            XGazePolicy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        trans = DEFAULT_TRANSFORM

    train_set = GazeDataset(dataset_path=config.data_dir, keys_to_use=datastore[sub_folder_use],
                            sub_folder=sub_folder_use, transform=trans, is_shuffle=True,
                            is_load_label=True, is_load_pose=config.is_load_pose)
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, num_workers=config.num_workers)
    return train_loader


def get_xgaze_test_loader(config):
    # load dataset
    refer_list_file = os.path.join(config.data_dir, 'train_test_split.json')
    print('load the train file list from: ', refer_list_file)
    print('using test set')

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    # there are three subsets for ETH-XGaze dataset: train, test and test_person_specific
    # train set: the training set includes 80 participants data
    # test set: the test set for cross-dataset and within-dataset evaluations
    # test_person_specific: evaluation subset for the person specific setting
    sub_folder_use = 'test'
    trans = DEFAULT_TRANSFORM

    test_set = GazeDataset(dataset_path=config.data_dir, keys_to_use=datastore[sub_folder_use],
                           sub_folder=sub_folder_use, transform=trans, is_shuffle=False,
                           is_load_label=False, is_load_pose=config.is_load_pose)
    test_loader = DataLoader(test_set, batch_size=config.test_batch_size, num_workers=config.num_workers)
    return test_loader


class GazeDataset(Dataset):
    def __init__(self, dataset_path, keys_to_use, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True, is_load_pose=True):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label
        self.is_load_pose = is_load_pose

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            file_path = os.path.join(self.path, self.sub_folder, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print('load the file: ', index_file)
            self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, idx):
        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.sub_folder, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf['face_patch'][idx, :]
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        # Get gaze and pose
        return_dict = {
            'image': image,
            # 'gaze': None,
            # 'pose': None
        }

        if self.is_load_pose:
            if self.is_load_label:
                gaze_label = self.hdf['face_gaze'][idx, :]
                pose_label = self.hdf['face_head_pose'][idx, :]
                return_dict['gaze'] = gaze_label.astype('float32')
                return_dict['pose'] = pose_label.astype('float32')
            else:
                pose_label = self.hdf['face_head_pose'][idx, :]
                return_dict['pose'] = pose_label.astype('float32')
        else:
            if self.is_load_label:
                gaze_label = self.hdf['face_gaze'][idx, :]
                return_dict['gaze'] = gaze_label.astype('float32')
            else:
                pass

        return return_dict

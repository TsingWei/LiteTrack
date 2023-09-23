import torch
import os
import os.path
import numpy as np
import csv
import pandas
from collections import OrderedDict

from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings
from .base_video_dataset import BaseVideoDataset


import glob
import json
import cv2

def list_sequences(root):

    sequence_list = []

    seq_dir = os.path.join(root)
    for filename in os.listdir(seq_dir):
        sequence_list.append(filename)

    return sequence_list


class UAV(BaseVideoDataset):

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):

        root = env_settings().uav_dir if root is None else root
        super().__init__('antiUAV', root, image_loader)

        self.sequence_list = self._get_sequence_list()
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        # self.sequence_list = list_sequences(self.root)
        self.ann_path = 'data/uav/train'

    def get_name(self):
        return 'uav'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_anno(self, anno_path):
        # gt = pandas.read_csv(anno_path, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        res_file = os.path.join(anno_path, 'IR_label.json')
        with open(res_file, 'r') as f:
            label_res = json.load(f)
        gt = label_res['gt_rect']
        new_gt = [[0,0,0,0] if len(f)==0 else f for f in gt]
        try:
            a = torch.tensor(new_gt).float()
        except:
            print(anno_path)
        return torch.tensor(new_gt).float()

    def _read_target_visible(self, seq_path):
        res_file = os.path.join(seq_path, 'IR_label.json')
        with open(res_file, 'r') as f:
            label_res = json.load(f)
        exist = label_res['exist']
        target_visible = torch.ByteTensor([int(v) for v in exist])

        return target_visible

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        anno = self._read_anno(seq_path)
        valid = (anno[:,2]>0) & (anno[:,3]>0)
        visible = self._read_target_visible(seq_path)
        visible = visible & valid.byte()

        return {'bbox': anno, 'valid': valid, 'visible': visible}

    def _get_frame_path(self,seq_path,frame_id):
        # result = []
        # for filename in os.listdir(seq_path):
        #     result.append(int(filename[0:5]))

        return os.path.join(seq_path, '{:06}.jpg'.format(frame_id+1))

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))
        # capture = cv2.VideoCapture(seq_path)
        # capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        # ret, frame = capture.read()
        # import random
        # c = random.randint(0, 2)
        # kernel = np.ones((5, 5), np.uint8)
        # if random.random() < 0.5:
        #     return cv2.dilate(frame[:, :, c], kernel)
        # return frame[:, :, c]
        # return frame

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
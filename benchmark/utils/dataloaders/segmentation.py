import os

import numpy as np
import cv2 as cv

from .base_dataloader import _BaseImageLoader
from ..factory import DATALOADERS

@DATALOADERS.register
class SegmentationImageLoader(_BaseImageLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._to_rgb = kwargs.pop('toRGB', False)
        self._point_label= self._load_point_and_label()

    def _load_point_and_label(self):
        points_labels = dict.fromkeys(self._files, None)
        for filename in self._files:
            if os.path.exists(os.path.join(self._path, '{}.txt'.format(filename[:-4]))):
                points_labels[filename] = np.loadtxt(os.path.join(self._path, '{}.txt'.format(filename[:-4])), ndmin=2)
            else:
                points_labels[filename] = None
        # for filename in self._files:
        #     label_file = os.path.join(self._path, '{}.txt'.format(filename[:-4]))
        #     if os.path.exists(label_file):
        #         # 假设标签文件的每一行格式为：x y label
        #         # 其中 x, y 是点的坐标，label 是标签（0 或 1）
        #         with open(label_file, 'r') as file:
        #             lines = file.readlines()
        #             current_point_label = []
        #             for line in lines:
        #                 parts = line.strip().split()
        #                 if len(parts) == 3:
        #                     x, y, label = map(int, parts)
        #                     current_point_label.append((x, y, label))
        #         points_labels[filename] = current_point_label
        #     else:
        #         points_labels[filename] = None
        return points_labels


    def _toRGB(self, image):
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)

    def __iter__(self):
        for filename in self._files:
            image = cv.imread(os.path.join(self._path, filename))
            
            if self._to_rgb:
                image = self._toRGB(image)
                
            if [0, 0] in self._sizes:
                point_and_label = self._point_label.get(filename)
                if point_and_label is not None:
                    yield filename, image, point_and_label
                else:
                    yield filename, image, None
            else:
                for size in self._sizes:
                    image_r = cv.resize(image, size)
                    point_and_label = self._point_label.get(filename)
                    if point_and_label is not None:
                        yield filename, image_r, point_and_label
                    else:
                        yield filename, image_r, None
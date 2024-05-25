import cv2 as cv

from .base_metric import BaseMetric
from ..factory import METRICS

@METRICS.register
class Segmentation(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, model, *args, **kwargs):
        img, point_and_label = args
        size = [img.shape[1], img.shape[0]]
        self._timer.reset()
        if point_and_label is not None:
            for idx, pl in enumerate(point_and_label):
                point = [[pl[0], pl[1]]]
                label = [[pl[2]]]
                for _ in range(self._warmup):
                    model.infer(img, point, label)
                for _ in range(self._repeat):
                    self._timer.start()
                    model.infer(img, point, label)
                    self._timer.stop()
        else:
            point = [[int(size[0]/2), int(size[1]/2)]]
            label = [[1]]
            for _ in range(self._warmup):
                model.infer(img, point, label)
            for _ in range(self._repeat):
                self._timer.start()
                model.infer(img, point, label)
                self._timer.stop()

        return self._timer.getRecords()

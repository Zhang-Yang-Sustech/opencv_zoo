from .base import BaseImageLoader, BaseVideoLoader
from .classification import ClassificationImageLoader
from .recognition import RecognitionImageLoader
from .tracking import TrackingVideoLoader
from .segmentation import SegmentationImageLoader

__all__ = ['BaseImageLoader', 'BaseVideoLoader', 'ClassificationImageLoader', 'RecognitionImageLoader', 'SegmentationImageLoader', 'TrackingVideoLoader']
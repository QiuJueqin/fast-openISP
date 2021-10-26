# File: scl.py
# Description: Scaler
# Created: 2021/10/26 22:36
# Author: Qiu Jueqin (qiujueqin@gmail.com)


from functools import partial
import cv2

from .basic_module import BasicModule


class SCL(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.resize = partial(
            cv2.resize, dsize=(self.params.width, self.params.height), interpolation=cv2.INTER_LINEAR
        )

    def execute(self, data):
        if 'y_image' and 'cbcr_image' in data:
            data['y_image'] = self.resize(data['y_image'])
            data['cbcr_image'] = self.resize(data['cbcr_image'])
        elif 'rgb_image' in data:
            data['rgb_image'] = self.resize(data['rgb_image'])
        else:
            raise NotImplementedError('can not resize Bayer array')

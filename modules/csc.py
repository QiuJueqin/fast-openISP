# File: csc.py
# Description: Color Space Conversion
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules


@register_dependent_modules('gac')
class CSC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.matrix = np.array([[66, 129, 25],
                                [-38, -74, 112],
                                [112, -94, -18]], dtype=np.int32).T  # x256
        self.bias = np.array([16, 128, 128], dtype=np.int32).reshape(1, 1, 3)

    def execute(self, data):
        rgb_image = data['rgb_image'].astype(np.int32)

        ycrcb_image = (np.right_shift(rgb_image @ self.matrix, 8) + self.bias).astype(np.uint8)

        data['y_image'] = ycrcb_image[..., 0]
        data['cbcr_image'] = ycrcb_image[..., 1:]

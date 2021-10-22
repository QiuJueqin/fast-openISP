# File: csc.py
# Description: Color Space Conversion
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule


class CSC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        csc = np.array(self.params.csc, dtype=np.int32).T  # x1024, (4, 3) right-matrix
        self.matrix = csc[:3, :]  # (3, 3) right-matrix
        self.bias = csc[3, :].reshape(1, 1, 3)  # (1, 1, 3)

    def execute(self, data):
        rgb_image = data['rgb_image'].astype(np.int32)

        ycrcb_image = np.right_shift(rgb_image @ self.matrix + self.bias, 10).astype(np.uint8)

        data['y_image'] = ycrcb_image[..., 0]
        data['cbcr_image'] = ycrcb_image[..., 1:]

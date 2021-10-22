# File: bnf.py
# Description: Bilateral Noise Filtering
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from utils.image_helpers import pad, shift_array


@register_dependent_modules('csc')
class BNF(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        dw = np.array(self.params.dw, dtype=np.int32)
        self.window_height, self.window_width = dw.shape
        self.dw = dw.flatten()
        self.rw = np.array(self.params.rw, dtype=np.int32)

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        padded_y_image = pad(y_image, pads=(self.window_height // 2, self.window_width // 2))
        shifted_arrays = shift_array(padded_y_image, window_size=(self.window_height, self.window_width))

        bnf_y_image = weights = 0
        for i, shifted_y_image in enumerate(shifted_arrays):
            diff = np.abs(shifted_y_image - y_image)
            cur_weights = self.dw[i] * (
                    (diff >= self.params.diff_thres[0]) * self.rw[0] +
                    (diff < self.params.diff_thres[0]) * (diff >= self.params.diff_thres[1]) * self.rw[1] +
                    (diff < self.params.diff_thres[1]) * (diff >= self.params.diff_thres[2]) * self.rw[2] +
                    (diff < self.params.diff_thres[2]) * self.rw[3]
            )
            bnf_y_image += cur_weights * shifted_y_image
            weights += cur_weights

        bnf_y_image = bnf_y_image / weights
        data['y_image'] = bnf_y_image.astype(np.uint8)

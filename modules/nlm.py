# File: nlm.py
# Description: Non-Local Means Denoising
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from .helpers import pad, shift_array, mean_filter


@register_dependent_modules('csc')
class NLM(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.distance_weights_lut = self.get_distance_weights_lut(h=self.params.h)

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        padded_y_image = pad(y_image, pads=self.params.search_window_size // 2)
        shifted_arrays = shift_array(padded_y_image, window_size=self.params.search_window_size)

        nlm_y_image = np.zeros_like(y_image)
        weights = np.zeros_like(y_image)

        for i, shifted_y_image in enumerate(shifted_arrays):
            distance = mean_filter((y_image - shifted_y_image) ** 2, filter_size=self.params.patch_size)
            weight = self.distance_weights_lut[distance]
            nlm_y_image += shifted_y_image * weight
            weights += weight

        nlm_y_image = nlm_y_image / weights
        data['y_image'] = nlm_y_image.astype(np.uint8)

    @staticmethod
    def get_distance_weights_lut(h):
        distance = np.arange(255 ** 2)
        lut = 1024 * np.exp(-distance / h ** 2)
        return lut.astype(np.int32)  # x1024

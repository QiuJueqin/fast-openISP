# File: nlm.py
# Description: Non-Local Means Denoising
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from utils.image_helpers import pad, shift_array, mean_filter


@register_dependent_modules('csc')
class NLM(BasicModule):
    def execute(self, data):
        y_image = data['y_image'].astype(np.float32)

        padded_y_image = pad(y_image, pads=self.params.search_window_size // 2)
        shifted_arrays = shift_array(padded_y_image, window_size=self.params.search_window_size)

        nlm_y_image = np.zeros_like(y_image)
        weights = np.zeros_like(y_image)

        for i, shifted_y_image in enumerate(shifted_arrays):
            distance = mean_filter((y_image - shifted_y_image) ** 2, filter_size=self.params.patch_size)
            weight = np.exp(-distance / self.params.h ** 2)
            nlm_y_image += shifted_y_image * weight
            weights += weight

        nlm_y_image /= weights

        data['y_image'] = nlm_y_image.astype(np.uint8)

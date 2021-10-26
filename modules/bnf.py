# File: bnf.py
# Description: Bilateral Noise Filtering
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from .helpers import bilateral_filter, gen_gaussian_kernel


@register_dependent_modules('csc')
class BNF(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.intensity_weights_lut = self.get_intensity_weights_lut(self.params.intensity_sigma)  # x1024
        spatial_weights = gen_gaussian_kernel(kernel_size=5, sigma=self.params.spatial_sigma)
        self.spatial_weights = (1024 * spatial_weights / spatial_weights.max()).astype(np.int32)  # x1024

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        bf_y_image = bilateral_filter(y_image, self.spatial_weights, self.intensity_weights_lut, right_shift=10)
        data['y_image'] = bf_y_image.astype(np.uint8)

    @staticmethod
    def get_intensity_weights_lut(intensity_sigma):
        intensity_diff = np.arange(255 ** 2)
        exp_lut = 1024 * np.exp(-intensity_diff / (2.0 * (255 * intensity_sigma) ** 2))
        return exp_lut.astype(np.int32)  # x1024

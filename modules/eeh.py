# File: eeh.py
# Description: Edge Enhancement
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from modules.basic_module import BasicModule, register_dependent_modules
from utils.image_helpers import bilateral_filter, gen_gaussian_kernel


@register_dependent_modules('csc')
class EEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.intensity_weights_lut = self.get_intensity_weights_lut(intensity_sigma=1.0)  # x1024
        spatial_weights = gen_gaussian_kernel(kernel_size=5, sigma=1.0)
        self.spatial_weights = (1024 * spatial_weights / spatial_weights.max()).astype(np.int32)  # x1024

        flat_slope = self.params.middle_threshold / (self.params.middle_threshold - self.params.flat_threshold + 1E-6)
        edge_slope = self.params.edge_gain / 256

        self.flat_slope = np.array(256 * flat_slope, dtype=np.int32)  # x256
        self.edge_slope = np.array(256 * edge_slope, dtype=np.int32)  # x256
        self.flat_intercept = np.array(-flat_slope * self.params.flat_threshold, dtype=np.int32)
        self.edge_intercept = np.array((1 - edge_slope) * self.params.edge_threshold, dtype=np.int32)

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        bf_y_image = bilateral_filter(y_image, self.spatial_weights, self.intensity_weights_lut, right_shift=10)

        delta = y_image - bf_y_image
        sign_map = np.sign(delta)
        abs_delta = np.abs(delta)

        flat_delta = np.right_shift(self.flat_slope * abs_delta, 8) + self.flat_intercept
        edge_delta = np.right_shift(self.edge_slope * abs_delta, 8) + self.edge_intercept
        enhanced_delta = sign_map * (
                (abs_delta > self.params.flat_threshold) * (abs_delta <= self.params.middle_threshold) * flat_delta +
                (abs_delta > self.params.middle_threshold) * (abs_delta <= self.params.edge_threshold) * abs_delta +
                (abs_delta > self.params.edge_threshold) * edge_delta
        )
        enhanced_delta = np.clip(enhanced_delta, -self.params.delta_threshold, self.params.delta_threshold)

        eeh_y_image = np.clip(bf_y_image + enhanced_delta, 0, self.cfg.saturation_values.sdr)

        data['y_image'] = eeh_y_image.astype(np.uint8)
        data['edge_map'] = delta

    @staticmethod
    def get_intensity_weights_lut(intensity_sigma):
        intensity_diff = np.arange(255 ** 2)
        exp_lut = 1024 * np.exp(-intensity_diff / (2.0 * (255 * intensity_sigma) ** 2))
        return exp_lut.astype(np.int32)  # x1024


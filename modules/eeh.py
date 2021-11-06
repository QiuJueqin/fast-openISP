# File: eeh.py
# Description: Edge Enhancement
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from .helpers import gaussian_filter, gen_gaussian_kernel


@register_dependent_modules('csc')
class EEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        kernel = gen_gaussian_kernel(kernel_size=7, sigma=5.0)
        self.kernel = (1024 * kernel / kernel.max()).astype(np.int32)  # x1024

        flat_slope = self.params.middle_threshold / (self.params.middle_threshold - self.params.flat_threshold + 1E-6)
        edge_slope = self.params.edge_gain / 256

        self.flat_slope = np.array(256 * flat_slope, dtype=np.int32)  # x256
        self.edge_slope = np.array(256 * edge_slope, dtype=np.int32)  # x256
        self.flat_intercept = -np.array(256 * flat_slope * self.params.flat_threshold, dtype=np.int32)  # x256
        self.edge_intercept = np.array(256 * (1 - edge_slope) * self.params.edge_threshold, dtype=np.int32)  # x256

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        gf_y_image = gaussian_filter(y_image, self.kernel)

        delta = y_image - gf_y_image
        sign_map = np.sign(delta)
        abs_delta = np.abs(delta)

        flat_delta = np.right_shift(self.flat_slope * abs_delta + self.flat_intercept, 8)
        edge_delta = np.right_shift(self.edge_slope * abs_delta + self.edge_intercept, 8)
        enhanced_delta = sign_map * (
                (abs_delta > self.params.flat_threshold) * (abs_delta <= self.params.middle_threshold) * flat_delta +
                (abs_delta > self.params.middle_threshold) * (abs_delta <= self.params.edge_threshold) * abs_delta +
                (abs_delta > self.params.edge_threshold) * edge_delta
        )
        enhanced_delta = np.clip(enhanced_delta, -self.params.delta_threshold, self.params.delta_threshold)

        eeh_y_image = np.clip(gf_y_image + enhanced_delta, 0, self.cfg.saturation_values.sdr)

        data['y_image'] = eeh_y_image.astype(np.uint8)
        data['edge_map'] = delta

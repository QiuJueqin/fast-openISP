# File: eeh.py
# Description: Edge Enhancement
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from utils.image_helpers import pad, shift_array


@register_dependent_modules('csc')
class EEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        gaussian = np.array([[1, 4, 7, 4, 1],
                             [4, 16, 26, 16, 4],
                             [7, 26, 41, 26, 7],
                             [4, 16, 26, 16, 4],
                             [1, 4, 7, 4, 1]])
        self.kernel_height, self.kernel_width = gaussian.shape
        self.gaussian = [int(1024 * x / gaussian.sum()) for x in gaussian.flatten()]  # x1024

        flat_slope = self.params.middle_thres / (self.params.middle_thres - self.params.flat_thres)
        edge_slope = self.params.edge_gain / 256

        self.x1 = 1024 * self.params.flat_thres  # flat threshold, x1024
        self.x2 = 1024 * self.params.middle_thres  # middle threshold, x1024
        self.x3 = 1024 * self.params.edge_thres  # edge threshold, x1024
        self.a1 = np.array(256 * flat_slope, dtype=np.int32)  # flat slope, x256
        self.a3 = np.array(256 * edge_slope, dtype=np.int32)  # edge slope, x256
        self.b1 = np.array(-flat_slope * self.x1, dtype=np.int32)  # flat_intercept, x1024
        self.b3 = np.array((1 - edge_slope) * self.x3, dtype=np.int32)  # edge_intercept, x1024
        self.delta_lower = -1024 * self.params.delta_thres  # enhanced delta upper limit, x1024
        self.delta_upper = 1024 * self.params.delta_thres  # enhanced delta lower limit, x1024

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        padded_y_image = pad(y_image, pads=(self.kernel_height // 2, self.kernel_width // 2))
        shifted_arrays = shift_array(padded_y_image, window_size=(self.kernel_height, self.kernel_width))

        gauss = 0
        for i, shifted_array in enumerate(shifted_arrays):
            gauss += self.gaussian[i] * shifted_array  # faster gaussian convolution

        delta = np.left_shift(y_image, 10) - gauss  # x1024

        sign_map = np.sign(delta)
        abs_delta = np.abs(delta)

        flat_delta = np.right_shift(self.a1 * abs_delta, 8) + self.b1  # x1024
        edge_delta = np.right_shift(self.a3 * abs_delta, 8) + self.b3  # x1024
        enhanced_delta = sign_map * (
                (abs_delta > self.x1) * (abs_delta <= self.x2) * flat_delta +
                (abs_delta > self.x2) * (abs_delta <= self.x3) * abs_delta +
                (abs_delta > self.x3) * edge_delta
        )  # x1024
        enhanced_delta = np.clip(enhanced_delta, self.delta_lower, self.delta_upper)  # x1024

        eeh_y_image = np.right_shift(gauss + enhanced_delta, 10)
        eeh_y_image = np.clip(eeh_y_image, 0, self.cfg.saturation_values.sdr)

        data['y_image'] = eeh_y_image.astype(np.uint8)
        data['edge_map'] = np.right_shift(delta, 10)

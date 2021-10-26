# File: bcc.py
# Description: Brightness Contrast Control
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules


@register_dependent_modules('csc')
class BCC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.brightness_offset = np.array(self.params.brightness_offset, dtype=np.int32)
        self.contrast_gain = np.array(self.params.contrast_gain, dtype=np.int32)  # x256

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)

        bcc_y_image = np.clip(y_image + self.brightness_offset, 0, self.cfg.saturation_values.sdr)

        y_median = np.median(bcc_y_image).astype(np.int32)
        bcc_y_image = np.right_shift((bcc_y_image - y_median) * self.contrast_gain, 8) + y_median
        bcc_y_image = np.clip(bcc_y_image, 0, self.cfg.saturation_values.sdr)

        data['y_image'] = bcc_y_image.astype(np.uint8)

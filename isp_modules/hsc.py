# File: hsc.py
# Description: Hue Saturation Control
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule, register_dependent_modules


@register_dependent_modules('csc')
class HSC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        hue_offset = np.pi * self.params.hue_offset / 180
        self.sin_hue = (256 * np.sin(hue_offset)).astype(np.int32)  # x256
        self.cos_hue = (256 * np.cos(hue_offset)).astype(np.int32)  # x256
        self.saturation_gain = np.array(self.params.saturation_gain, dtype=np.int32)  # x256

    def execute(self, data):
        cbcr_image = data['cbcr_image'].astype(np.int32)

        cb_image, cr_image = np.split(cbcr_image, 2, axis=2)

        hsc_cb_image = np.right_shift(self.cos_hue * (cb_image - 128) - self.sin_hue * (cr_image - 128), 8)  # x256
        hsc_cb_image = np.right_shift(self.saturation_gain * hsc_cb_image, 8) + 128

        hsc_cr_image = np.right_shift(self.sin_hue * (cb_image - 128) + self.cos_hue * (cr_image - 128), 8)  # x256
        hsc_cr_image = np.right_shift(self.saturation_gain * hsc_cr_image, 8) + 128

        hsc_cbcr_image = np.dstack([hsc_cb_image, hsc_cr_image])
        hsc_cbcr_image = np.clip(hsc_cbcr_image, 0, self.cfg.saturation_values.sdr)

        data['cbcr_image'] = hsc_cbcr_image.astype(np.uint8)

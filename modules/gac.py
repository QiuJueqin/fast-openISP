# File: gac.py
# Description: Gamma Correction
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule


class GAC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.gain = np.array(self.params.gain, dtype=np.uint32)  # x256
        x = np.arange(self.cfg.saturation_values.hdr + 1)
        lut = ((x / self.cfg.saturation_values.hdr) ** self.params.gamma) * self.cfg.saturation_values.sdr
        self.lut = lut.astype(np.uint8)

    def execute(self, data):
        rgb_image = data['rgb_image'].astype(np.uint32)

        gac_rgb_image = np.right_shift(self.gain * rgb_image, 8)
        gac_rgb_image = np.clip(gac_rgb_image, 0, self.cfg.saturation_values.hdr)
        gac_rgb_image = self.lut[gac_rgb_image]

        data['rgb_image'] = gac_rgb_image

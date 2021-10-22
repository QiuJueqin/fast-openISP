# File: ccm.py
# Description: Color Correction Matrix
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule


class CCM(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        ccm = np.array(self.params.ccm, dtype=np.int32).T  # x1024, (4, 3) right-matrix
        self.matrix = ccm[:3, :]  # (3, 3) right-matrix
        self.bias = ccm[3, :].reshape(1, 1, 3)  # (1, 1, 3)

    def execute(self, data):
        rgb_image = data['rgb_image'].astype(np.int32)

        ccm_rgb_image = np.right_shift(rgb_image @ self.matrix + self.bias, 10)
        ccm_rgb_image = np.clip(ccm_rgb_image, 0, self.cfg.saturation_values.hdr)

        data['rgb_image'] = ccm_rgb_image.astype(np.uint16)

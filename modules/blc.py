# File: blc.py
# Description: Black Level Compensation
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule
from .helpers import split_bayer, reconstruct_bayer


class BLC(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.alpha = np.array(self.params.alpha, dtype=np.int32)  # x1024
        self.beta = np.array(self.params.beta, dtype=np.int32)  # x1024

    def execute(self, data):
        bayer = data['bayer'].astype(np.int32)

        r, gr, gb, b = split_bayer(bayer, self.cfg.hardware.bayer_pattern)
        r = np.clip(r - self.params.bl_r, 0, None)
        b = np.clip(b - self.params.bl_b, 0, None)
        gr -= (self.params.bl_gr - np.right_shift(r * self.alpha, 10))
        gb -= (self.params.bl_gb - np.right_shift(b * self.beta, 10))
        blc_bayer = reconstruct_bayer(
            (r, gr, gb, b), self.cfg.hardware.bayer_pattern
        )

        data['bayer'] = np.clip(blc_bayer, 0, None).astype(np.uint16)

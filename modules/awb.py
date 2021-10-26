# File: awb.py
# Description: Auto White Balance (actually not Auto)
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule
from .helpers import split_bayer, reconstruct_bayer


class AWB(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.r_gain = np.array(self.params.r_gain, dtype=np.uint32)  # x1024
        self.gr_gain = np.array(self.params.gr_gain, dtype=np.uint32)  # x1024
        self.gb_gain = np.array(self.params.gb_gain, dtype=np.uint32)  # x1024
        self.b_gain = np.array(self.params.b_gain, dtype=np.uint32)  # x1024

    def execute(self, data):
        bayer = data['bayer'].astype(np.uint32)

        sub_arrays = split_bayer(bayer, self.cfg.hardware.bayer_pattern)
        gains = (self.r_gain, self.gr_gain, self.gb_gain, self.b_gain)

        wb_sub_arrays = []
        for sub_array, gain in zip(sub_arrays, gains):
            wb_sub_arrays.append(
                np.right_shift(gain * sub_array, 10)
            )
        wb_bayer = reconstruct_bayer(wb_sub_arrays, self.cfg.hardware.bayer_pattern)
        wb_bayer = np.clip(wb_bayer, 0, self.cfg.saturation_values.hdr)

        data['bayer'] = wb_bayer.astype(np.uint16)

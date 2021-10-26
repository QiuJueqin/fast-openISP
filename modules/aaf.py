# File: aaf.py
# Description: Anti-aliasing Filter
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule
from .helpers import pad, split_bayer, reconstruct_bayer, shift_array


class AAF(BasicModule):
    def execute(self, data):
        bayer = data['bayer'].astype(np.uint32)

        padded_bayer = pad(bayer, pads=2)
        padded_sub_arrays = split_bayer(padded_bayer, self.cfg.hardware.bayer_pattern)

        aaf_sub_arrays = []
        for padded_array in padded_sub_arrays:
            shifted_arrays = shift_array(padded_array, window_size=3)
            aaf_sub_array = 0
            for i, shifted_array in enumerate(shifted_arrays):
                mul = 8 if i == 4 else 1
                aaf_sub_array += mul * shifted_array

            aaf_sub_arrays.append(np.right_shift(aaf_sub_array, 4))

        aaf_bayer = reconstruct_bayer(aaf_sub_arrays, self.cfg.hardware.bayer_pattern)

        data['bayer'] = aaf_bayer.astype(np.uint16)

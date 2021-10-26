# File: dpc.py
# Description: Dead Pixel Correction
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from .basic_module import BasicModule
from .helpers import pad, split_bayer, reconstruct_bayer, shift_array


class DPC(BasicModule):
    def execute(self, data):
        bayer = data['bayer'].astype(np.int32)

        padded_bayer = pad(bayer, pads=2)
        padded_sub_arrays = split_bayer(padded_bayer, self.cfg.hardware.bayer_pattern)

        dpc_sub_arrays = []
        for padded_array in padded_sub_arrays:
            shifted_arrays = tuple(shift_array(padded_array, window_size=3))   # generator --> tuple

            mask = (np.abs(shifted_arrays[4] - shifted_arrays[1]) > self.params.diff_threshold) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[7]) > self.params.diff_threshold) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[3]) > self.params.diff_threshold) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[5]) > self.params.diff_threshold) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[0]) > self.params.diff_threshold) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[2]) > self.params.diff_threshold) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[6]) > self.params.diff_threshold) * \
                   (np.abs(shifted_arrays[4] - shifted_arrays[8]) > self.params.diff_threshold)

            dv = np.abs(2 * shifted_arrays[4] - shifted_arrays[1] - shifted_arrays[7])
            dh = np.abs(2 * shifted_arrays[4] - shifted_arrays[3] - shifted_arrays[5])
            ddl = np.abs(2 * shifted_arrays[4] - shifted_arrays[0] - shifted_arrays[8])
            ddr = np.abs(2 * shifted_arrays[4] - shifted_arrays[6] - shifted_arrays[2])
            indices = np.argmin(np.dstack([dv, dh, ddl, ddr]), axis=2)[..., None]

            neighbor_stack = np.right_shift(np.dstack([
                shifted_arrays[1] + shifted_arrays[7],
                shifted_arrays[3] + shifted_arrays[5],
                shifted_arrays[0] + shifted_arrays[8],
                shifted_arrays[6] + shifted_arrays[2]
            ]), 1)
            dpc_array = np.take_along_axis(neighbor_stack, indices, axis=2).squeeze(2)
            dpc_sub_arrays.append(
                mask * dpc_array + ~mask * shifted_arrays[4]
            )

        dpc_bayer = reconstruct_bayer(dpc_sub_arrays, self.cfg.hardware.bayer_pattern)

        data['bayer'] = dpc_bayer.astype(np.uint16)

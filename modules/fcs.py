# File: fcs.py
# Description: False Color Suppression
# Created: 2021/10/22 20:50
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import numpy as np

from modules.basic_module import BasicModule, register_dependent_modules


@register_dependent_modules(('csc', 'eeh'))
class FCS(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        slope = (self.params.edge_gain - self.params.flat_gain) / (self.params.delta_max - self.params.delta_min)
        intercept = -slope * self.params.delta_max
        self.slope = np.array(slope, dtype=np.int32)  # x256
        self.intercept = np.array(intercept, dtype=np.int32)

    def execute(self, data):
        cbcr_image = data['cbcr_image'].astype(np.int32)
        edge_map = data['edge_map']

        gain_map = self.slope * np.abs(edge_map) + self.intercept
        gain_map = np.clip(gain_map, self.params.edge_gain, self.params.flat_gain)
        fcs_cbcr_image = np.right_shift(gain_map[..., None] * (cbcr_image - 128), 8) + 128
        fcs_cbcr_image = np.clip(fcs_cbcr_image, 0, self.cfg.saturation_values.sdr)

        data['cbcr_image'] = fcs_cbcr_image.astype(np.uint8)

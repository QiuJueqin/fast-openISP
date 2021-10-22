# File: isp_pipeline.py
# Description: Main pipeline of fast-openISP
# Created: 2021/10/22 20:34
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import time
import copy
import importlib
from collections import OrderedDict

import cv2
import numpy as np

from utils.yacs import Config


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg

        saturation_values = self.get_saturation_values()
        with self.cfg.unfreeze():
            self.cfg.saturation_values = saturation_values

        enabled_modules = tuple(m for m, en in self.cfg.module_enable_status.items() if en)

        self.modules = OrderedDict()
        for module_name in enabled_modules:
            package = importlib.import_module('isp_modules.{}'.format(module_name))
            module_cls = getattr(package, module_name.upper())
            module = module_cls(self.cfg)

            if hasattr(module, 'dependent_modules'):
                for m in module.register_dependent_modules:
                    if m not in enabled_modules:
                        raise RuntimeError('{} is available only if {} is activated'.format(module_name, m))

            self.modules[module_name] = module

    def get_saturation_values(self):
        """
        Get saturation pixel values in different stages in the pipeline.
        Raw stage: dataflow before the BLC modules (not included)
        HDR stage: dataflow after the BLC modules (included) and before the bit-depth compression
            module, i.e., Gamma in openISP (not included)
        SDR stage: dataflow after the Gamma modules (included)
        """

        raw_max_value = 2 ** self.cfg.hardware.raw_bit_depth - 1
        sdr_max_value = 255

        # Saturation values should be carefully calculated if BLC module is activated
        if 'blc' in self.cfg.module_enable_status:
            hdr_r_max_value = raw_max_value - self.cfg.blc.bl_r
            hdr_b_max_value = raw_max_value - self.cfg.blc.bl_b
            hdr_gr_max_value = int(raw_max_value - self.cfg.blc.bl_gr + hdr_r_max_value * self.cfg.blc.alpha / 1024)
            hdr_gb_max_value = int(raw_max_value - self.cfg.blc.bl_gb + hdr_b_max_value * self.cfg.blc.beta / 1024)
            hdr_max_value = max([hdr_r_max_value, hdr_b_max_value, hdr_gr_max_value, hdr_gb_max_value])
        else:
            hdr_max_value = raw_max_value

        return Config({'raw': raw_max_value,
                       'hdr': hdr_max_value,
                       'sdr': sdr_max_value})

    def execute(self, bayer, save_intermediates=False, verbose=True):
        """
        ISP pipeline execution
        :param bayer: input Bayer array, np.ndarray(H, W)
        :param save_intermediates: whether to save intermediate results from all ISP modules
        :param verbose: whether to print timing messages
        :return:
            data: a dict containing results from different domains (Bayer, RGB, and YCbCr)
                and the final RGB output (data['output'])
            intermediates: a dict containing intermediate results if save_intermediates=True,
                otherwise a empty dict
        """

        pipeline_start = time.time()

        data = OrderedDict(bayer=bayer)
        intermediates = OrderedDict()

        for module_name, module in self.modules.items():
            start = time.time()
            if verbose:
                print('Executing {}... '.format(module_name), end='', flush=True)

            module.execute(data)
            if save_intermediates:
                intermediates[module_name] = copy.copy(data)

            if verbose:
                print('Done. Elapsed {:.3f}s'.format(time.time() - start))

        data['output'] = self.get_rgb_output(data)

        if verbose:
            print('Pipeline elapsed {:.3f}s'.format(time.time() - pipeline_start))

        return data, intermediates

    def get_rgb_output(self, data):
        """
        Post-process the pipeline result to get the final RGB output
        :param data: argument returned by self.execute()
        :return: displayable result: np.ndarray(H, W, 3) in np.uint8 dtype
        """

        if 'y_image' in data and 'cbcr_image' in data:
            ycrcb_image = np.dstack([
                data['y_image'][..., None], data['cbcr_image'][..., ::-1]
            ])
            rgb_output = cv2.cvtColor(ycrcb_image, code=cv2.COLOR_YCrCb2RGB)
        elif 'rgb_image' in data:
            rgb_output = data['rgb_image']
            if rgb_output.dtype != np.uint8:
                rgb_output = rgb_output.astype(np.float32)
                rgb_output = (255 * rgb_output / self.cfg.saturation_values.hdr).astype(np.uint8)
        elif 'bayer' in data:
            rgb_output = data['bayer']  # actually not an RGB image, looks very dark for most cameras
            rgb_output = rgb_output.astype(np.float32)
            rgb_output = (255 * rgb_output / self.cfg.saturation_values.raw).astype(np.uint8)
        else:
            raise NotImplementedError

        return rgb_output

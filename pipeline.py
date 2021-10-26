# File: pipeline.py
# Description: Main pipeline of fast-openISP
# Created: 2021/10/22 20:34
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import time
import copy
import importlib
from collections import OrderedDict

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
            package = importlib.import_module('modules.{}'.format(module_name))
            module_cls = getattr(package, module_name.upper())
            module = module_cls(self.cfg)

            if hasattr(module, 'dependent_modules'):
                for m in module.dependent_modules:
                    if m not in enabled_modules:
                        raise RuntimeError('{} is available only if {} is activated'.format(module_name, m))

            self.modules[module_name] = module

    def get_saturation_values(self):
        """
        Get saturation pixel values in different stages in the pipeline.
        Raw stage: dataflow before the BLC modules (not included)
        HDR stage: dataflow after the BLC modules (included) and before the bit-depth compression
            module, i.e., Gamma in openISP (not included)
        SDR stage: dataflow after the Gamma module (included)
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

        data['output'] = self.get_output(data)

        if verbose:
            print('Pipeline elapsed {:.3f}s'.format(time.time() - pipeline_start))

        return data, intermediates

    def get_output(self, data):
        """
        Post-process the pipeline result to get the final output
        :param data: argument returned by self.execute()
        :return: displayable result: np.ndarray(H, W, 3) in np.uint8 dtype
        """

        if 'y_image' in data and 'cbcr_image' in data:
            ycbcr_image = np.dstack([data['y_image'][..., None], data['cbcr_image']])
            output = ycbcr_to_rgb(ycbcr_image)
        elif 'rgb_image' in data:
            output = data['rgb_image']
            if output.dtype != np.uint8:
                output = output.astype(np.float32)
                output = (255 * output / self.cfg.saturation_values.hdr).astype(np.uint8)
        elif 'bayer' in data:
            output = data['bayer']  # actually not an RGB image, looks very dark for most cameras
            output = output.astype(np.float32)
            output = (255 * output / self.cfg.saturation_values.raw).astype(np.uint8)
        else:
            raise NotImplementedError

        return output


def ycbcr_to_rgb(ycbcr_array):
    """ Convert YCbCr 3-channel array into sRGB array """

    assert ycbcr_array.dtype == np.uint8

    matrix = np.array([[298, 0, 411],
                       [298, -101, -211],
                       [298, 519, 0]], dtype=np.int32).T  # x256
    bias = np.array([-57344, 34739, -71117], dtype=np.int32).reshape(1, 1, 3)  # x256

    ycbcr_array = ycbcr_array.astype(np.int32)
    rgb_array = np.right_shift(ycbcr_array @ matrix + bias, 8)
    rgb_array = np.clip(rgb_array, 0, 255)

    return rgb_array.astype(np.uint8)

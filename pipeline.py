# File: pipeline.py
# Description: Main pipeline of fast-openISP
# Created: 2021/10/22 20:34
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import os.path as op
import sys
import time
import copy
import math
import importlib
from collections import OrderedDict
from multiprocessing import Process

import numpy as np

from utils.yacs import Config
from modules.basic_module import MODULE_DEPENDENCIES


class Pipeline:
    """ Core fast-openISP pipeline """
    def __init__(self, cfg):
        """
        :param cfg: yacs.Config object, configurations about camera specs and module parameters
        """
        self.cfg = cfg

        saturation_values = self.get_saturation_values()
        with self.cfg.unfreeze():
            self.cfg.saturation_values = saturation_values

        self.modules = self.get_modules()

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
            blc = self.cfg.blc
            hdr_max_r = raw_max_value - blc.bl_r
            hdr_max_b = raw_max_value - blc.bl_b
            hdr_max_gr = int(raw_max_value - blc.bl_gr + hdr_max_r * blc.alpha / 1024)
            hdr_max_gb = int(raw_max_value - blc.bl_gb + hdr_max_b * blc.beta / 1024)
            hdr_max_value = max(hdr_max_r, hdr_max_b, hdr_max_gr, hdr_max_gb)
        else:
            hdr_max_value = raw_max_value

        return Config({'raw': raw_max_value,
                       'hdr': hdr_max_value,
                       'sdr': sdr_max_value})

    def get_modules(self):
        """ Get activated ISP modules according to the configuration """
        if op.dirname(__file__) not in sys.path:
            sys.path.insert(0, op.dirname(__file__))

        enabled_modules = tuple(m for m, en in self.cfg.module_enable_status.items() if en)

        modules = OrderedDict()
        for module_name in enabled_modules:
            package = importlib.import_module('modules.{}'.format(module_name))
            module_cls = getattr(package, module_name.upper())
            module = module_cls(self.cfg)

            for m in MODULE_DEPENDENCIES.get(module_cls.__name__, []):
                if m not in enabled_modules:
                    raise RuntimeError(
                        '{} is unavailable when {} is deactivated'.format(module_name, m)
                    )

            modules[module_name] = module

        return modules

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

        def print_(*args, **kwargs):
            return print(*args, **kwargs) if verbose else None

        pipeline_start = time.time()

        data = OrderedDict(bayer=bayer)
        intermediates = OrderedDict()

        for module_name, module in self.modules.items():
            start = time.time()
            print_('Executing {}... '.format(module_name), end='', flush=True)

            module.execute(data)
            if save_intermediates:
                intermediates[module_name] = copy.copy(data)

            print_('Done. Elapsed {:.3f}s'.format(time.time() - start))

        data['output'] = self.get_output(data)
        print_('Pipeline elapsed {:.3f}s'.format(time.time() - pipeline_start))

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

    def run(self, raw_path, save_dir, load_raw_fn, suffix=''):
        """
        A higher level API that writes ISP result into disk
        :param raw_path: path to the raw file to be processed
        :param save_dir: directory to save the output (shares the same filename as the input)
        :param load_raw_fn: function to load the Bayer array from the raw_path
        :param suffix: suffix to added to the output filename
        """
        import cv2

        bayer = load_raw_fn(raw_path)
        data, _ = self.execute(bayer, save_intermediates=False, verbose=False)
        output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)

        filename = op.splitext(op.basename(raw_path))[0]
        save_path = op.join(save_dir, '{}.png'.format(filename + suffix))
        cv2.imwrite(save_path, output)

    def batch_run(self, raw_paths, save_dirs, load_raw_fn, suffixes='', num_processes=1):
        """
        Batch version of self.run via multiprocessing
        :param raw_paths: list of paths to the raw files to be executed
        :param save_dirs: list of directories to save the outputs. If given a string, it will be
            copied to a N-element list, where N is the number of paths in raw_paths
        :param load_raw_fn: function to load the Bayer array from the raw_path
        :param suffixes: a list of suffixes to added to the output filenames
        :param num_processes: number of processes in multiprocessing
        """
        num_files = len(raw_paths)
        num_batches = math.ceil(num_files / num_processes)

        if not isinstance(save_dirs, (list, tuple)):
            save_dirs = [save_dirs for _ in range(num_files)]
        if not isinstance(suffixes, (list, tuple)):
            suffixes = [suffixes for _ in range(num_files)]

        for batch_id in range(num_batches):
            indices = [batch_id * num_processes + rank for rank in range(num_processes)]
            indices = [i for i in indices if i < num_files]
            batch_size = len(indices)

            raw_paths_batch = [raw_paths[i] for i in indices]
            save_dirs_batch = [save_dirs[i] for i in indices]
            suffixes_batch = [suffixes[i] for i in indices]

            pool = []
            for rank in range(batch_size):
                pool.append(
                    Process(target=self.run,
                            kwargs={'raw_path': raw_paths_batch[rank],
                                    'save_dir': save_dirs_batch[rank],
                                    'load_raw_fn': load_raw_fn,
                                    'suffix': suffixes_batch[rank]})
                )

            for p in pool:
                p.start()

            for p in pool:
                p.join()


def ycbcr_to_rgb(ycbcr_array):
    """ Convert YCbCr 3-channel array into sRGB array """
    assert ycbcr_array.dtype == np.uint8

    matrix = np.array([[298, 0, 409],
                       [298, -100, -208],
                       [298, 516, 0]], dtype=np.int32).T  # x256
    bias = np.array([-56992, 34784, -70688], dtype=np.int32).reshape(1, 1, 3)  # x256

    ycbcr_array = ycbcr_array.astype(np.int32)
    rgb_array = np.right_shift(ycbcr_array @ matrix + bias, 8)
    rgb_array = np.clip(rgb_array, 0, 255)

    return rgb_array.astype(np.uint8)

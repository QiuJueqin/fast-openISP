import os
import os.path as op

import cv2
import numpy as np
import skimage.io

from pipeline import Pipeline
from utils.yacs import Config


output_dir = './output'
os.makedirs(output_dir, exist_ok=True)


def demo_test_raw():
    print('Processing test raw started')

    cfg = Config('configs/test.yaml')
    pipeline = Pipeline(cfg)

    raw_path = 'raw/test.RAW'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))

    data, _ = pipeline.execute(bayer)

    output_path = op.join(output_dir, 'test.png')
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)


def demo_nikon_d3x():
    print('\nProcessing Nikon D3x raw started')

    cfg = Config('configs/nikon_d3x.yaml')
    pipeline = Pipeline(cfg)

    pgm_path = 'raw/color_checker.pgm'
    bayer = skimage.io.imread(pgm_path).astype(np.uint16)

    data, _ = pipeline.execute(bayer)

    output_path = op.join(output_dir, 'color_checker.png')
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)


if __name__ == '__main__':
    demo_test_raw()
    demo_nikon_d3x()

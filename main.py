import skimage.io
import numpy as np

from isp_pipeline import Pipeline
from utils.yacs import Config


def load_raw(raw_path):
    bayer = np.fromfile(raw_path, dtype=np.uint16)
    return bayer.reshape((1080, 1920))


def main():
    cfg = Config('configs/test.yaml')
    pipeline = Pipeline(cfg)

    test_raw_path = 'raw/test.RAW'
    bayer = load_raw(test_raw_path)

    data, _ = pipeline.execute(bayer)

    rgb_path = test_raw_path.replace('.RAW', '.png')
    skimage.io.imsave(rgb_path, data['output'])


if __name__ == '__main__':
    main()

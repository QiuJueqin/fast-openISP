# File: ceh.py
# Description: Contrast Enhancement (contrast limited adaptive histogram equalization)
# Created: 2021/11/12 21:46
# Author: Qiu Jueqin (qiujueqin@gmail.com)


import math
import numpy as np

from .basic_module import BasicModule, register_dependent_modules
from .helpers import pad, crop


@register_dependent_modules('csc')
class CEH(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.y_tiles, self.x_tiles = self.params.tiles
        assert self.y_tiles >= 2 and self.x_tiles >= 2, 'only tiles >= 2 is supported'

        self.tile_height = math.ceil(cfg.hardware.raw_height / self.y_tiles)
        self.tile_width = math.ceil(cfg.hardware.raw_width / self.x_tiles)

        y_pads = self.tile_height * self.y_tiles - cfg.hardware.raw_height
        x_pads = self.tile_width * self.x_tiles - cfg.hardware.raw_width
        self.pads = (y_pads // 2, y_pads - y_pads // 2, x_pads // 2, x_pads - x_pads // 2)

        # Weights for LUTs interpolation
        self.left_lut_weights = np.linspace(1024, 0, self.tile_width, dtype=np.int32).reshape((1, -1))  # x1024
        self.top_lut_weights = np.linspace(1024, 0, self.tile_height, dtype=np.int32).reshape((-1, 1))  # x1024

        self.luts = np.empty(shape=(self.y_tiles, self.x_tiles, 256), dtype=np.uint8)  # LUTs w.r.t. tiles

    def execute(self, data):
        y_image = data['y_image'].astype(np.int32)
        y_image = pad(y_image, pads=self.pads)

        # ---------- Generate tile-wise look-up tables ----------
        for ty in range(self.y_tiles):
            for tx in range(self.x_tiles):
                y_tile = y_image[ty * self.tile_height: (ty + 1) * self.tile_height,
                                 tx * self.tile_width: (tx + 1) * self.tile_width]
                self.luts[ty, tx] = self._get_tile_lut(y_tile)

        # ---------- Interpolate and apply LUTs for different image blocks ----------
        ceh_y_image = np.empty_like(y_image).astype(np.uint8)
        for iy in range(self.y_tiles + 1):
            for ix in range(self.x_tiles + 1):
                y0 = iy * self.tile_height - self.tile_height // 2
                y1 = min(y0 + self.tile_height, y_image.shape[0])
                x0 = ix * self.tile_width - self.tile_width // 2
                x1 = min(x0 + self.tile_width, y_image.shape[1])
                y0 = max(y0, 0)
                x0 = max(x0, 0)

                y_block = y_image[y0:y1, x0:x1]

                if self._is_corner_block(ix, iy):
                    lut_y_idx = 0 if iy == 0 else self.y_tiles - 1
                    lut_x_idx = 0 if ix == 0 else self.x_tiles - 1
                    lut = self.luts[lut_y_idx, lut_x_idx]
                    ceh_y_image[y0:y1, x0:x1] = lut[y_block]

                elif self._is_top_or_bottom_block(ix, iy):
                    lut_y_idx = 0 if iy == 0 else self.y_tiles - 1
                    left_lut = self.luts[lut_y_idx, ix - 1]
                    right_lut = self.luts[lut_y_idx, ix]
                    ceh_y_image[y0:y1, x0:x1] = self._interp_top_bottom_block(y_block, left_lut, right_lut)

                elif self._is_left_or_right_block(ix, iy):
                    lut_x_idx = 0 if ix == 0 else self.x_tiles - 1
                    top_lut = self.luts[iy - 1, lut_x_idx]
                    bottom_lut = self.luts[iy, lut_x_idx]
                    ceh_y_image[y0:y1, x0:x1] = self._interp_left_right_block(y_block, top_lut, bottom_lut)

                else:
                    tl_lut = self.luts[iy - 1, ix - 1]
                    tr_lut = self.luts[iy - 1, ix]
                    bl_lut = self.luts[iy, ix - 1]
                    br_lut = self.luts[iy, ix]
                    ceh_y_image[y0:y1, x0:x1] = self._interp_neighbor_block(y_block, tl_lut, tr_lut, bl_lut, br_lut)

        data['y_image'] = crop(ceh_y_image, self.pads)

    def _get_tile_lut(self, tiled_array):
        hist = np.histogram(tiled_array, bins=256, range=(0, self.cfg.saturation_values.sdr))[0]
        clipped_hist = np.clip(hist, 0, self.params.clip_limit * max(hist))

        num_clipped_pixels = (hist - clipped_hist).sum()

        hist = clipped_hist + num_clipped_pixels / 256
        pdf = hist / hist.sum()
        cdf = np.cumsum(pdf)

        lut = (cdf * self.cfg.saturation_values.sdr).astype(np.uint8)
        return lut

    def _interp_top_bottom_block(self, block, left_lut, right_lut):
        return np.right_shift(
            self.left_lut_weights * left_lut[block].astype(np.int32) +
            (1024 - self.left_lut_weights) * right_lut[block].astype(np.int32), 10
        ).astype(np.uint8)

    def _interp_left_right_block(self, block, top_lut, bottom_lut):
        return np.right_shift(
            self.top_lut_weights * top_lut[block].astype(np.int32) +
            (1024 - self.top_lut_weights) * bottom_lut[block].astype(np.int32), 10
        ).astype(np.uint8)

    def _interp_neighbor_block(self, block, tl_lut, tr_lut, bl_lut, br_lut):
        top_block = self._interp_top_bottom_block(block, tl_lut, tr_lut).astype(np.int32)
        bottom_block = self._interp_top_bottom_block(block, bl_lut, br_lut).astype(np.int32)
        return np.right_shift(
            self.top_lut_weights * top_block + (1024 - self.top_lut_weights) * bottom_block, 10
        ).astype(np.uint8)

    def _is_corner_block(self, ix, iy):
        """ Determine if the current image block is locating in a corner region """
        return ((iy == 0 and ix == 0) or
                (iy == 0 and ix == self.x_tiles) or
                (iy == self.y_tiles and ix == 0) or
                (iy == self.y_tiles and ix == self.x_tiles))

    def _is_top_or_bottom_block(self, ix, iy):
        return (iy == 0 or iy == self.y_tiles) and not self._is_corner_block(ix, iy)

    def _is_left_or_right_block(self, ix, iy):
        return (ix == 0 or ix == self.x_tiles) and not self._is_corner_block(ix, iy)

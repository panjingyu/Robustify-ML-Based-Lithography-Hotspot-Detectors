#!/usr/bin/env python3

import argparse
import os
import gdspy
import numpy as np
import cv2
from skimage.draw import rectangle
from tqdm import tqdm

from utils.gds_reader import get_layout_location, shape_to_grid, gen_shapes


def test():
    gds_dir = 'vias/via-merge/test'
    test_dir = 'vias/test'
    test_files = sorted(os.listdir(test_dir))
    gds_files = [d[:-4] for d in test_files]
    gds_file = gds_files[17]
    IN_FILE = os.path.join(gds_dir, gds_file)
    OUT_FILE = gds_file + '.png'
    REF_FILE = os.path.join('vias/test', OUT_FILE)
    ref_img = cv2.imread(REF_FILE, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('ref.png', ref_img)

    gdsii = gdspy.GdsLibrary(infile=IN_FILE)
    layers = gdsii.cells['TOP_new'].get_polygons(by_spec=True)
    srafs = layers[2, 0]
    vias = layers[0, 0]
    x_min, y_min, x_max, y_max = get_layout_location(srafs)
    x_diff = x_max - x_min
    y_diff = y_max - y_min

    STEP = 1e-3
    # X_OFFSET = 568 * STEP
    X_OFFSET = x_diff / 2.5
    # Y_OFFSET = 561 * STEP
    Y_OFFSET = y_diff / 2.5
    LEN = 2048

    gen_img = np.zeros((LEN, LEN), dtype=np.uint8)

    for v in vias:
        x1, x2, y1, y2 = shape_to_grid(v, X_OFFSET, Y_OFFSET, STEP)
        rec = rectangle((y1, x1), end=(y2, x2))
        gen_img[tuple(rec)] = 255

    for s in srafs:
        x1, x2, y1, y2 = shape_to_grid(s, X_OFFSET, Y_OFFSET, STEP)
        rec = rectangle((y1, x1), end=(y2, x2))
        gen_img[tuple(rec)] = 255

    cv2.imwrite('gen.png', gen_img)

    diff = np.full_like(gen_img, fill_value=255)
    diff *= gen_img != ref_img
    # diff *= gen_img == 255
    cv2.imwrite('diff.png', diff)
    print(np.sum(diff) / 255)

def main(args):
    LEN = 2048
    gds_files = sorted(os.listdir(args.gds_dir))
    vias_dir = os.path.join(args.gds_dir, '../png-vias')
    os.makedirs(vias_dir, exist_ok=True)
    srafs_dir = os.path.join(args.gds_dir, '../png-srafs')
    os.makedirs(srafs_dir, exist_ok=True)
    merge_dir = os.path.join(args.gds_dir, '../png-merge')
    os.makedirs(merge_dir, exist_ok=True)
    for gds in tqdm(gds_files):
        gds_path = os.path.join(args.gds_dir, gds)
        gdsii = gdspy.GdsLibrary(infile=gds_path)
        layers = gdsii.cells['TOP_new'].get_polygons(by_spec=True)
        srafs = layers[2, 0]
        vias = layers[0, 0]
        x_min, y_min, x_max, y_max = get_layout_location(srafs)
        assert x_max - x_min < LEN * args.step, 'min = {}, max = {}'.format(x_min, x_max)
        assert y_max - y_min < LEN * args.step, 'min = {}, max = {}'.format(y_min, y_max)
        x_offset = (LEN * args.step - x_max - x_min) / 2
        y_offset = (LEN * args.step - y_max - y_min) / 2
        img_vias = gen_shapes(vias, (LEN, LEN), x_offset, y_offset, args.step)
        img_srafs = gen_shapes(srafs, (LEN, LEN), x_offset, y_offset, args.step)
        vias_png_path = os.path.join(vias_dir, gds + '.vias.png')
        srafs_png_path = os.path.join(srafs_dir, gds + '.srafs.png')
        merge_png_path = os.path.join(merge_dir, gds + '.png')
        cv2.imwrite(vias_png_path, img_vias)
        cv2.imwrite(srafs_png_path, img_srafs)
        cv2.imwrite(merge_png_path, img_vias + img_srafs)

if __name__ == '__main__':
    # test()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gds-dir', type=str, default='./data/train/gds',
                        help='Directory to gds files')
    parser.add_argument('--step', type=float, default=1e-3,
                        help='GDS sampling step size (in microns)')
    args = parser.parse_args()
    assert os.path.isdir(args.gds_dir)
    main(args)

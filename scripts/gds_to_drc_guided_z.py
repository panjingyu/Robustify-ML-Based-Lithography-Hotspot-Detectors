#!/usr/bin/env python3

import argparse
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm

from utils.model import DCT128x128
from utils.adversary import gen_srafs_vias_from_gds, _generate_sraf_sub, get_sraf_add_region


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gds-dir', type=str, default='./data/train/gds',
                        help='Directory to gds files')
    return parser.parse_args()

@torch.no_grad()
def main(args):
    gds_files = sorted(f for f in os.listdir(args.gds_dir) if f.startswith('N'))
    # gds_files = sorted(os.listdir(args.gds_dir))
    z_dir = os.path.join(args.gds_dir, '../drc_guided_z')
    z_png_dir = os.path.join(args.gds_dir, '../drc_guided_z.png')
    os.makedirs(z_dir, exist_ok=True)
    os.makedirs(z_png_dir, exist_ok=True)

    dct = DCT128x128('config/dct-conv-filter.npy').to('cuda').eval()

    for gds in tqdm(gds_files, desc='Parsing masks', colour='green'):
        gds_path = os.path.join(args.gds_dir, gds)
        srafs, vias = gen_srafs_vias_from_gds(gds_path)
        add_region = get_sraf_add_region(vias, srafs).astype(np.float32) * 255
        sub_all = _generate_sraf_sub(srafs).sum(0)
        t = torch.from_numpy(add_region + sub_all).view(1, 1, *sub_all.shape).cuda()
        t.requires_grad_(False)
        z = dct(t).view(32, 16, 16)
        z_path = os.path.join(z_dir, gds + '.pt')
        torch.save(z, z_path)
        # save png
        png = np.stack((add_region, ) * 3, axis=-1)
        png[...,2] -= sub_all
        cv2.imwrite(os.path.join(z_png_dir, gds + '.png'), png)


if __name__ == '__main__':
    # test()
    args = parse_args()
    assert os.path.isdir(args.gds_dir)
    main(args)

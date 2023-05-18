import argparse
import numpy as np
import os

import torch

from utils.model import DCT128x128, DlhsdNetAfterDCT, NetV2AfterDCT
from utils import get_latest_ckpt
from utils.adversary import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved', type=str, default='./log/')
    parser.add_argument('--save-to', type=str, default='./data/adv/')
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--net', type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
    saved_path = args.saved
    latest_ckpt = get_latest_ckpt(os.path.join(saved_path, 'model'))

    '''
    Initialize Path and Global Params
    '''
    test_path = 'config/train-H_all.csv'
    test_list = open(test_path, 'r').readlines()
    blockdim = 16
    fealen = 32
    max_iter = 500
    _max_candidates = 10000
    max_perturbation = 20
    lr = 1e-2

    png_dir = 'data/vias-merge/train/png'
    img_save_dir = os.path.join(args.save_to, 'png')
    dct_save_dir = os.path.join(args.save_to, 'dct')
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(dct_save_dir, exist_ok=True)

    '''
    Prepare the Input
    '''
    test_list_hs = [item.startswith('H') for item in test_list]
    test_list_hs = np.array(test_list_hs)
    idx = np.where(test_list_hs == True) #total = 80152, hs = 6107
    dct = DCT128x128('config/dct-conv-filter.npy').cuda()
    if args.net == 'bn':
        net = NetV2AfterDCT(blockdim, fealen, aug=False).cuda()
    else:
        net = DlhsdNetAfterDCT(blockdim, fealen, aug=False).cuda()
    ckpt_path = latest_ckpt
    net.load_state_dict(torch.load(ckpt_path))
    dct_net = torch.nn.Sequential(dct, net)
    success = 0
    total = 0
    for id in idx[0]:
        try:
            ret = attack(id, dct_net, png_dir, test_list, _max_candidates, max_iter, lr, max_perturbation, img_save_dir)
        except InvalidAttackException:
            continue
        total += 1
        if ret is not None:
            success += 1
            ret = torch.from_numpy(ret).view(1, 1, *ret.shape).to('cuda')
            out = dct(ret).cpu()
            torch.save(out, os.path.join(dct_save_dir, test_list[id].rstrip()[:-4] + '.pt'))
        print(f'success attack: [{success:3d} / {total:3d}]')
    print('All done.')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    main(args)
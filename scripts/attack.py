import argparse
import os
from subprocess import check_call

import torch

from utils.model import DlhsdNetAfterDCT, NetV2AfterDCT
from utils.log_helper import make_logger
from utils import get_latest_ckpt
from utils.adversary import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved', type=str, default='./log/')
    parser.add_argument('--csv', type=str, default='val-num100-seed42')
    parser.add_argument('--net', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('-p', '--max-perturbation', type=int, default=20)
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--skip-finished', action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    saved_path = args.saved
    latest_ckpt = get_latest_ckpt(os.path.join(saved_path, 'model'))
    if not latest_ckpt.endswith('step-40000.pt'):
        print(f'Still training, found {latest_ckpt}, skipped.')
        exit()
    attack_id = f'attack.p{args.max_perturbation}.{args.csv}'
    log_file = os.path.join(saved_path, f'{attack_id}.log')

    '''
    Initialize Path and Global Params
    '''
    # test_path = 'config/test-H_1.csv'
    test_path = f'config/{args.csv}.csv'
    test_list = open(test_path, 'r').readlines()
    test_list = list(filter(
        lambda x: os.path.basename(x).startswith('H'),
        test_list))
    print(f'Found {len(test_list)} hotspots to attack in {test_path}')

    img_save_dir = os.path.join(saved_path, attack_id)
    if os.path.isdir(img_save_dir):
        if args.skip_finished:
            print('Skipping finished attack.')
            exit()
        if not args.override:
            check_call(['rm', '-rf', img_save_dir])
    os.makedirs(img_save_dir, exist_ok=True)

    logger = make_logger(log_file)

    '''
    Prepare the Input
    '''
    blockdim = 16
    fealen = 32
    if args.net == 'bn':
        net = NetV2AfterDCT(blockdim, fealen, aug=False).cuda()
    else:
        net = DlhsdNetAfterDCT(blockdim, fealen, aug=False).cuda()
    ckpt_path = latest_ckpt
    net.load_state_dict(torch.load(ckpt_path))

    max_iter = 500
    max_candidates = 10000
    max_perturbation = args.max_perturbation
    lr = args.lr
    logger.info(f'Attack lr = {lr}')
    run_attack_ggm(net, test_list, logger, img_save_dir, lr, max_iter, max_candidates, max_perturbation)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    main(args)
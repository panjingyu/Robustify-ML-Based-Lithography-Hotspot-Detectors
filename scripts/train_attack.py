import argparse
import numpy as np
import os
import wandb

import torch

from utils import get_timestamp
from utils.log_helper import make_logger, get_log_id
from scripts.train import run_dct_training
from scripts.attack import run_attack


val_num = 100
blockdim = 16
fealen = 32
maxitr = 40000
l_step = 20   #display step
c_step = 100 #check point step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--reg', type=float, default=1e-4)
    parser.add_argument('--cure-l', type=float, default=0.)
    parser.add_argument('--cure-h', type=float, default=0.)
    parser.add_argument('--log', type=str, default='./log/')
    parser.add_argument('--adv', type=str, default=None)
    parser.add_argument('--net', type=str, default=None)
    parser.add_argument('--enable-bias', action='store_true')
    parser.add_argument('--drc-guided', action='store_true')
    parser.add_argument('--drc-include-nonhotspots', action='store_true')
    # attack args
    parser.add_argument('-p', '--max-perturbation', type=int, default=20)
    parser.add_argument('--csv', type=str, default='val-num100-seed42')
    parser.add_argument('--override', action='store_true')
    parser.add_argument('--skip-finished', action='store_true')
    args = parser.parse_args()
    return args


def attack_net(net, saved_path, args):
    attack_id = f'attack.p{args.max_perturbation}.{args.csv}'
    log_file = os.path.join(saved_path, f'{attack_id}.log')

    logger = make_logger(log_file)

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
    os.makedirs(img_save_dir, exist_ok=True)

    '''
    Prepare the Input
    '''
    max_iter = 500
    max_candidates = 10000
    max_perturbation = args.max_perturbation
    lr = args.lr
    logger.info(f'Attack lr = {lr}')
    return run_attack(net, test_list, logger, img_save_dir, lr, max_iter, max_candidates, max_perturbation)



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    args = parse_args()
    log_id = get_log_id(args)
    log_id += f'@{get_timestamp()}'
    saved_path = os.path.join(args.log, log_id)
    config = {
        'hostname': os.uname().nodename,
        'saved path': saved_path,
        'lr': args.lr,
        'bs': args.bs,
        'reg': args.reg,
        'cure-l': args.cure_l,
        'cure-h': args.cure_h,
        'drc-guided': args.drc_guided,
        'drc-include-nonhotspots': args.drc_include_nonhotspots,
        '#max-pert': args.max_perturbation,
    }
    for k in config:
        print(f'{k}:\t{config[k]}')
    wandb.init(project='Vias Security',
               config=config,
               sync_tensorboard=True)
    try:
        net = run_dct_training(args, saved_path)
        success, total = attack_net(net, saved_path, args)
        wandb.log({
            '#success': success, '#trial': total,
            'attack success rate': success / total,
        })
    except KeyboardInterrupt:
        pass
    wandb.finish()
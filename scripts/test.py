import argparse
import time
import os
from tqdm import trange
import cv2

import torch
from utils.model import DlhsdNetAfterDCT, DCT128x128
from utils.log_helper import make_logger
from utils.common import get_latest_ckpt
from utils.metrics import roc_auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved', type=str, default='./saved/')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('-p', '--perts', type=int, default=None)
    args = parser.parse_args()
    return args


@torch.no_grad()
def old_tester(dct, net, test_list, test_png_dir):
    chs = 0   #correctly predicted hs
    ahs = 0   #actual hs
    start = time.time()
    for titr in trange(len(test_list), desc='Detecting ID {}'.format(args.id)):
        png_name = test_list[titr].rstrip()
        test_path = os.path.join(test_png_dir, png_name)
        tdata = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        tdata = torch.from_numpy(tdata).float().cuda()
        x_data = dct(tdata.reshape(1, 1, 2048, 2048))
        out = net(x_data)
        predict = out.argmax(dim=1, keepdim=True)
        chs += predict.item()
        ahs += 1
    if not ahs == 0:
        hs_accu = 1.0*chs/ahs
    end = time.time()
    print('Hotspot Detection Accuracy is %f'%hs_accu)
    print('Test Runtime is %f seconds'%(end-start))

@torch.no_grad()
def test_auc_acc(dct, net, test_list):
    start = time.time()
    labels = torch.Tensor([os.path.basename(t).startswith('H') for t in test_list])
    outs, predicts = [], []
    for i in range(len(test_list)):
        test_path = test_list[i].rstrip()
        tdata = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        tdata = torch.from_numpy(tdata).float().cuda()
        x_data = dct(tdata.reshape(1, 1, 2048, 2048))
        out = net(x_data)
        predict = out.argmax(dim=1, keepdim=True)
        outs.append(out)
        predicts.append(predict)
    outs = torch.stack(outs).cpu().squeeze().numpy()
    outs = outs[...,1] - outs[...,0]
    predicts = torch.stack(predicts).cpu().numpy()
    auc = roc_auc(labels, outs, device='cpu')
    end = time.time()
    print(f'AUC = {auc:.3f}')


@torch.no_grad()
def test_auc_after_atk(dct, net, test_list, adv_dir):
    adv_pngs = filter(lambda x: x.endswith('.png'), os.listdir(adv_dir))
    adv_pngs = sorted(adv_pngs, key=lambda x: int(x.split('.')[0]))
    labels = torch.Tensor([os.path.basename(t).startswith('H') for t in test_list])
    h_cnt = 0
    outs, predicts = [], []
    for i in range(len(test_list)):
        test_path = None
        if labels[i] == 1:
            for x in adv_pngs:
                if x.startswith(f'{h_cnt}.'):
                    test_path = os.path.join(adv_dir, x)
                    tdata = cv2.imread(test_path)[...,1]    # take the G channel
                    break
            h_cnt += 1
        if test_path is None:
            test_path = test_list[i].rstrip()
            tdata = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
        tdata = torch.from_numpy(tdata).float().cuda()
        x_data = dct(tdata.reshape(1, 1, 2048, 2048))
        out = net(x_data)
        predict = out.argmax(dim=1, keepdim=True)
        outs.append(out)
        predicts.append(predict)
    outs = torch.stack(outs).cpu().squeeze().numpy()
    outs = outs[...,1] - outs[...,0]
    predicts = torch.stack(predicts).cpu().numpy()
    auc = roc_auc(labels, outs, device='cpu')
    print(f'AUC after atk = {auc:.3f}')


def main(args):
    saved_path = args.saved
    log_path = os.path.join(saved_path, f'test-{os.path.basename(args.csv)[:-4]}.log')
    logger = make_logger(log_path, log_stdin=True)

    print(args)

    '''
    Prepare the Input
    '''
    with open(args.csv, 'r') as testfile:
        test_list = testfile.readlines()

    blockdim = 16
    fealen = 32

    latest_ckpt = get_latest_ckpt(os.path.join(saved_path, 'model'))
    print(f'Latest checkpoint: {latest_ckpt}')
    net = DlhsdNetAfterDCT(blockdim, fealen, aug=False).cuda()
    net.load_state_dict(torch.load(latest_ckpt))
    net.eval()
    dct = DCT128x128('./config/dct-conv-filter.npy').cuda()
    dct.eval()

    if args.perts is None:
        test_auc_acc(dct, net, test_list)
    else:
        adv_dir = os.path.join(saved_path, f'attack.p{args.perts}.{os.path.basename(args.csv[:-4])}')
        assert os.path.isdir(adv_dir)
        test_auc_after_atk(dct, net, test_list, adv_dir)
        test_auc_acc(dct, net, test_list)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        exit()
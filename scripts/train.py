import argparse
import copy
import numpy as np
import os
from datetime import datetime
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import get_timestamp, to_tensor
from utils.data import Data, DataGds, processlabel, split_valHN_datagds
from utils.model import DlhsdNetAfterDCT, NetV2AfterDCT
from utils.cure import regularizer as cure
from utils.regularizer import l2_reg
from utils.metrics import roc_auc
from utils.log_helper import make_logger, get_log_id


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
    parser.add_argument('--adv', type=str, default='./data/adv/')
    parser.add_argument('--net', type=str, requried=True, default=None)
    parser.add_argument('--enable-bias', action='store_true')
    parser.add_argument('--drc-guided', action='store_true')
    parser.add_argument('--drc-include-nonhotspots', action='store_true')
    args = parser.parse_args()
    return args

def loss_to_bias(loss, alpha, threshold=0.3):
    ''' calculate the bias term for batch biased learning
    args:
        loss: the average loss of current batch with respect to the label without bias
        threshold: start biased learning when loss is below the threshold
    return: the bias value to calculate the gradient
    '''
    if loss >= threshold:
        bias = 0
    else:
        bias = 1.0/(1 + torch.exp(alpha * loss))
    return bias

@torch.no_grad()
def validate(net, data_val, loss):
    net.eval()
    net.aug = False
    x_data = to_tensor(data_val.ft_buffer, blockdim, fealen).cuda()
    y_gt = torch.from_numpy(processlabel(data_val.label_buffer)).cuda()
    val_out = net(x_data)
    val_loss = loss(val_out, y_gt)
    val_predict = val_out.argmax(dim=1, keepdim=True)
    correct = val_predict.eq(y_gt.argmax(dim=1, keepdim=True)).cpu()
    acc = correct.sum() / correct.numel()
    return acc, val_loss, val_out, y_gt

def trainval(**kwargs):
    net = kwargs['net']
    train_data = kwargs['train_data']
    valid_h_data = kwargs['valid_h_data']
    valid_n_data = kwargs['valid_n_data']
    bs = kwargs['batch_size']
    reg = kwargs['reg']
    cure_h, cure_l = kwargs['cure_h'], kwargs['cure_l']
    loss = kwargs['loss']
    opt = kwargs['opt']
    saved_path = kwargs['saved_path']
    tb_writer = kwargs['tb_writer']
    logger = kwargs['logger']
    drc_guided = kwargs['drc_guided']
    enable_bias = kwargs['enable_bias']
    for step in range(maxitr):
        batch = train_data.nextbatch_beta(bs)
        batch_data = batch[0]
        batch_label= batch[1]
        batch_nhs  = batch[2]
        batch_nhs_label = batch[3]
        if drc_guided:
            idxh, idxn = batch[4], batch[5]
            batch_drc_z = train_data.get_drc_z([*idxh, *idxn])
        else:
            batch_drc_z = None
        batch_label_all_without_bias = processlabel(batch_label)
        batch_label_nhs_without_bias = processlabel(batch_nhs_label)
        net.aug = True
        with torch.no_grad():
            x_data = to_tensor(batch_nhs, blockdim, fealen).cuda()
            y_gt = torch.from_numpy(batch_label_nhs_without_bias).cuda()
            net.eval()
            net_out1 = net(x_data)
            nhs_loss = loss(net_out1, y_gt)
            delta1 = loss_to_bias(nhs_loss.detach(), alpha=6)
            batch_label_all_with_bias = processlabel(batch_label, delta1=delta1)
            x_data = to_tensor(batch_data, blockdim, fealen).cuda()
            y_gt = torch.from_numpy(batch_label_all_without_bias).cuda()
            net_out2 = net(x_data)
            training_loss = loss(net_out2, y_gt)
            net_predict2 = net_out2.argmax(dim=1, keepdim=True)
            correct = net_predict2.eq(y_gt.argmax(dim=1, keepdim=True)).cpu()
            training_acc = correct.sum() / correct.numel()
        if enable_bias:
            y_gt = torch.from_numpy(batch_label_all_with_bias).cuda()
        else:
            y_gt = torch.from_numpy(batch_label_all_without_bias).cuda()
        net.train()
        opt.zero_grad()
        reg_cure, norm_grad = cure(net, x_data.detach(), y_gt, loss, drc_guided_z=batch_drc_z, h=cure_h, lambda_=cure_l)
        reg_l2 = l2_reg(net, reg, excluded=('fc2'))
        net_out = net(x_data)
        loss_ = loss(net_out, y_gt)
        loss_ += reg_cure + reg_l2
        loss_.backward()
        opt.step()
        tb_writer.add_scalar('loss/all', training_loss, step)
        tb_writer.add_scalar('acc/training', training_acc, step)
        tb_writer.add_scalar('loss/nhs', nhs_loss, step)
        tb_writer.add_scalar('bias', delta1, step)
        tb_writer.add_scalar('norm_grad', norm_grad, step)
        tb_writer.add_scalar('loss/L2', reg_l2, step)
        tb_writer.add_scalar('loss/CURE', reg_cure, step)
        if step % l_step == 0:
            format_str = ('%s: step %d, loss = %.2f, training_accu = %f, nhs_loss = %.2f, bias = %.3f, norm_grad = %.3f')
            log_line = format_str % (datetime.now(), step, training_loss, training_acc, nhs_loss, delta1, norm_grad)
            logger.info(log_line)
            print(log_line, end='\r')
            tb_writer.flush()
        if (step + 1) % c_step == 0 or (step + 1) == maxitr:
            path = os.path.join(saved_path, f'model/step-{step + 1}.pt')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(net.state_dict(), path)
            # Validation
            val_acc, val_loss, val_out_h, y_gt_h = validate(net, valid_h_data, loss)
            tb_writer.add_scalar('acc/val-H', val_acc, step)
            tb_writer.add_scalar('loss/val-H', val_loss, step)
            val_acc, val_loss, val_out_n, y_gt_n = validate(net, valid_n_data, loss)
            tb_writer.add_scalar('acc/val-N', val_acc, step)
            tb_writer.add_scalar('loss/val-N', val_loss, step)
            val_out = torch.concat([val_out_h, val_out_n])
            val_out = val_out[:,1] - val_out[:,0]
            y_gt = torch.concat([y_gt_h, y_gt_n])[:,1]
            val_auc = roc_auc(y_gt.cpu().numpy(), val_out.cpu().numpy())
            tb_writer.add_scalar('acc/val-AUC', val_auc, step)
    tb_writer.close()

def run_dct_training(args, saved_path):
    os.makedirs(saved_path)
    print(f'Model saved to {saved_path}')
    log_path = os.path.join(saved_path, 'train.log')
    logger = make_logger(log_path)
    tb_writer = SummaryWriter(log_dir=saved_path)

    lr = args.lr
    bs = args.bs
    cure_h, cure_l = args.cure_h, args.cure_l

    train_data = DataGds('data/vias-merge/train/')
    # archive_data = './archive/benchmarks/vias/train'
    # train_data = Data(archive_data, os.path.join(archive_data, 'label.csv'), preload=True)
    train_data, valid_h_data, valid_n_data = split_valHN_datagds(train_data, val_num=val_num)
    if args.drc_guided:
        train_data.load_drc_guided_z(only_hotspots=not args.drc_include_nonhotspots)

    # Include adversarial data
    if args.adv is not None:
        if args.drc_guided:
            raise NotImplementedError('No DRC guided CURE + adv training')
        train_data.load_adv_data(os.path.join(args.adv, 'dct'))

    if args.net == 'bn':
        net = NetV2AfterDCT(blockdim, fealen, aug=True).to('cuda')
    else:
        net = DlhsdNetAfterDCT(blockdim, fealen, aug=True).to('cuda')

    loss = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr, betas=[.9, .999],
                     amsgrad=True)

    if args.enable_bias:
        print('Enable bias')
    trainval(
        net=net,
        train_data=train_data,
        valid_h_data=valid_h_data,
        valid_n_data=valid_n_data,
        reg = args.reg,
        batch_size=bs, drc_guided=args.drc_guided,
        cure_h=cure_h, cure_l=cure_l,
        loss=loss, opt=opt, saved_path=saved_path,
        tb_writer=tb_writer, logger=logger,
        enable_bias=args.enable_bias,
    )
    return net

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
    }
    wandb.init(project='Vias Security',
               config=config,
               sync_tensorboard=True)
    try:
        run_dct_training(args, saved_path)
    except KeyboardInterrupt:
        pass
    wandb.finish()
import os
import numpy as np
import torch
from datetime import datetime

def get_timestamp(fmt=r'%Y%m%d%H%M%S'):
    return datetime.utcnow().strftime(fmt)

def to_tensor(batch: np.ndarray, blockdim, fealen):
    """Transform NHWC numpy array to NCHW PyTorch tensor
    """
    batch_size = batch.shape[0]
    x_data = torch.from_numpy(batch.reshape(batch_size, blockdim, blockdim, fealen))
    x_data = x_data.permute([0, 3, 1, 2])   # NHWC -> NCHW
    return x_data.float()

def get_latest_ckpt(model_dir, prefix='step-', suffix='.pt'):
    ckpt_list = sorted(int(c[len(prefix):-len(suffix)])
                       for c in os.listdir(model_dir)
                       if c.endswith(suffix) and c.startswith(prefix))
    ckpt = prefix + str(ckpt_list[-1]) + suffix
    ckpt = os.path.join(model_dir, ckpt)
    assert os.path.isfile(ckpt), f'{ckpt} is not a file!'
    return ckpt


if __name__ == '__main__':
    print('get_timestamp:', get_timestamp())

import os
import torch
import cv2
from tqdm import tqdm
from utils.model import DCT128x128

def main(png_dir, out_dir):
    dct = DCT128x128('config/mydct_conv.npy').to('cuda')
    fealen = dct.kernel.weight.size(0)
    for png in tqdm(os.listdir(png_dir), desc='Running DCT'):
        png_path = os.path.join(png_dir, png)
        img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
        img = torch.from_numpy(img).view(1, 1, *img.shape).cuda().float()
        out = dct(img).view(fealen, -1).swapaxes(0, 1).cpu().detach()
        out_name = png[:-4] + '.pt'
        torch.save(out, os.path.join(out_dir, out_name))


if __name__ == '__main__':
    png_dir = 'data/test/png-merge'
    assert os.path.isdir(png_dir)
    out_dir = png_dir + '.dct'
    os.makedirs(out_dir, exist_ok=True)
    main(png_dir, out_dir)

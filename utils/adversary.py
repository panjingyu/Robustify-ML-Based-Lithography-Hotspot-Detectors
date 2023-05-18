import numpy as np
import os
import cv2
import torch
from skimage.draw import rectangle
from tqdm import trange

from utils.gds_reader import gen_srafs_vias_from_gds
from utils.model import DCT128x128


min_dis_to_vias = 100
max_dis_to_vias = 500
min_dis_to_sraf = 60
img_size = 2048


class InvalidAttackException(BaseException):
    pass


def get_image_from_input_id(test_file_list, id):
    '''
    return a image and its label
    '''
    filename = test_file_list[id]
    img = cv2.imread(filename.rstrip(), cv2.IMREAD_GRAYSCALE)
    label = os.path.basename(filename).startswith('H')
    return img, label


def _find_shapes(img_):
    shapes = [] #[upper_left_corner_location_y, upper_left_corner_location_x, y_length, x_length]
    img = np.copy(img_)
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i][j] == 255 and img[i-1][j] == 0 and img[i][j-1] == 0:
                j_ = j
                while j_ < img.shape[1]-1 and img[i][j_] == 255:
                    j_ += 1
                x_length = j_ - j
                i_ = i
                while i_ < img.shape[0]-1 and img[i_][j] == 255:
                    i_ += 1
                y_length = i_ - i
                shapes.append([i,j,y_length,x_length])
                img[i:i+y_length][j:j+x_length] = 0
    return np.array(shapes)


def _find_vias(shapes_):
    shapes = np.copy(shapes_)
    squares = shapes[np.where(shapes[:,2]==shapes[:,3])]
    squares_shape = squares[:,2]
    vias_shape = np.amax(squares_shape, axis=0)
    vias_idx = np.where(squares_shape == vias_shape)
    vias = squares[vias_idx]
    srafs = np.delete(shapes, vias_idx, 0)
    return vias, srafs


def _generate_sraf_sub(srafs, save_img=False, save_dir="generate_sraf_sub/"):
    sub = []
    for y1, x1, ylen, xlen in srafs:
        black_img = np.zeros((img_size, img_size), dtype=np.float32)
        rec = rectangle((y1, x1), end=(y1+ylen, x1+xlen))
        black_img[tuple(rec)] = -255.
        sub.append(black_img)
    if save_img:
        for cnt, (y1, x1, ylen, xlen) in enumerate(srafs):
            black_img = np.zeros((img_size, img_size), dtype=np.uint8)
            rec = rectangle((y1, x1), end=(y1+ylen, x1+xlen))
            black_img[tuple(rec)] = 255
            cv2.imwrite(os.path.join(save_dir, f'{cnt + 1:02d}.png'), black_img)
    return np.array(sub, dtype=np.float32)


def plot_srafs_vias(srafs, vias, s_color=255, v_color=np.array((0, 0, 255))):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for y1, x1, ylen, xlen in vias:
        rec = rectangle((y1, x1), end=(y1+ylen, x1+xlen))
        img[tuple(rec)] = v_color
    for y1, x1, ylen, xlen in srafs:
        rec = rectangle((y1, x1), end=(y1+ylen, x1+xlen))
        img[tuple(rec)] = s_color
    return img


def get_sraf_add_region(vias, srafs):
    region = np.zeros(shape=(img_size, img_size), dtype=bool)
    for item in vias:
        center = [item[0]+int(item[2]/2), item[1]+int(item[3]/2)]
        y1 = max(0, center[0] - max_dis_to_vias)
        y2 = min(region.shape[0], center[0] + max_dis_to_vias)
        x1 = max(0, center[1] - max_dis_to_vias)
        x2 = min(region.shape[1], center[1] + max_dis_to_vias)
        region[y1:y2, x1:x2] = True
    for item in vias:
        center = [item[0]+int(item[2]/2), item[1]+int(item[3]/2)]
        y1 = max(0, center[0]-min_dis_to_vias)
        y2 = min(region.shape[0], center[0]+min_dis_to_vias)
        x1 = max(0, center[1]-min_dis_to_vias)
        x2 = min(region.shape[1], center[1]+min_dis_to_vias)
        region[y1:y2, x1:x2] = False
    for item in srafs:
        y1 = max(0, item[0]-min_dis_to_sraf)
        y2 = min(region.shape[0], item[0]+item[2]+min_dis_to_sraf)
        x1 = max(0, item[1]-min_dis_to_sraf)
        x2 = min(region.shape[1], item[1]+item[3]+min_dis_to_sraf)
        region[y1:y2, x1:x2] = False
    return region

def _generate_sraf_add(vias, srafs, insert_shape=[40,90], save_img=False, save_dir="generate_sraf_add/"):
    add = []
    region = get_sraf_add_region(vias, srafs)
    if save_img:
        region_tmp = plot_srafs_vias(srafs, vias, s_color=np.array((0, 255, 0)))
        for i in range(3):
            region_tmp[...,i] += (region * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, '_sraf_region.png'), region_tmp)
    # iterate the space and add sraf one by one.
    # srafs are generated randomly with width = 40 and length in range insert_shape
    for i in range(1, region.shape[0]-1):
        for j in range(1, region.shape[1]-1):
            if region[i][j] == False:
                continue
            shape = np.random.randint(insert_shape[0], high=insert_shape[1]+1, size=2)
            shape[np.random.randint(0,high=2)] = 40
            if i+shape[0] <= region.shape[0] and j+shape[1] <= region.shape[1] and region[i:i+shape[0],j:j+shape[1]].all():
                img = np.zeros((img_size, img_size), dtype=np.uint8)
                img[i:i+shape[0],j:j+shape[1]] = 255
                add.append(img)
                y1 = max(0, i-min_dis_to_sraf)
                y2 = min(region.shape[0], i+shape[0]+min_dis_to_sraf)
                x1 = max(0, j-min_dis_to_sraf)
                x2 = min(region.shape[1], j+shape[1]+min_dis_to_sraf)
                region[y1:y2, x1:x2] = False
    if save_img:
        orig_img = plot_srafs_vias(srafs, vias)
        os.makedirs(save_dir, exist_ok=True)
        for count, item in enumerate(add):
            orig_img[..., 1] += item
            cv2.imwrite(os.path.join(save_dir, f'{count + 1:02d}.png'), orig_img)
            orig_img[..., 1] -= item
    return np.array(add, dtype=np.float32)


def generate_candidates(test_file_list, id, img_save_dir, logger=None, load_gds=False):
    '''
    gengerate all candidates and save them
    '''
    img, _ = get_image_from_input_id(test_file_list, id)
    img_path = test_file_list[id].rstrip()
    img_dir = os.path.dirname(img_path)
    img_name = os.path.basename(img_path)
    if load_gds:
        gds_dir = os.path.join(img_dir, '../gds')
        gds_name = img_name[:-4]
        gds_path = os.path.join(gds_dir, gds_name)
        assert os.path.isfile(gds_path), f'{gds_path} not found!'
        srafs, vias = gen_srafs_vias_from_gds(gds_path)
    else:
        img_shape_dir = os.path.join(img_dir, '_shapes')
        img_shapes_path = os.path.join(img_shape_dir, img_name + '.npy')
        if os.path.isfile(img_shapes_path):
            shapes = np.load(img_shapes_path)
        else:
            shapes = _find_shapes(img)
            os.makedirs(img_shape_dir, exist_ok=True)
            np.save(img_shapes_path, shapes)
        vias, srafs = _find_vias(shapes)
    sraf_img_dir = os.path.join(img_save_dir, f'{id}')
    sraf_sub_img_dir = os.path.join(sraf_img_dir, 'sub')
    os.makedirs(sraf_img_dir, exist_ok=True)
    os.makedirs(sraf_sub_img_dir, exist_ok=True)
    add = _generate_sraf_add(vias, srafs, save_img=True, save_dir=sraf_img_dir) # FIXME: slow! ~7 secs
    sub = _generate_sraf_sub(srafs, save_img=True, save_dir=sraf_sub_img_dir)
    if logger is not None:
        logger.info(f'#candidates generated: add={len(add)} sub={len(sub)}')
    return np.concatenate((add, sub))


def load_candidates(sub_dir="generate_sraf_sub/", add_dir="generate_sraf_add/"):
    '''
    load candidates. call this function if candidates have been saved
    by calling gengerate_candidates() in previous run.
    '''
    print("Loading candidates...")
    X = []
    for root, dirs, files in os.walk(add_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32)
                X.append(img)
    for root, dirs, files in os.walk(sub_dir):
        for name in files:
            if ".png" in name:
                img = np.array(cv2.imread(os.path.join(root,name),0),dtype=np.float32)
                X.append(img)
    print("Loading candidates done. Total candidates: "+str(len(X)))
    return np.array(X)


def generate_adversarial_image_torch(img, X, idx, logger):
    tmp = X[idx].sum(dim=0).view_as(img)
    img_t = img + tmp
    if img_t.max() > 256:
        logger.warn('SRAF overlapping detected!')
        img_t.clip_(0, 255)
    return img_t


def attack_trial(alpha, X, img_t, net, target_idx, max_perturbation, img_save_dir, logger):
    idx = torch.zeros_like(alpha, dtype=torch.bool)
    for i in range(max_perturbation):
        max_idx = alpha.argmax()
        idx[max_idx] = True
        alpha[max_idx] = -float('inf')
        perturbation = X[idx].sum(dim=0).view_as(img_t)
        in_all = img_t + perturbation
        out = net(in_all)
        diff = out[:,1] - out[:,0]
        if diff < 0:
            ret = in_all.cpu().squeeze().numpy()
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            add = ((perturbation > 0) * 255).cpu().squeeze().numpy().astype(np.uint8)
            sub = ((perturbation < 0) * 255).cpu().squeeze().numpy().astype(np.uint8)
            img[...,0] = ret - add
            img[...,1] = ret
            img[...,2] = ret - add + sub
            cv2.imwrite(os.path.join(img_save_dir, f'{target_idx}.pert{i + 1}.png'), img)
            logger.info(f'Found solution with #perturbations = {i + 1}')
            return ret
    return None


def attack(target_idx, net, test_list, max_candidates, max_iter, lr, max_perturbation, img_save_dir, logger):
    # test misclassification
    img, _ = get_image_from_input_id(test_list, target_idx)
    net.eval()
    with torch.no_grad():
        img_t = torch.from_numpy(img).float().view(1, 1, *img.shape).cuda()
        out = net(img_t)
        if out.argmax(dim=1).item() == 0:
            logger.info(f'Misclassification: ID={target_idx}')
            raise InvalidAttackException

    logger.info(f'start attacking on id: {target_idx}')
    X = generate_candidates(test_list, target_idx, img_save_dir, logger=logger, load_gds=True)
    np.random.shuffle(X)
    if max_candidates < X.shape[0]:
        X = X[:max_candidates]
    else:
        max_candidates = X.shape[0]
    X = torch.from_numpy(X).cuda()

    alpha = torch.full((max_candidates,),
                       fill_value=1 / (1 + np.exp(-10)),
                       requires_grad=True,
                       device='cuda')
    la = torch.tensor(1e5, requires_grad=True, device='cuda')
    opt = torch.optim.RMSprop([alpha, la], lr=lr)

    '''
    first attack method by minimizing L(alpha, lambda)
    '''
    interval = 10

    net.eval()
    for iter in range(max_iter):
        opt.zero_grad()
        perturbation = alpha.matmul(X.flatten(1))
        perturbation = perturbation.view_as(img_t)
        loss_1 = perturbation.norm()
        in_all = img_t + perturbation
        out = net(in_all)
        diff = out[:,1] - out[:,0]
        loss = loss_1 + la * diff
        loss.backward()
        opt.step()

        if iter % interval == 0:
            if diff < 0:
                interval = 5
                ret = attack_trial(alpha.detach(), X, img_t, net, target_idx, max_perturbation, img_save_dir, logger)
                if ret is not None:
                    return ret

    logger.info('max iteration reached')
    ret = attack_trial(alpha.detach(), X, img_t, net, target_idx, max_perturbation, img_save_dir, logger)
    if ret is not None:
        return ret

    return None


def test_generate_sraf_add():
    data_dir = './data/vias-merge/train'
    test_sample = 'Hvia21241_mb_mb_lccout.oas.gds'
    gds_path = os.path.join(data_dir, 'gds', test_sample)
    srafs, vias = gen_srafs_vias_from_gds(gds_path)
    _generate_sraf_add(vias, srafs, save_img=True, save_dir='./test_generate_sraf_add')


def run_attack_ggm(net, test_list, logger, img_save_dir, lr, max_iter, max_candidates, max_perturbation):
    dct = DCT128x128('config/dct-conv-filter.npy').cuda()
    dct_net = torch.nn.Sequential(dct, net)
    success = 0
    total = 0
    pbar = trange(len(test_list))
    for id in pbar:
        if any(p.startswith(f'{id}.') and p.endswith('.png') for p in os.listdir(img_save_dir)):
            # Solution found previously, skip
            total += 1
            success += 1
            logger.info(f'ATTACK ON {id} SUCCEEDED (previous solution found)')
            continue
        try:
            ret = attack(id, dct_net, test_list, max_candidates, max_iter, lr, max_perturbation, img_save_dir, logger)
        except InvalidAttackException:
            continue
        total += 1
        if ret is not None:
            success += 1
            logger.info(f'ATTACK ON {id} SUCCEEDED')
        else:
            logger.info(f'ATTACK ON {id} FAILED')
        pbar.set_description_str(f'Attacking: {success:3d}/{total:3d}/{len(test_list):3d}')
        pbar.refresh()
        logger.info(f'Success attacks: [{success:3d} / {total:3d}]')
    pbar.close()
    logger.info('Finished')
    return success, total


if __name__ == '__main__':
    test_generate_sraf_add()

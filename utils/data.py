import csv
import cv2
import random
import os
import copy
import numpy as np
import pandas as pd
import torch
from itertools import islice
from tqdm import tqdm, trange


def readcsv(target, fealen=32):
    #read label
    path  = target + '/label.csv'
    label = np.genfromtxt(path, delimiter=',')
    #read feature
    feature = []
    for dirname, dirnames, filenames in os.walk(target):
        for i in trange(0, len(filenames)-1, desc='Loading data'):
            if i==0:
                file = '/dc.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).values
                feature.append(featemp)
            else:
                file = '/ac'+str(i)+'.csv'
                path = target + file
                featemp = pd.read_csv(path, header=None).values
                feature.append(featemp)
            # print(np.asarray(featemp).shape)
    return np.rollaxis(np.asarray(feature), 0, 3)[:,:,0:fealen], label

'''
    processlabel: adjust ground truth for biased learning
    args:
        label: numpy array contains labels
        cato : number of classes in the task
        delta1: bias for class 1
        delta2: bias for class 2
    return: softmax label with bias
'''
def processlabel(label, cato=2, delta1 = 0, delta2=0):
    softmaxlabel=np.zeros(len(label)*cato, dtype=np.float32).reshape(len(label), cato)
    for i in range(0, len(label)):
        if int(label[i])==0:
            softmaxlabel[i,0]=1-delta1
            softmaxlabel[i,1]=delta1
        if int(label[i])==1:
            softmaxlabel[i,0]=delta2
            softmaxlabel[i,1]=1-delta2
    return softmaxlabel

"""
    data: a class to handle the training and testing data, implement minibatch fetch
    args:
        fea: feature tensor of whole data set
        lab: labels of whole data set
        ptr: a pointer for the current location of minibatch
        maxlen: length of entire dataset
        preload: in current version, to reduce the indexing overhead of SGD, we load all the data into memeory at initialization.
    methods:
        nextinstance():  returns a single instance and its label from the training set, used for SGD
        nextbatch(): returns a batch of instances and their labels from the training set, used for MGD
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label
        sgd_batch(): returns a batch of instances and their labels from the trainin set randomly, number of hs and nhs are equal.
            args:
                batch: minibatch number
                channel: the channel length of feature tersor, lenth > channel will be discarded
                delta1, delta2: see process_label

"""
class Data:
    def __init__(self, fea, lab):
        self.dat=fea
        self.label=lab
        self.ft_buffer, self.label_buffer=readcsv(self.dat)
        self.reset()

    def stat(self):
        total=self.maxlen
        hs=sum(self.label_buffer)
        nhs=total - hs
        return hs, nhs

    def reset(self):
        self.ptr=0
        self.ptr_h=0
        self.ptr_n=0
        self.maxlen=len(self.label_buffer)
        self.labexn = np.where(self.label_buffer==0)[0]
        self.labexh = np.where(self.label_buffer==1)[0]
        self.n_length = self.labexn.size
        self.h_length = self.labexh.size

    def nextinstance(self):
        temp_fea=[]
        label=None
        idx=random.randint(0,self.maxlen)
        for dirname, dirnames, filenames in os.walk(self.dat):
            for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            r=csv.reader(f)
                            fea=[[int(s) for s in row] for j,row in enumerate(r) if j==idx]
                            temp_fea.append(np.asarray(fea))
        with open(self.label) as l:
            temp_label=np.asarray(list(l)[idx]).astype(int)
            if temp_label==0:
                label=[1,0]
            else:
                label=[0,1]
        return np.rollaxis(np.array(temp_fea),0,3),np.array([label])

    def sgd(self, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
        length=labelist.size
        idx=random.randint(0, length-1)
        temp_label=labelist[idx]
        if temp_label==0:
            label=[1,0]
        else:
            label=[0,1]
        ft= self.ft_buffer[idx]

        return ft, np.array(label)
    def sgd_batch_2(self, batch, channel=None, delta1=0, delta2=0):
        with open(self.label) as l:
            labelist=np.asarray(list(l)).astype(int)
            labexn = np.where(labelist==0)[0]
            labexh = np.where(labelist==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        label = processlabel(label,2, 0,0 )
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label


    def sgd_batch(self, batch, channel=None, delta1=0, delta2=0):

        labexn = np.where(self.label_buffer==0)[0]
        labexh = np.where(self.label_buffer==1)[0]
        n_length = labexn.size
        h_length = labexh.size
        if not batch % 2 == 0:
            print('ERROR:Batch size must be even')
            print('Abort.')
            quit()
        else:
            num = batch // 2
        idxn = labexn[(np.random.rand(num)*n_length).astype(int)]
        idxh = labexh[(np.random.rand(num)*h_length).astype(int)]
        label = np.concatenate((np.zeros(num), np.ones(num)))
        #label = processlabel(label,2, delta1, delta2)
        ft_batch = np.concatenate((self.ft_buffer[idxn], self.ft_buffer[idxh]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs
    '''
    nextbatch_beta: returns the balalced batch, used for training only
    '''
    def nextbatch_beta(self, batch):
        def update_ptr(ptr, batch, length):
            return (ptr + batch) % length

        assert batch % 2 == 0, 'Batch size must be even'
        num = batch // 2
        assert num < self.n_length and num < self.h_length, 'Batch size exceeds data size'
        if self.ptr_n+num <self.n_length:
            idxn = self.labexn[self.ptr_n:self.ptr_n+num]
        elif self.ptr_n+num >=self.n_length:
            idxn = np.concatenate((self.labexn[self.ptr_n:], self.labexn[:self.ptr_n+num-self.n_length]))
        self.ptr_n = update_ptr(self.ptr_n, num, self.n_length)
        if self.ptr_h+num <self.h_length:
            idxh = self.labexh[self.ptr_h:self.ptr_h+num]
        elif self.ptr_h+num >=self.h_length:
            idxh = np.concatenate((self.labexh[self.ptr_h:], self.labexh[:self.ptr_h+num-self.h_length]))
        self.ptr_h = update_ptr(self.ptr_h, num, self.h_length)
        label = np.concatenate((np.ones(num), np.zeros(num)))
        ft_batch = np.concatenate((self.ft_buffer[idxh], self.ft_buffer[idxn]))
        ft_batch_nhs = self.ft_buffer[idxn]
        label_nhs = np.zeros(num)
        return ft_batch, label, ft_batch_nhs, label_nhs, idxh, idxn
    '''
    nextbatch_without_balance: returns the normal batch. Suggest to use for training and validation
    '''
    def nextbatch_without_balance_alpha(self, batch, channel=None, delta1=0, delta2=0):
        def update_ptr(ptr, batch, length):
            if ptr+batch<length:
                ptr+=batch
            if ptr+batch>=length:
                ptr=ptr+batch-length
            return ptr
        if self.ptr + batch < self.maxlen:
            label = self.label_buffer[self.ptr:self.ptr+batch]
            ft_batch = self.ft_buffer[self.ptr:self.ptr+batch]
        else:
            label = np.concatenate((self.label_buffer[self.ptr:self.maxlen], self.label_buffer[0:self.ptr+batch-self.maxlen]))
            ft_batch = np.concatenate((self.ft_buffer[self.ptr:self.maxlen], self.ft_buffer[0:self.ptr+batch-self.maxlen]))
        self.ptr = update_ptr(self.ptr, batch, self.maxlen)
        return ft_batch, label
    def nextbatch(self, batch, channel=None, delta1=0, delta2=0):
        #print('recommed to use nextbatch_beta() instead')
        databat=None
        temp_fea=[]
        label=None
        if batch>self.maxlen:
            print('ERROR:Batch size exceeds data size')
            print('Abort.')
            quit()
        if self.ptr+batch < self.maxlen:
            #processing labels
            with open(self.label) as l:
                temp_label=np.asarray(list(l)[self.ptr:self.ptr+batch])
                label=processlabel(temp_label, 2, delta1, delta2)
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            temp_fea.append(np.genfromtxt(islice(f, self.ptr, self.ptr+batch),delimiter=','))
            self.ptr=self.ptr+batch
        elif (self.ptr+batch) >= self.maxlen:

            #processing labels
            with open(self.label) as l:
                a=np.genfromtxt(islice(l, self.ptr, self.maxlen),delimiter=',')
            with open(self.label) as l:
                b=np.genfromtxt(islice(l, 0, self.ptr+batch-self.maxlen),delimiter=',')
            #processing data
            if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                temp_label=b
            elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                temp_label=a
            else:
                temp_label=np.concatenate((a,b))
            label=processlabel(temp_label,2, delta1, delta2)
            #print label.shape
            for dirname, dirnames, filenames in os.walk(self.dat):
                for i in range(0, len(filenames)-1):
                    if i==0:
                        file='/dc.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, None, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
                    else:
                        file='/ac'+str(i)+'.csv'
                        path=self.dat+file
                        with open(path) as f:
                            a=np.genfromtxt(islice(f, self.ptr, self.maxlen),delimiter=',')
                        with open(path) as f:
                            b=np.genfromtxt(islice(f, 0, self.ptr+batch-self.maxlen),delimiter=',')
                        if self.ptr==self.maxlen-1 or self.ptr==self.maxlen:
                            temp_fea.append(b)
                        elif self.ptr+batch-self.maxlen==1 or self.ptr+batch-self.maxlen==0:
                            temp_fea.append(a)
                        else:
                            try:
                                temp_fea.append(np.concatenate((a,b)))
                            except:
                                print (a.shape, b.shape, self.ptr)
            self.ptr=self.ptr+batch-self.maxlen
        #print np.asarray(temp_fea).shape
        return np.rollaxis(np.asarray(temp_fea), 0, 3)[:,:,0:channel], label


class DataGds(Data):
    def __init__(self, path, shuffle_seed=42, load_drc_guided_z=False):
        self.path = path
        self.gds_files = sorted(f for f in os.listdir(os.path.join(path, 'gds'))
                                if f.endswith('.gds'))
        self.seed = shuffle_seed
        random.Random(self.seed).shuffle(self.gds_files)
        self.n_samples = len(self.gds_files)
        print(f'Found {self.n_samples} GDS files in {path}')
        # Load DCT files
        merge_dct_dir = os.path.join(path, 'dct')
        assert os.path.isdir(merge_dct_dir)
        print('Found pre-transformed DCT data')
        dct, label = [], []
        for gds in tqdm(self.gds_files, desc='Loading pt files', colour='blue'):
            dct_path = os.path.join(merge_dct_dir, gds + '.pt')
            t = torch.load(dct_path).detach().numpy()
            dct.append(t)
            label.append(gds.startswith('H'))
        self.ft_buffer = np.stack(dct)
        self.label_buffer = np.array(label, dtype=np.int32)
        self.reset()

    def load_drc_guided_z(self, only_hotspots=False):
        drc_guided_z = []
        for gds in tqdm(self.gds_files, desc='Loading DRC-guided z', colour='green'):
            if not only_hotspots or gds.startswith('H'):
                z_path = os.path.join(self.path, 'drc_guided_z', gds + '.pt')
                t = torch.load(z_path).detach()
                drc_guided_z.append(t)
            else:
                drc_guided_z.append(torch.zeros((32, 16, 16), dtype=torch.float32, device='cuda'))
        self.drc_guided_z = torch.stack(drc_guided_z)

    def load_adv_data(self, path, shuffle_seed=None):
        adv_ft= []
        dct_files = sorted(os.listdir(path))[:500]
        for dct in tqdm(dct_files, desc='Loading adv pt files', colour='blue'):
            dct_path = os.path.join(path, dct)
            t = torch.load(dct_path).detach().numpy()
            adv_ft.append(t)
        adv_ft = np.stack(adv_ft).squeeze()
        adv_ft = adv_ft.reshape(*adv_ft.shape[:2], -1).swapaxes(1, 2)
        adv_label = np.ones(adv_ft.shape[0], dtype=self.label_buffer.dtype)
        self.ft_buffer = np.concatenate((self.ft_buffer, adv_ft), axis=0)
        self.label_buffer = np.hstack((self.label_buffer, adv_label))
        if shuffle_seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(shuffle_seed)
        p = np.random.permutation(self.ft_buffer.shape[0])
        self.ft_buffer = self.ft_buffer[p]
        self.label_buffer = self.label_buffer[p]
        self.reset()

    def get_drc_z(self, idx):
        z = self.drc_guided_z[idx]
        return z

def test_datagds():
    archive_train = 'archive/benchmarks/vias/train/'
    # NOTE: Data.ft_buffer.shape == (n_samples, blockdim * blockdim, fealen)
    # NOTE: Data.label_buffer.shape == (n_samples, )
    # train_data = Data(archive_train, archive_train + 'label.csv', preload=True)
    # print(train_data.ft_buffer.shape)
    # print(train_data.label_buffer.shape)
    # mask = np.ones(len(train_data.label_buffer), dtype=bool)
    # train_data.ft_buffer = train_data.ft_buffer[mask]
    # train_data.label_buffer = train_data.label_buffer[mask]
    # train_data.reset()
    # train_data.stat()
    data_gds = DataGds('data/train/')
    mask = np.ones(len(data_gds.label_buffer), dtype=bool)
    data_gds.ft_buffer = data_gds.ft_buffer[mask]
    data_gds.label_buffer = data_gds.label_buffer[mask]
    data_gds.reset()
    data_gds.stat()

def split_valHN_datagds(train_data, val_num=100):
    def get_png_path(base_dir, gds):
        return os.path.join(
            base_dir, 'png', gds + '.png'
        )
    def filter_path_with_mask(paths, mask):
        ret = []
        for i, m in enumerate(mask):
            if m:
                ret.append(paths[i])
        return ret
    train_dir = train_data.path
    """Hotspot validation set"""
    valid_h_data = copy.deepcopy(train_data)
    hs_idx = np.where(valid_h_data.label_buffer == 1)[0]
    valid_idx = hs_idx[:val_num]
    valid_paths = [get_png_path(train_dir, valid_h_data.gds_files[i]) for i in valid_idx]
    mask = np.ones(len(valid_h_data.label_buffer), dtype=bool)
    mask[valid_idx]=False
    valid_h_data.gds_files = filter_path_with_mask(valid_h_data.gds_files, mask)
    valid_h_data.ft_buffer = valid_h_data.ft_buffer[valid_idx]
    valid_h_data.label_buffer = valid_h_data.label_buffer[valid_idx]
    valid_h_data.reset()
    valid_h_data.stat()

    """Non-hotspot validation set"""
    valid_n_data = copy.deepcopy(train_data)
    nhs_idx = np.where(valid_n_data.label_buffer == 0)[0]
    valid_idx = nhs_idx[:val_num]
    valid_paths += [get_png_path(train_dir, valid_n_data.gds_files[i]) for i in valid_idx]
    mask[valid_idx]=False
    valid_n_data.gds_files = filter_path_with_mask(valid_n_data.gds_files, mask)
    valid_n_data.ft_buffer = valid_n_data.ft_buffer[valid_idx]
    valid_n_data.label_buffer = valid_n_data.label_buffer[valid_idx]
    valid_n_data.reset()
    valid_n_data.stat()

    train_data.gds_files = filter_path_with_mask(train_data.gds_files, mask)
    train_data.ft_buffer = train_data.ft_buffer[mask]
    train_data.label_buffer = train_data.label_buffer[mask]
    train_data.reset()
    train_data.stat()

    valid_csv = f'./config/val-num{val_num}-seed{train_data.seed}.csv'
    with open(valid_csv, 'w') as f:
        f.write('\n'.join(valid_paths))
    return train_data, valid_h_data, valid_n_data


if __name__ == '__main__':
    train_data = DataGds('data/vias-merge/train')
    train_data, _ ,_ = split_valHN_datagds(train_data)
    train_data.load_drc_guided_z()

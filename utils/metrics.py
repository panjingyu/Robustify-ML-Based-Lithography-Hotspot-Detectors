"""Metric tools."""


import numpy as np
import sklearn.metrics
import torch


def rankdata_avg(a, axis=None, device='cpu'):
    if axis is not None:
        a = np.asarray(a)
        if a.size == 0:
            # The return values of `normalize_axis_index` are ignored.  The
            # call validates `axis`, even though we won't use it.
            # use scipy._lib._util._normalize_axis_index when available
            np.core.multiarray.normalize_axis_index(axis, a.ndim)
            return np.empty(a.shape, dtype=np.float64)
        return np.apply_along_axis(rankdata_avg, axis, a, device=device)

    arr = torch.from_numpy(np.asarray(a).ravel()).to(device)
    sorter = torch.argsort(arr)

    inv = torch.empty_like(sorter)
    inv[sorter] = torch.arange(sorter.numel(),
                               device=inv.device,
                               dtype=inv.dtype)

    arr = arr[sorter].cpu().numpy()
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv.cpu().numpy()]

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)


def roc_auc(actual, predicted, device='cpu'):
    """Return ROC-AUC.
    Done in O(nlogn) time, faster than sklearn.metrics.roc_auc_score.
    """
    actual = np.asarray(actual)
    pred_ranks = rankdata_avg(predicted, device=device)
    n_pos = actual.sum()
    n_neg = len(actual) - n_pos
    a = pred_ranks[actual==1].sum() - .5*n_pos*(n_pos+1)
    b = n_pos * n_neg
    if a == 0 or b == 0:
        print(a, b)
        print('n_pos', n_pos)
        print('n_neg', n_neg)
        exit()
    auc = a / b
    # auc = (pred_ranks[actual==1].sum() - .5*n_pos*(n_pos+1)) / (n_pos * n_neg)
    return auc


if __name__ == '__main__':
    pred = np.random.rand(500, 224, 224)
    # pred = np.load('pred.npy')
    print('pred created')
    label = np.random.randint(2, size=pred.shape)
    # label = np.load('target.npy')
    print('label created')

    import time

    start = time.time()
    auc = sklearn.metrics.roc_auc_score(label.ravel(), pred.ravel())
    end = time.time()
    print('sklearn -> time: {:.2f}, auc={:.5f}'.format(end - start, auc))

    start = time.time()
    auc = roc_auc(label.ravel(), pred.ravel(), device='cuda')
    end = time.time()
    print('roc_auc -> time: {:.2f}, auc={:.5f}'.format(end - start, auc))
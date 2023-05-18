"""Regularization."""

import torch.nn as nn


def l2_reg(module, alpha=1., excluded=()):
    terms = []
    for name, m in module.named_modules():
        if any(name.endswith(e) for e in excluded):
            continue # skip this module
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            terms.append(m.weight.norm())
    return alpha * sum(terms)

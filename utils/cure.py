import torch


def find_z(net, inputs, targets, criterion):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()
    outputs = net.eval()(inputs)
    loss_z = criterion(outputs, targets)
    loss_z.backward()
    grad = inputs.grad
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach()
    eps = 1e-7
    z = (z + eps) / (z.flatten(start_dim=1).norm(dim=1).view(z.size(0), 1, 1, 1) + eps)
    inputs.grad.zero_()
    net.zero_grad()

    return z.detach(), norm_grad


def regularizer(net, inputs, targets, criterion, h=3., lambda_=4, drc_guided_z=None):
    '''
    Regularizer term in CURE
    '''
    z, norm_grad = find_z(net, inputs, targets, criterion)
    # z.size() == (bs, 32, 16, 16)
    bs = inputs.size(0)

    if drc_guided_z is not None:
        z = drc_guided_z
        z = z / (z.flatten(start_dim=1).norm(dim=1).view(bs, 1, 1, 1) + 1e-7)

    inputs.requires_grad_()
    outputs_pos = net.eval()(inputs + h * z)
    outputs_orig = net.eval()(inputs)

    loss_pos = criterion(outputs_pos, targets)
    loss_orig = criterion(outputs_orig, targets)
    grad_diff = torch.autograd.grad(loss_pos - loss_orig, inputs,
                                    create_graph=True)[0]
    reg = grad_diff.reshape(bs, -1).norm(dim=1)
    reg = reg * reg
    inputs.grad.zero_()

    return lambda_ * reg.sum() / bs, norm_grad

def test():
    from utils.model import DlhsdNetAfterDCT
    dev = torch.device('cuda')
    net = DlhsdNetAfterDCT(block_dim=16, ft_length=32).to(dev).eval()
    x = torch.randn(4, 32, 16, 16, device=dev)
    z = torch.randn_like(x).sign() * 255.
    targets = torch.zeros(4, dtype=torch.long, device=dev)
    loss = torch.nn.CrossEntropyLoss()
    reg, norm_g = regularizer(net, x, targets, loss, h=.1, lambda_=1, drc_guided_z=z)
    reg.backward()

if __name__ == '__main__':
    test()

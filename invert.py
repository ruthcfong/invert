import torch
import torch.nn as nn

from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

import os

import matplotlib.pyplot as plt

from PIL import Image

def alpha_prior(x, alpha=2.):
    return torch.abs(x.view(-1)**alpha).sum()


def tv_norm(x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = -img[:,:-1,:] + img[:,1:,:]
    dx = -img[:,:,:-1] + img[:,:,1:]
    return ((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum()


def norm_loss(input, target):
    return torch.div(alpha_prior(input - target, alpha=2.), alpha_prior(target, alpha=2.))


class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Clip(object):
    def __init__(self):
        return

    def __call__(self, tensor):
        t = tensor.clone()
        t[t>1] = 1
        t[t<0] = 0
        return t


#function to decay the learning rate
def decay_lr(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor 


def get_pytorch_module(net, blob):
    modules = blob.split('.')
    if len(modules) == 1:
        return net._modules.get(blob)
    else:
        curr_m = net
        for m in modules:
            curr_m = curr_m._modules.get(m)
        return curr_m


def invert(image, network='alexnet', size=227, layer='features.4', alpha=6, beta=2, 
        alpha_lambda=1e-5,  tv_lambda=1e-5, epochs=200, learning_rate=1e2, 
        momentum=0.9, decay_iter=100, decay_factor=1e-1, print_iter=25, 
        cuda=False):

    mu = [0.485, 0.456, 0.406]
    sigma = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Scale(size=size),
        transforms.CenterCrop(size=size),
        transforms.ToTensor(),
        transforms.Normalize(mu, sigma),
    ])

    detransform = transforms.Compose([
        Denormalize(mu, sigma),
        Clip(),
        transforms.ToPILImage(),
    ])

    model = models.__dict__[network](pretrained=True)
    model.eval()
    if cuda:
        model.cuda()

    img_ = transform(Image.open(image)).unsqueeze(0)

    activations = []

    def hook_acts(module, input, output):
        activations.append(output)

    def get_acts(model, input): 
        del activations[:]
        _ = model(input)
        assert(len(activations) == 1)
        return activations[0]

    _ = get_pytorch_module(model, layer).register_forward_hook(hook_acts)
    input_var = Variable(img_.cuda() if cuda else img_)
    ref_acts = get_acts(model, input_var).detach()

    x_ = Variable((1e-3 * torch.randn(*img_.size()).cuda() if cuda else 
        1e-3 * torch.randn(*img_.size())), requires_grad=True)

    alpha_f = lambda x: alpha_prior(x, alpha=alpha)
    tv_f = lambda x: tv_norm(x, beta=beta)
    loss_f = lambda x: norm_loss(x, ref_acts)

    optimizer = torch.optim.SGD([x_], lr=learning_rate, momentum=momentum)

    for i in range(epochs):
        acts = get_acts(model, x_)

        alpha_term = alpha_f(x_)
        tv_term = tv_f(x_)
        loss_term = loss_f(acts)

        tot_loss = alpha_lambda*alpha_term + tv_lambda*tv_term + loss_term

        if (i+1) % print_iter == 0:
            print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f' % (i+1,
                alpha_term.data.cpu().numpy()[0], tv_term.data.cpu().numpy()[0],
                loss_term.data.cpu().numpy()[0], tot_loss.data.cpu().numpy()[0]))

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        if (i+1) % decay_iter == 0:
            decay_lr(optimizer, decay_factor)

    f, ax = plt.subplots(1,2)
    ax[0].imshow(detransform(img_[0]))
    ax[1].imshow(detransform(x_[0].data.cpu()))
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', type=str,
                default='grace_hopper.jpg')
        parser.add_argument('--network', type=str, default='alexnet')
        parser.add_argument('--size', type=int, default=227)
        parser.add_argument('--layer', type=str, default='features.4')
        parser.add_argument('--alpha', type=float, default=6.)
        parser.add_argument('--beta', type=float, default=2.)
        parser.add_argument('--alpha_lambda', type=float, default=1e-5)
        parser.add_argument('--tv_lambda', type=float, default=1e-5)
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--learning_rate', type=int, default=1e2)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--print_iter', type=int, default=25)
        parser.add_argument('--decay_iter', type=int, default=100)
        parser.add_argument('--decay_factor', type=float, default=1e-1)
        parser.add_argument('--gpu', type=int, nargs='*', default=None)

        args = parser.parse_args()

        gpu = args.gpu
        cuda = True if gpu is not None else False
        use_mult_gpu = isinstance(gpu, list)
        if cuda:
            if use_mult_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        print(torch.cuda.device_count(), use_mult_gpu, cuda)

        invert(image=args.image, network=args.network, layer=args.layer, 
                alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda, 
                tv_lambda=args.tv_lambda, epochs=args.epochs,
                learning_rate=args.learning_rate, momentum=args.momentum, 
                print_iter=args.print_iter, decay_iter=args.decay_iter,
                decay_factor=args.decay_factor, cuda=cuda)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)



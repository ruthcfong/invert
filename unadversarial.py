import torch
import torch.nn as nn

from torch.autograd import Variable

import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np

import os
import time

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from PIL import Image, ImageFilter

def get_short_name(label_i, label_names=np.loadtxt('synset_words.txt', str, delimiter='\t')):
    return ' '.join(label_names[label_i-1].split(',')[0].split()[1:])


def get_blurred_image(img_path, sigma = 10):
    return Image.open(img_path).convert('RGB').filter(ImageFilter.GaussianBlur(sigma))

def l1_prior(x, offset=-1):
    return 1./x.view(-1).size(0)*torch.abs(x+offset).sum()

def alpha_prior(x, alpha=2.):
    return 1./x.view(-1).size(0)*(torch.abs(x.view(-1)**alpha).sum())**(1./alpha)


def tv_norm(x, beta=2.):
    assert(x.size(0) == 1)
    img = x[0]
    dy = -img[:,:-1,:] + img[:,1:,:]
    dx = -img[:,:,:-1] + img[:,:,1:]
    return 1./x.view(-1).size(0)*((dx.pow(2) + dy.pow(2)).pow(beta/2.)).sum()


def norm_loss(input, target, selector, alpha=2.):
    return alpha_prior(selector*(input-target), alpha=alpha)
    #return torch.div(alpha_prior(input - target, alpha=alpha), alpha_prior(target, alpha=alpha))

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


mu = [0.485, 0.456, 0.406]
sigma = [0.229, 0.224, 0.225]


def invert(image, network='alexnet', size=227, layer='features.4', selector_idx=None, perturbation='noise', alpha=6,
           beta=2, alpha_lambda=1e-5, tv_lambda=1e-5, epochs=200, learning_rate=1e2, momentum=0.9, decay_iter=100,
           decay_factor=1e-1, print_iter=25, cuda=False):

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

    activation_sizes = []
    def hook_size(module, input, output):
        activation_sizes.append(output.data.size())

    def get_size(model, input):
        del activation_sizes[:]
        _ = model(input)
        assert(len(activation_sizes) == 1)
        return activation_sizes[0]

    def get_act_mags(model, input):
        acts = get_acts(model, input).detach()
        return torch.mean(torch.mean(acts, 3), 2)[0].data.cpu().numpy()

    image_var = Variable(img_.cuda() if cuda else img_)

    module = get_pytorch_module(model, layer)

    # get activation tensor shape
    size_hook = module.register_forward_hook(hook_size)
    act_size = get_size(model, image_var)
    print act_size

    act_hook = module.register_forward_hook(hook_acts)

    softmax = nn.Softmax()
    output_var = model(image_var)
    orig_score, orig_label_i = torch.max(softmax(output_var), 1)
    orig_score = orig_score.data.cpu().numpy()[0]
    orig_label_i = orig_label_i.data.cpu().numpy()[0]
    orig_label = get_short_name(orig_label_i)

    ref_acts = get_acts(model, image_var).detach()

    if selector_idx is None:
        selector = np.ones(act_size)
    else:
        selector = np.zeros(act_size)
        selector[0][selector_idx] = 1
    selector_var = Variable(torch.Tensor(selector).cuda() if cuda else torch.Tensor(selector))

    if perturbation == 'noise':
        x_ = Variable((torch.randn(*img_.size()).cuda() if cuda else
                       torch.randn(*img_.size())), requires_grad=True)
        combine_f = lambda x: image_var + x
        alpha_f = lambda x: -1*alpha_prior(x, alpha=alpha)
    elif perturbation == 'blur':
        x_ = Variable(torch.rand(1,1,28,28).cuda() if cuda else
                      torch.rand(1,1,28,28), requires_grad=True)
        null_img_ = transform(get_blurred_image(image, sigma=10)).unsqueeze(0)
        null_var = Variable(null_img_.cuda() if cuda else null_img_)
        upsample = nn.Upsample(size=image_var.size()[2:], mode='bilinear')
        combine_f = (lambda x: image_var * (1-upsample(x).expand_as(image_var))
                               + null_var * (upsample(x).expand_as(image_var)))
        alpha_f = lambda x: l1_prior(x, offset=1)
    else:
        assert(False)

    loss_f = lambda x: norm_loss(x, ref_acts, selector_var)
    tv_f = lambda x: tv_norm(x, beta=beta)

    optimizer = torch.optim.SGD([x_], lr=learning_rate, momentum=momentum)
    #optimizer = torch.optim.Adam([x_], lr=learning_rate)

    #display_iter = 1 
    display_iter = None

    for i in range(epochs):
        input_var = combine_f(x_)
        acts = get_acts(model, input_var)

        alpha_term = alpha_f(x_)
        loss_term = loss_f(acts)
        tv_term = tv_f(x_)

        tot_loss = loss_term + alpha_lambda*alpha_term + tv_lambda*tv_term

        if (i+1) % print_iter == 0:
            print('Epoch %d:\tAlpha: %f\tTV: %f\tLoss: %f\tTot Loss: %f' % (i+1,
                alpha_lambda*alpha_term.data.cpu().numpy()[0], tv_lambda*tv_term.data.cpu().numpy()[0],
                loss_term.data.cpu().numpy()[0], tot_loss.data.cpu().numpy()[0]))

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

        x_.data.clamp_(0,1)

        if display_iter is not None and (i+1) % display_iter == 0:
            f, ax = plt.subplots(1,3)
            ax[0].imshow(detransform(image_var[0].data.cpu()))
            ax[1].imshow(detransform(x_[0].data.cpu()))
            ax[2].imshow(detransform(input_var[0].data.cpu()))
            for a in ax:
                a.set_xticks([])
                a.set_yticks([])
            plt.show()

        if (i+1) % decay_iter == 0:
            decay_lr(optimizer, decay_factor)

    output_var = model(input_var)
    softmax_scores = softmax(output_var)
    sorted_score, sorted_label_idx = torch.sort(softmax_scores, dim=1, descending=True)
    comb_label_i = sorted_label_idx[0][0].data.cpu().numpy()[0]
    comb_score = sorted_score[0][0].data.cpu().numpy()[0]
    comb_label = get_short_name(comb_label_i)
    orig_rank = np.where(sorted_label_idx[0].data.cpu().numpy() == orig_label_i)[0]
    comb_orig_score = softmax_scores[0][orig_label_i].data.cpu().numpy()[0]

    f, ax = plt.subplots(2,3,figsize=(10,6))
    ax[0][0].imshow(detransform(image_var[0].data.cpu()))
    ax[0][0].set_title('Orig: %s %.2f' % (orig_label, orig_score))
    if perturbation == 'noise':
        ax[0][1].imshow(detransform(x_[0].data.cpu()))
    elif perturbation == 'blur':
        im = ax[0][1].imshow(np.array(detransform(x_[0].data.cpu()))/255.)
        #cax = f.add_axes([0,0,0.5,0.05])
        divider = make_axes_locatable(ax[0][1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        f.colorbar(im, cax=cax, orientation='vertical')
    ax[0][1].set_title('%d: %s %.2f' % (orig_rank, orig_label, comb_orig_score))
    ax[0][2].imshow(detransform(input_var[0].data.cpu()))
    ax[0][2].set_title('Comb: %s %.2f' % (comb_label, comb_score))
    ax[1][0].bar(range(act_size[1]), get_act_mags(model, image_var))
    ax[1][2].bar(range(act_size[1]), get_act_mags(model, input_var))

    print np.argsort(get_act_mags(model, image_var))[::-1][:10]
    print np.argsort(get_act_mags(model, input_var))[::-1][:10]

    for a in ax[0]:
        a.set_xticks([])
        a.set_yticks([])
    plt.show()

    act_hook.remove()


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
        parser.add_argument('--filters', type=int, nargs='*', default=None)
        parser.add_argument('--perturbation', type=str, default='noise')
        parser.add_argument('--start', type=int, default=None)
        parser.add_argument('--end', type=int, default=None)
        parser.add_argument('--alpha', type=float, default=6.)
        parser.add_argument('--beta', type=float, default=2.)
        parser.add_argument('--alpha_lambda', type=float, default=1e-5)
        parser.add_argument('--tv_lambda', type=float, default=1e-5)
        parser.add_argument('--epochs', type=int, default=200)
        parser.add_argument('--learning_rate', type=float, default=1e2)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--print_iter', type=int, default=25)
        parser.add_argument('--decay_iter', type=int, default=100)
        parser.add_argument('--decay_factor', type=float, default=1e-1)
        parser.add_argument('--gpu', type=int, nargs='*', default=None)

        args = parser.parse_args()

        gpu = args.gpu
        cuda = True if gpu is not None else False
        use_mult_gpu = isinstance(gpu, list) and len(gpu) > 1
        if cuda:
            if use_mult_gpu:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu).strip('[').strip(']')
            else:
                if isinstance(gpu, list):
                    assert(len(gpu) == 1)
                    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu[0]
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu
        print(torch.cuda.device_count(), use_mult_gpu, cuda)

        if args.filters is not None:
            selector_idx = args.filters
        elif args.start is not None and args.end is not None:
            selector_idx = range(args.start, args.end)
        else:
            selector_idx = None
        invert(image=args.image, network=args.network, layer=args.layer, selector_idx=selector_idx,
                perturbation=args.perturbation, alpha=args.alpha, beta=args.beta, alpha_lambda=args.alpha_lambda,
                tv_lambda=args.tv_lambda, epochs=args.epochs,
                learning_rate=args.learning_rate, momentum=args.momentum, 
                print_iter=args.print_iter, decay_iter=args.decay_iter,
                decay_factor=args.decay_factor, cuda=cuda)
    except:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)



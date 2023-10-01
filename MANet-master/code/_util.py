# -*- coding: utf-8 -*-
from _options import opt
import numpy as np

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# def adjust_lr(optimizer, init_lr, epoch, total_epoch, decay_rate=0.1, decay_epoch=30):
#     # decay = decay_rate ** (epoch // decay_epoch)
#     if epoch < 4:
#         lr = init_lr * (epoch + 1) / 4
#     else:
#         decay = (1 - epoch * 1.0 / total_epoch) ** 0.9
#         lr = decay * init_lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#     return lr

def adjust_lr(optimizer, epoch):
    if epoch < opt.warmup:
        lr = opt.lr * (epoch + 1) / opt.warmup
    else:
        if opt.scheduler == 'cos':
            lr = opt.lr * 0.5 * (1 + np.cos(np.pi * epoch / opt.epoch))
        elif opt.scheduler == 'poly':
            lr = opt.lr * (1 - epoch * 1.0 / opt.epoch) ** 0.9
        # elif opt.scheduler == 'step':
        #     lr = opt.lr
        #     for milestone in eval(opt.milestones):
        #         if epoch >= milestone:
        #             lr *= opt.gamma
        else:
            raise ValueError('Unknown lr mode {}'.format(opt.scheduler))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
#!/usr/bin/env python

import os

import numpy as np
from PIL import Image

import chainer
import chainer.cuda
from chainer import Variable

def out_image(updater, depro, enc, dec, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        xp = enc.xp

        w_ = 192
        h_ = 192
        in_ch = 3
        out_ch = 1

        in_all = np.zeros((n_images, in_ch, w_, h_)).astype("f")
        gt_all = np.zeros((n_images, out_ch, w_, h_)).astype("f")
        gen_all = np.zeros((n_images, out_ch, w_, h_)).astype("f")

        for it in range(n_images):
            batch = updater.get_iterator('test').next()
            batchsize = len(batch)

            x_in = xp.zeros((batchsize, in_ch, w_, h_)).astype("f")
            t_out = xp.zeros((batchsize, out_ch, w_, h_)).astype("f")

            for i in range(batchsize):

                x_in[i,:] = xp.asarray(batch[i][0])
                t_out[i,:] = xp.asarray(batch[i][1])
            x_in = Variable(x_in)

            x_in2 = depro.predict(x_in)

            z = enc(x_in2)
            x_out = dec(z)

            in_all[it,:] = x_in.data.get()[0,:]
            gt_all[it,:] = t_out.get()[0,:]
            gen_all[it,:] = x_out.data.get()[0,:]


        def save_image(x, name, mode=None):
            _, C, H, W = x.shape
            x = x.reshape((rows, cols, C, H, W))
            x = x.transpose(0, 3, 1, 4, 2)
            if C==1:
                x = x.reshape((rows*H, cols*W))
            else:
                x = x.reshape((rows*H, cols*W, C))

            preview_dir = '{}/preview'.format(dst)
            preview_path = preview_dir +\
                '/image_{:0>8}_{}.png'.format(trainer.updater.iteration, name)
            if not os.path.exists(preview_dir):
                os.makedirs(preview_dir)
            Image.fromarray(x, mode=mode).convert('RGB').save(preview_path)

        x = np.asarray(gen_all * 5 + 5)
        x = x / x.max() * 255
        x = np.asarray(np.clip(x, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gen")

        x = np.asarray(np.clip(in_all * 128+128, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "in")

        x = np.asarray(gt_all * 5 + 5)
        x = x / x.max() * 255
        x = np.asarray(np.clip(x, 0.0, 255.0), dtype=np.uint8)
        save_image(x, "gt")

    return make_image

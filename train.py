#!/usr/bin/env python

# python train_facade.py -g 0 -i ./facade/base --out result_facade --snapshot_interval 10000

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers

from net import Discriminator
from net import Encoder
from net import Decoder
from net import *
from updater import PicUpdater

from img_dataset import *
from pic_visualizer import out_image

# dataset paths
#will fix


def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=500,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result_dehighlight',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=10000,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=1)
    dis = Discriminator(in_ch=3, out_ch=1)
    gen = Generator(in_ch=3, out_ch=1)
    serializers.load_npz( "depro.npz", depro)
    gen.encoder = enc
    gen.decoder = dec

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        enc.to_gpu()  # Copy the model to the GPU
        dec.to_gpu()
        dis.to_gpu()
        gen.to_gpu()

    depro.disable_update()

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)


    #will fix
    train_d = NyuDataset("E:/nyu_depth_v2_labeled.mat",startnum=0,endnum=1000)
    test_d = NyuDataset("E:/nyu_depth_v2_labeled.mat",startnum=1000,endnum=1449)
    #train_iter = chainer.iterators.MultiprocessIterator(train_d, args.batchsize, n_processes=4)
    #test_iter = chainer.iterators.MultiprocessIterator(test_d, args.batchsize, n_processes=4)
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)
    test_iter2 = chainer.iterators.SerialIterator(test_d, args.batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = PicUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter
            },
        optimizer={
            'enc': opt_enc, 'dec': opt_dec,
            'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
                   trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss_enc', 'dec/loss_dec', 'dis/loss_dis', "validation/main/loss"
    ]), trigger=display_interval)
    trainer.extend(extensions.Evaluator(test_iter2, gen, device=args.gpu))
    trainer.extend(extensions.PlotReport(['dec/loss_dec'], x_key='iteration', file_name='dec_loss.png', trigger=display_interval))
    trainer.extend(extensions.PlotReport(['dis/loss_dis'], x_key='iteration', file_name='dis_loss.png', trigger=display_interval))
    trainer.extend(extensions.PlotReport(["validation/main/loss"], x_key='iteration', file_name='gen_loss.png', trigger=display_interval))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_image(
            updater, depro, enc, dec,
            3, 3, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

if __name__ == '__main__':
    main()

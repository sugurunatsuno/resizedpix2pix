#!/usr/bin/env python

from __future__ import print_function

import numpy

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L

# U-net https://arxiv.org/pdf/1611.07004v1.pdf

# convolution-batchnormalization-(dropout)-relu
class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        elif sample=='same':
            layers['c'] = L.Convolution2D(ch0, ch1, 3, 1, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)

    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,6):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(768, 256, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c4'] = CBR(384, 64, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c5'] = L.Convolution2D(128, 256, 3, 1, 1, initialW=w)
        
        layers['d0'] = CBR(256, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=True)
        layers['d1'] = CBR(256, out_ch, bn=True, sample='down', activation=F.leaky_relu, dropout=True)

        super(Decoder, self).__init__(**layers)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1,6):
            h = F.concat([h, hs[-i-1]])
            h = self['c%d'%i](h)
            
        for i in range(0,2):
            h = self['d%d'%i](h)
        return h

class Generator(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers["encoder"] = Encoder(in_ch)
        layers["decoder"] = Decoder(out_ch)
        super(Generator, self).__init__(**layers)

    def __call__(self, x, t=None):
        h = self.encoder(x)
        y = self.decoder(h)
        if not t is None:
            loss = F.mean_squared_error(y, t)
            chainer.report({"loss":loss}, self)
            return loss
        else:
            return y


class Discriminator(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0_0'] = CBR(in_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c0_1'] = CBR(out_ch, 64, bn=False, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c1'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = L.Convolution2D(512, 1, 3, 1, 1, initialW=w)
        super(Discriminator, self).__init__(**layers)

    def __call__(self, x_0, x_1):
        h = F.concat([self.c0_0(x_0), self.c0_1(x_1)])
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        #h = F.average_pooling_2d(h, h.data.shape[2], 1, 0)
        return h
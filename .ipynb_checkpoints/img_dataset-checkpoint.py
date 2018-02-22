import os

import numpy
from PIL import Image
import six

import numpy as np
import cv2

from io import BytesIO
import os
import pickle
import json
import numpy as np
import glob

import skimage.io as io

from chainer.dataset import dataset_mixin

#will fix



class NyuDataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, startnum=0, endnum=2000):

        import h5py
        self.mat = h5py.File(path)

        self.src = self.mat["images"]
        self.dst = self.mat["depths"]

        self.s = startnum
        self.e = endnum

        self.length = -startnum + endnum

    def __len__(self):
        return self.length

    def _sq_trim(self, imgs, size=224):
        shape = imgs[0].shape
        y, x = np.random.randint(shape[0]-size), np.random.randint(shape[1]-size)
        imgs[0], imgs[1] = imgs[0][y:y+size, x:x+size], imgs[1][y:y+size, x:x+size]

        return imgs

    def _flip(self, imgs):
        a = np.random.rand()
        if a < 0.5:
            imgs[0], imgs[1] = cv2.flip(imgs[0], 1), cv2.flip(imgs[1], 1)

        return imgs

    def _resize(self, imgs):
        a = np.random.rand()
        if a < 0.5:
            b = np.random.rand() * 0.1 + 0.95
            imgs[0] = cv2.resize(imgs[0], (int(imgs[0].shape[1]*b), int(imgs[0].shape[0]*b)))
            imgs[1] = cv2.resize(imgs[1], (int(imgs[1].shape[1]*b), int(imgs[1].shape[0]*b)))

        return imgs

    def _add_gaussian_noise(self, imgs):
        a = np.random.rand()
        if a < 0.5:

            def addGaussianNoise(src):
                row,col,ch= src.shape
                mean = 0
                var = 0.1
                sigma = 15
                gauss = np.random.normal(mean,sigma,(row,col,ch))
                gauss = gauss.reshape(row,col,ch)
                noisy = src + gauss.astype(np.uint8)

                return noisy

            for i, d in enumerate(imgs):
                imgs[i] = addGaussianNoise(d)

        return imgs

    def _highcontrast(self, imgs, bottom=50, top=205):
        a = np.random.rand()
        if True:

            def highcontrast(src, bottom=bottom, top=top):
                bottom = np.random.randint(0, bottom)
                top = np.random.randint(top, 255)
                diff = top - bottom
                lut = np.arange(256)

                for i in range(0, bottom):
                    lut[i] = 0
                for i in range(bottom, top):
                    lut[i] = 255 * (i - bottom) / diff
                for i in range(top, 255):
                    lut[i] = 255

                return cv2.LUT(src, lut)

            for i, d in enumerate(imgs):
                imgs[i] = highcontrast(d)

        return imgs

    def _lowcontrast(self, imgs, bottom=50, top=205):
        a = np.random.rand()
        if True:

            def lowcontrast(src, bottom=bottom, top=top):
                bottom = np.random.randint(0, bottom)
                top = np.random.randint(top, 255)
                diff = top - bottom
                lut = np.arange(256)

                for i in range(256):
                    lut[i] = bottom + i * (diff) / 255

                return cv2.LUT(src, lut)

            for i, d in enumerate(imgs):
                imgs[i] = lowcontrast(d)

        return imgs

    def _gammacontrast(self, imgs, bottom=0.7, top=1.3):
        a = np.random.rand()
        if True:

            def gammacontrast(src, bottom=bottom, top=top):
                lut = np.arange(256)
                b = np.random.rand() * (top - bottom) + bottom

                for i in range(256):
                    lut[i] = 255 * pow(float(i) / 255, 1.0 / b)

                return cv2.LUT(src, lut)

            for i, d in enumerate(imgs):
                imgs[i] = gammacontrast(d)

        return imgs

    def get_example(self, i):
        srcArray = np.array(self.src[i - self.e])
        #dstArray = np.array(self.src[i - self.e])
        dstArray = self.dst[i - self.e]

        srcArray = srcArray.transpose(2,1,0)
        dstArray = dstArray.transpose(1,0)
        srcArray = cv2.resize(srcArray, (srcArray.shape[1]//2, srcArray.shape[0]//2))
        dstArray = cv2.resize(dstArray, (dstArray.shape[1]//2//4, dstArray.shape[0]//2//4))

        srcArray, dstArray  = self._flip([srcArray, dstArray])
        srcArray, dstArray  = self._resize([srcArray, dstArray])
        #preprocessed
        srcArray = self._add_gaussian_noise([srcArray])[0]
        contrasrfunc = np.random.choice([self._lowcontrast, self._highcontrast, self._gammacontrast])
        srcArray = contrasrfunc([srcArray])[0]

        srcArray, dstArray  = self._sq_trim([srcArray, dstArray], size=192)
        dstArray = dstArray.reshape(dstArray.shape[0], dstArray.shape[1], 1)

        srcArray = srcArray.astype("f").transpose(2,0,1)/128.0-1.0
        dstArray = dstArray.astype("f").transpose(2,0,1)/5.0-1.0

        return srcArray, dstArray

import os
import torch
import torchvision
import numpy as np
from torchvision import transforms as T
from PIL import Image, ImageOps, ImageFilter
from typing import Callable, Optional, Tuple, Any
from torch.utils.data import Dataset, Sampler


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            transform: Callable = None,
            str_transform: Optional[Callable] = None,
            multi_crop_transform: Optional[Tuple[Any, Callable]] = (6, None),
            supervised: Tuple = (True, 10),
    ):
        super().__init__(root, train, transform, target_transform, download)
        self.supervised, self.views = supervised
        self.trans = transform
        self.mc_trans, self.mc = multi_crop_transform
        self.str_trans = str_transform

        if self.supervised:
            self.targets, self.data = self._text_loader(self.targets, self.data)
            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == t))
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            if self.supervised:
                return *[self.transform(img) for _ in range(self.views)], target
            else:
                img_1 = self.transform(img)
                img_2 = self.transform(img)
                if self.mc > 0 and self.mc_trans is not None:
                    mc_imgs = [self.mc_trans(img) for _ in range(int(self.mc))]
                    return img_1, img_2, *mc_imgs, target
                return img_1, img_2, target
        return img, target

    @staticmethod
    def _text_loader(targets, samples, keep_file=None, training=None):
        """ Transforms applied to dataset at the start of training """
        new_targets, new_samples = [], []
        if training and (keep_file is not None):
            assert os.path.exists(keep_file), 'keep file does not exist'
            with open(keep_file, 'r') as rfile:
                for line in rfile:
                    indx = int(line.split('\n')[0])
                    new_targets.append(targets[indx])
                    new_samples.append(samples[indx])
        else:
            new_targets, new_samples = targets, samples
        return np.array(new_targets), np.array(new_samples)


class ClassStratifiedSampler(Sampler):
    def __init__(
            self,
            data_source,
            world_size,
            rank,
            batch_size=1,
            classes_per_batch=10,
            epochs=1,
            seed=0,
            unique_classes=False
    ):
        super(ClassStratifiedSampler, self).__init__(data_source)
        self.data_source = data_source

        self.rank = rank
        self.world_size = world_size
        self.cpb = classes_per_batch
        self.unique_cpb = unique_classes
        self.batch_size = batch_size
        self.num_classes = len(data_source.classes)
        self.epochs = epochs
        self.outer_epoch = 0

        if not self.unique_cpb:
            assert self.num_classes % self.cpb == 0

        self.base_seed = seed  # instance seed
        self.seed = seed  # subsample sampler seed

    def set_epoch(self, epoch):
        self.outer_epoch = epoch

    def set_inner_epochs(self, epochs):
        self.epochs = epochs

    def _next_perm(self):
        self.seed += 1
        g = torch.Generator()
        g.manual_seed(self.seed)
        self._perm = torch.randperm(self.num_classes, generator=g)

    def _get_perm_ssi(self):
        start = self._ssi
        end = self._ssi + self.cpb
        subsample = self._perm[start:end]
        return subsample

    def _next_ssi(self):
        if not self.unique_cpb:
            self._ssi = (self._ssi + self.cpb) % self.num_classes
            if self._ssi == 0:
                self._next_perm()
        else:
            self._ssi += self.cpb * self.world_size
            max_end = self._ssi + self.cpb * (self.world_size - self.rank)
            if max_end > self.num_classes:
                self._ssi = self.rank * self.cpb
                self._next_perm()

    def _get_local_samplers(self, epoch):
        """ Generate samplers for local data set in given epoch """
        seed = int(self.base_seed + epoch
                   + self.epochs * self.rank
                   + self.outer_epoch * self.epochs * self.world_size)
        g = torch.Generator()
        g.manual_seed(seed)
        samplers = []
        for t in range(self.num_classes):
            t_indices = np.array(self.data_source.target_indices[t])
            if not self.unique_cpb:
                i_size = len(t_indices) // self.world_size
                if i_size > 0:
                    t_indices = t_indices[self.rank * i_size:(self.rank + 1) * i_size]
            if len(t_indices) > 1:
                t_indices = t_indices[torch.randperm(len(t_indices), generator=g)]
            samplers.append(iter(t_indices))
        return samplers

    def _subsample_samplers(self, samplers):
        """ Subsample a small set of samplers from all class-samplers """
        subsample = self._get_perm_ssi()
        subsampled_samplers = []
        for i in subsample:
            subsampled_samplers.append(samplers[i])
        self._next_ssi()
        return zip(*subsampled_samplers)

    def __iter__(self):
        self._ssi = self.rank * self.cpb if self.unique_cpb else 0
        self._next_perm()

        # -- iterations per epoch (extract batch-size samples from each class)
        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size)) * self.batch_size

        for epoch in range(self.epochs):

            # -- shuffle class order
            samplers = self._get_local_samplers(epoch)
            subsampled_samplers = self._subsample_samplers(samplers)

            counter, batch = 0, []
            for i in range(ipe):
                batch += list(next(subsampled_samplers))
                counter += 1
                if counter == self.batch_size:
                    yield batch
                    counter, batch = 0, []
                    if i + 1 < ipe:
                        subsampled_samplers = self._subsample_samplers(samplers)

    def __len__(self):
        if self.batch_size == 0:
            return 0

        ipe = (self.num_classes // self.cpb if not self.unique_cpb
               else self.num_classes // (self.cpb * self.world_size))
        return self.epochs * ipe


def make_transforms(
        size=18,
        scale=(0.3, 0.75),
        color_distortion=0.5,
):

    def get_color_distortion(s):
        # s is the strength of color distortion.
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)

        color_distort = T.Compose([
            rnd_color_jitter,
            Solarize(p=0.2),
            Equalize(p=0.2)])
        return color_distort

    soft_trans = T.Compose(
        [T.RandomResizedCrop(size=size, scale=scale),
         T.RandomHorizontalFlip(),
         T.RandomSolarize(p=1, threshold=0.8),
         T.ToTensor(),
         T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2023, 0.1994, 0.2010))])

    hard_trans = T.Compose(
        [T.RandomResizedCrop(size=size, scale=scale),
         T.RandomHorizontalFlip(),
         get_color_distortion(s=color_distortion),
         T.ToTensor(),
         T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2023, 0.1994, 0.2010))])

    test_trans = T.Compose(
        [T.CenterCrop(size=32),
         T.ToTensor(),
         T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                     std=(0.2023, 0.1994, 0.2010))])
    return soft_trans, hard_trans, test_trans


class Solarize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        v = torch.rand(1) * 256
        return ImageOps.solarize(img, v)


class Equalize(object):
    def __init__(self, p=0.2):
        self.prob = p

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img
        return ImageOps.equalize(img)

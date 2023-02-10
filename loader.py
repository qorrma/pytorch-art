import os
import torch
import torchvision
import numpy as np
from torchvision import transforms as T
from PIL import Image, ImageOps, ImageFilter
from typing import Callable, Optional, Tuple, Any
from torch.utils.data import Dataset, Sampler
from logging import getLogger

logging = getLogger()


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str = './data',
            train: bool = True,
            download: bool = False,
            supervised: Tuple = (False, 10),
            text_path: str = './text',
            transforms: Optional[Tuple[Any, Any]] = (None, None),
            mc_transforms: Optional[Tuple[Any, Any]] = (6, None),
    ):
        super().__init__(root, train, None, None, download)
        self.training = train
        self.supervised, self.views = supervised
        self.trans, self.str_trans = transforms
        self.mc_transforms = mc_transforms

        if self.supervised:
            self.data, self.targets = self._text_loader(
                self.data, self.targets, root=text_path)
            # mint = None
            self.target_indices = []

            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == t))
                self.target_indices.append(indices)
                # mint = len(indices) if mint is None else min(mint, len(indices))
                logging.info(f'num-labeled target {t} {len(indices)}')

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.trans and self.training:
            if self.supervised:
                return *[self.trans(img) for _ in range(self.views)], target
            else:
                img_1 = self.trans(img)
                img_2 = self.str_trans(img)
                mc, mc_trans = self.mc_transforms
                if mc > 0 and mc_trans is not None:
                    mc_imgs = [mc_trans(img) for _ in range(int(mc))]
                    return img_1, img_2, *mc_imgs, target
                return img_1, img_2, target
        else:
            img = self.trans(img)
            return img, target

    @staticmethod
    def _text_loader(samples, targets, root="txt"):
        """ Transforms applied to dataset at the start of training """
        project_path = os.path.dirname(os.path.abspath(__file__))
        keep_file = os.path.join(project_path, root)
        print(keep_file)
        new_targets, new_samples = [], []
        if os.path.exists(keep_file):
            with open(keep_file, 'r') as rfile:
                for line in rfile:
                    indx = int(line.split('\n')[0])
                    new_targets.append(targets[indx])
                    new_samples.append(samples[indx])
        else:
            new_targets, new_samples = targets, samples
        return np.array(new_samples), np.array(new_targets)


class ClassStratifiedSampler(torch.utils.data.Sampler):
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
        """
        ClassStratifiedSampler

        Batch-sampler that samples 'batch-size' images from subset of randomly
        chosen classes e.g., if classes a,b,c are randomly sampled,
        the sampler returns
            torch.cat([a,b,c], [a,b,c], ..., [a,b,c], dim=0)
        where a,b,c, are images from classes a,b,c respectively.
        Sampler, samples images WITH REPLACEMENT (i.e., not epoch-based)

        :param data_source: dataset of type "TransImageNet" or "TransCIFAR10'
        :param world_size: total number of workers in network
        :param rank: local rank in network
        :param batch_size: num. images to load from each class
        :param classes_per_batch: num. classes to randomly sample for batch
        :param epochs: num consecutive epochs thru data_source before gen.reset
        :param seed: common seed across workers for subsampling classes
        :param unique_classes: true ==> each worker samples a distinct set of classes; false ==> all workers sample the same classes
        """
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
                    t_indices = t_indices[self.rank*i_size:(self.rank+1)*i_size]
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
        self._ssi = self.rank*self.cpb if self.unique_cpb else 0
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


def make_transforms(config):
    def get_color_distortion(s):
        # s is the strength of color distortion.
        color_jitter = T.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        rnd_color_jitter = T.RandomApply([color_jitter], p=0.8)

        color_distort = T.Compose([
            rnd_color_jitter,
            Solarize(p=0.2),
            Equalize(p=0.2)])
        return color_distort

    soft_trans = T.Compose([
        T.RandomCrop(size=config.size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=config.mean, std=config.std)
    ])

    hard_trans = T.Compose([
        T.RandomCrop(size=config.size),
        T.RandomHorizontalFlip(),
        get_color_distortion(s=config.color_jitter),
        T.ToTensor(),
        T.Normalize(mean=config.mean, std=config.std)
    ])

    test_trans = T.Compose([
        T.Resize(size=config.size),
        T.ToTensor(),
        T.Normalize(mean=config.mean, std=config.std)
    ])

    if config.multi_crop > 0:
        mc_trans = T.Compose([
            T.RandomResizedCrop(size=config.mc_size, scale=config.mc_scale),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=config.mean, std=config.std)
        ])
        logging.info(f"Multi_crop is employed with ({config.mc_size}) size,"
                     f" and ({config.mc_scale}) scale")
        return soft_trans, hard_trans, mc_trans, test_trans
    return soft_trans, hard_trans, None, test_trans


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


class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


def make_labels_matrix(
    num_classes,
    s_batch_size,
    device,
    smoothing=0.0
):
    # PAWS
    local_images = s_batch_size * num_classes
    total_images = local_images
    off_value = smoothing / num_classes
    labels = torch.zeros(total_images, num_classes).to(device) + off_value
    for i in range(num_classes):
        labels[i::num_classes][:, i] = 1. - smoothing + off_value
    return labels

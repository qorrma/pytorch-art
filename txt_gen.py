import os
import random
import argparse
import numpy as np
import torchvision

proj_path = os.path.dirname(os.path.abspath(__file__))
seeds = [0, 10, 28, 94, 98]


def ArgumentsParse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', type=str, default="imagenet",
                        choices=["cifar10", "cifar100", "svhn", "stl10", "imagenet"])
    parser.add_argument('--txt', type=str, default="txt")
    parser.add_argument('--return_label', type=bool, default=False)
    parser.add_argument('--num', type=int, default=4000)
    parser.add_argument('--frac', type=float, default=0.01)
    return parser.parse_args()


def text_generator(args, return_label):
    if args.data == "cifar10":
        ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        targets = ds.targets
    elif args.data == "cifar100":
        ds = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
        targets = ds.targets
    elif args.data == "svhn":
        ds = torchvision.datasets.SVHN(root='./data', download=True)
        targets = ds.labels
    elif args.data == "stl10":
        ds = torchvision.datasets.STL10(root='./data', download=True)
        targets = ds.labels
    else:
        ds = torchvision.datasets.ImageFolder(root=r'E:\datasets\image\ILSVRC2012\train')
        targets = ds.targets

    num_classes = np.unique(targets)
    ipc = args.num / len(num_classes)

    if args.data == "imagenet":
        ipc = len(num_classes) * args.frac

    print(f"Each class return {int(ipc)} samples")
    save_dir = os.path.join(proj_path, args.txt)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    labeled_idx = []
    for c in num_classes:
        c_data = np.where(targets == c)[0]
        print(f"The number of {c+1}-th class is {len(c_data)}")
        idx = random.sample(range(len(c_data)), k=int(ipc))
        sample_data = c_data[idx].tolist()
        labeled_idx.extend(sample_data)

    if return_label:
        labels = [targets[idx] for idx in labeled_idx]
        return labeled_idx, labels, save_dir
    return labeled_idx, save_dir


if __name__ == '__main__':
    args = ArgumentsParse()

    for seed in seeds:
        random.seed(seed)
        fname = f"{args.data}_{args.num}_{seed}.txt"
        try:
            indices, labels, ckpt = text_generator(args, return_label=args.return_label)
            fname = f"{args.data}_{args.num}_{seed}_target.txt"
            with open(f"{os.path.join(ckpt, fname)}", 'w') as f:
                for label in labels:
                    f.write(f"{label}\n")

        except Exception as e:
            print("labels list is not returned")
            indices, ckpt = text_generator(args, return_label=args.return_label)

        finally:
            with open(f"{os.path.join(ckpt, fname)}", 'w') as f:
                for index in indices:
                    f.write(f"{index}\n")

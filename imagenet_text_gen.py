import os
import random
import argparse
import numpy as np
import torchvision

proj_path = os.path.dirname(os.path.abspath(__file__))
seeds = [0, 10, 28, 94, 98]


def ArgumentsParse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--txt', type=str, default="txt")
    parser.add_argument('--return_label', type=bool, default=False)
    parser.add_argument('--frac', type=float, default=0.01, choices=[0.01, 0.1])
    return parser.parse_args()


def text_generator(args, return_label):
    ds = torchvision.datasets.ImageFolder(root=r'E:\datasets\image\ILSVRC2012\train')
    targets = ds.targets

    num_classes = np.unique(targets)
    save_dir = os.path.join(proj_path, args.txt)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    labeled_idx = []
    for c in num_classes:
        c_data = np.where(targets == c)[0]
        print(f"The number of {c + 1}-th class is {len(c_data)}")
        idx = random.sample(range(len(c_data)), k=round(len(c_data)*args.frac))
        print(f"The {c + 1}-th class return {len(idx)} samples")
        sample_data = c_data[idx].tolist()
        labeled_idx.extend(sample_data)

    if return_label:
        labels = [targets[idx] for idx in labeled_idx]
        return labeled_idx, labels, save_dir
    return labeled_idx, save_dir


if __name__ == '__main__':
    args = ArgumentsParse()
    data = "imagenet"
    for seed in seeds:
        random.seed(seed)
        fname = f"{data}_{args.frac*100}%_{seed}.txt"

        try:
            indices, labels, ckpt = text_generator(args, return_label=args.return_label)
            fname = f"{data}_{args.num}_{seed}_target.txt"
            with open(f"{os.path.join(ckpt, fname)}", 'w') as f:
                for label in labels:
                    f.write(f"{label}\n")
        except Exception as e:
            print(e)
            indices, ckpt = text_generator(args, return_label=args.return_label)

        finally:
            with open(f"{os.path.join(ckpt, fname)}", 'w') as f:
                for index in indices:
                    f.write(f"{index}\n")

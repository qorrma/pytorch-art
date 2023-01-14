import logging
import os
import torch
import random
import logging
import argparse
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp


from loader import (
    CIFAR10,
    ClassStratifiedSampler,
    make_labels_matrix,
    make_transforms)

from utils import (
    Config,
    load_checkpoint,
    WarmupCosineSchedule,
    init_model)

is_cuda, device = None, None


def ArgumentsParse():
    parser = argparse.ArgumentParser(description='Adversarial representation teaching')
    parser.add_argument('--tag', type=str, default='msb2_training')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--alg', type=str, default='msb2')
    parser.add_argument('--cfg', type=str, default='./configs')
    parser.add_argument('--num', type=int, default=4000)
    parser.add_argument('--text', type=str, default='txt')
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--save', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=list, default=[0, 10, 28, 94, 98])
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--adv_noise', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=100)
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def set_cuda(cuda):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda
    global is_cuda, device
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if is_cuda else torch.device('cpu')


def set_logger(filename):
    logging.basicConfig(
        filename=filename,
        format="[%(asctime)s] %(message)s",
        filemode='w+',
        level=logging.INFO)
    logger = logging.getLogger()
    sh = logging.StreamHandler()
    logger.addHandler(sh)
    return logger


if __name__ == '__main__':
    # Set dynamic parameters
    args = ArgumentsParse()
    tag = "[Preparation] "
    log_name = f"{args.data}_{args.num}l_{args.seed}s_{args.num}a.txt"
    logger = set_logger(filename=log_name)
    logger.info(args)

    # print(dist.is_available(), dist.is_initialized())
    # world_size, rank = dist.get_world_size(), dist.get_rank()

    # if dist.is_available() and dist.is_initialized():
    #     world_size, rank = dist.get_world_size(), dist.get_rank()
    #     print(world_size, rank)

    # Set fixed parameters
    cfg = Config(path=args.cfg, name=args.alg)

    if not os.path.exists(cfg.root):
        os.makedirs(cfg.root)

    # Return trained model and log for evaluation
    iterations = len(args.seed)

    # Train step
    tb_iter = []
    for it in range(iterations):
        tag = f"[Training] "
        seed = args.seed[it]

        set_random_seed(seed)
        set_cuda(args.gpu)

        trans, str_trans, test_trans = make_transforms()
        labeled = os.path.join(args.text, f"{args.data}_{args.num}_{seed}.txt")
        sup_set = CIFAR10(root=cfg.root,
                          train=True,
                          download=True,
                          supervised=(True, cfg.view),
                          text_path=labeled,
                          transforms=(trans, None))

        unl_set = CIFAR10(root=cfg.root,
                          train=True,
                          download=True,
                          transforms=(trans, str_trans))

        inf_set = CIFAR10(root=cfg.root,
                          train=False,
                          transforms=(test_trans, None),)
        ipe = len(unl_set)
        logger.info(tag + f"The number of support samples: {len(sup_set)}")
        logger.info(tag + f"The number of unlabeled samples: {len(unl_set)}")
        logger.info(tag + f"The number of test samples: {len(inf_set)}")

        logger.info(tag + f"The iteration per epoch is: {ipe}")
        # breakpoint()

        sampler = ClassStratifiedSampler(
            data_source=sup_set,
            batch_size=cfg.s_batch,
            classes_per_batch=cfg.view,
            epochs=cfg.epochs,
            seed=seed,
            rank=0, world_size=1,
            unique_classes=False)

        sup_loader = DataLoader(
            dataset=sup_set,
            sampler=None,
            batch_size=cfg.s_batch,
            drop_last=cfg.drop_last,
            num_workers=cfg.worker,
            pin_memory=cfg.pin_memory)

        # Generate labels matrix (class_num * views, class_num)
        y_l = make_labels_matrix(
            num_classes=cfg.view,
            s_batch_size=cfg.s_batch,
            world_size=1,
            device=device,
            unique_classes=False,
            smoothing=0.1)
        print(y_l.shape)
        breakpoint()

        un_loader = DataLoader(
            dataset=unl_set,
            batch_size=cfg.u_batch,
            shuffle=cfg.shuffle,
            drop_last=cfg.drop_last,
            num_workers=cfg.worker,
            pin_memory=cfg.pin_memory)

        student = init_model(
            device=device,
            model_name=cfg.model_name,
            use_pred=cfg.use_pred_head,
            output_dim=cfg.output_dim)
        teacher = student

        optim = torch.optim.SGD(
            params=student.parameters(),
            lr=cfg.ref_lr,
            weight_decay=cfg.weight_decay,
            momentum=cfg.momentum)

        sched = WarmupCosineSchedule(
            optim,
            warmup_steps=cfg.warm_up * ipe,
            start_lr=cfg.min_lr,
            ref_lr=cfg.ref_lr,
            final_lr=cfg.max_lr,
            T_max=cfg.epochs * ipe)

        start_epoch = 0
        latest_path = os.path.join(args.save, f'{args.tag}-latest.pth.tar')
        best_path = os.path.join(args.save, f'{args.tag}-best.pth.tar')

        if args.load_model:
            student, optimizer, start_epoch = load_checkpoint(
                path=latest_path,
                model=student,
                opt=optim)

            for _ in range(start_epoch):
                # for _ in range(args.ipe):
                sched.step()

        best_loss = 10
        for ep in range(start_epoch, cfg.epochs):
            student.train()
            print(f'[{ep + 1}/{cfg.epochs} training...]')

            lr = optim.param_groups[0]['lr']
            for idx, (u, u_s, target) in enumerate(un_loader):
                sup_ds = iter(sup_loader)
                x_l = next(sup_ds)

                labels = torch.cat([y_l for _ in range(cfg.view)])
                ep_l = [s.to(device, non_blocking=True) for s in x_l[:-1]]

                with torch.cuda.amp.autocast(enabled=True):
                    optim.zero_grad()
                    z_s = student(ep_l)
                    # z = student(torch.cat([u, u_s], dim=0))
                loss = F.cross_entropy(input=z_s, target=labels)
                loss.backward()
                optim.step()
                sched.step()

                if it % args.log_freq == 0:
                    print(f'[{it:03d}/{args.ipe:03d}]-loss: {loss:.4f}, lr: {lr:.4f}')
            save_dict = {'model': student.state_dict(),
                         'opt': optim.state_dict(),
                         'epoch': ep + 1,
                         'lr': lr}
            torch.save(save_dict, latest_path)
        seed += 1

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
from math import ceil
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
    init_model,
    init_art_loss)

is_cuda, device = None, None


def ArgumentsParse():
    parser = argparse.ArgumentParser(description='Adversarial representation teaching')
    parser.add_argument('--tag', type=str, default='art')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--alg', type=str, default='art')
    parser.add_argument('--cfg', type=str, default='./configs')
    parser.add_argument('--num', type=int, default=4000)
    parser.add_argument('--text', type=str, default='txt')
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--save', type=str, default='./checkpoints')
    parser.add_argument('--seed', type=list, default=10)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--adv_noise', type=int, default=100)
    parser.add_argument('--log_freq', type=int, default=10)
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
    device = torch.device("cuda:0") if is_cuda else torch.device("cpu")
    print(f"Computation is performed on {device} device")


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


def sharpen(p, T=0.25):
    sharp_p = p ** (1. / T)
    sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p


def snn(query, supports, labels, tau=0.1):
    """ Soft Nearest Neighbours similarity classifier """
    # Step 1: normalize embeddings
    query = F.normalize(query)
    supports = F.normalize(supports)
    # Step 2: compute similarlity between local embeddings
    return F.softmax(query @ supports.T / tau, dim=1) @ labels


if __name__ == '__main__':
    # Set dynamic parameters
    args = ArgumentsParse()
    tag = "[Preparation] "
    log_name = f"{args.data}_{args.num}l_{args.seed}s_{args.num}a.txt"
    logger = set_logger(filename=log_name)
    logger.info(args)
    # Set fixed parameters
    cfg = Config(path=args.cfg, name=args.alg)
    if not os.path.exists(cfg.root):
        os.makedirs(cfg.root)

    # Train step
    tag = f"[Training] "
    seed = args.seed
    set_random_seed(seed)
    set_cuda(args.gpu)

    trans, str_trans, mc_trans, test_trans = make_transforms(config=cfg)
    labeled = os.path.join(args.text, f"{args.data}_{args.num}_{seed}.txt")
    sup_set = CIFAR10(root=cfg.root,
                      train=True,
                      download=True,
                      supervised=(True, cfg.views),
                      text_path=labeled,
                      transforms=(trans, None))
    unl_set = CIFAR10(root=cfg.root,
                      train=True,
                      download=True,
                      transforms=(trans, str_trans),
                      mc_transforms=(cfg.multi_crop, mc_trans))

    inf_set = CIFAR10(root=cfg.root,
                      train=False,
                      transforms=(test_trans, None),)

    logger.info(tag + f"The number of support samples: {len(sup_set)}")
    logger.info(tag + f"The number of unlabeled samples: {len(unl_set)}")
    logger.info(tag + f"The number of test samples: {len(inf_set)}")

    sampler = ClassStratifiedSampler(
        data_source=sup_set,
        batch_size=cfg.s_batch,
        classes_per_batch=cfg.classes_per_batch,
        seed=seed,
        rank=0, world_size=1,
        unique_classes=False)

    sup_loader = DataLoader(
        dataset=sup_set,
        batch_sampler=sampler,
        num_workers=cfg.worker,
        pin_memory=cfg.pin_memory)
    sup_ds = None

    y_mat = make_labels_matrix(
        num_classes=cfg.classes_per_batch,
        s_batch_size=cfg.s_batch,
        device=device,
        smoothing=0.1)
    y_l = torch.cat([y_mat for _ in range(cfg.views)]).detach()

    un_loader = DataLoader(
        dataset=unl_set,
        batch_size=cfg.u_batch,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
        num_workers=cfg.worker,
        pin_memory=cfg.pin_memory)
    ipe = len(un_loader)
    logger.info(tag + f"The iteration per epoch is: {ipe}")

    if len(sup_loader) > 0:
        tmp = ceil(len(un_loader) / len(sup_loader))
        sampler.set_inner_epochs(tmp)
        logger.info(tag + f'supervised-reset-period {tmp}')

    # Generate labels matrix (class_num * views, class_num)

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

    art = init_art_loss(
        multicrop=cfg.multi_crop,
        tau=cfg.temperature,
        T=cfg.sharpen,
        me_max=cfg.me_max)

    best_loss = 10
    tag = "[Pre-text] "
    num_support = cfg.s_batch * cfg.views * cfg.classes_per_batch

    for ep in range(start_epoch, cfg.epochs):
        student.train()
        print(f'[{ep + 1}/{cfg.epochs} training...]')
        lr = optim.param_groups[0]['lr']
        sup_ds = iter(sup_loader)

        for idx, udata in enumerate(un_loader):
            sup_data = next(sup_ds)
            x_l = [t.to(device, non_blocking=True) for t in sup_data[:-1]]
            uimgs = [u.to(device, non_blocking=True) for u in udata[:-1]]

            with torch.cuda.amp.autocast(enabled=True):
                optim.zero_grad()
                with torch.no_grad():
                    h_l, z_l = teacher(x_l, return_before_head=True)
                h, z = student(uimgs, return_before_head=True)

                with torch.cuda.amp.autocast(enabled=False):
                    target_supports, anchor_supports, = h_l.float(), z_l.float()
                    target_views, anchors_views = h.float(), z.float()

                    # anchors = [weak, strong]
                    # targets = [strong, weak]
                    targets = torch.cat([
                        target_views[cfg.u_batch:2 * cfg.u_batch],
                        target_views[:cfg.u_batch]], dim=0)

                    if cfg.multi_crop > 0:
                        # anchors = [weak, strong, mc]
                        # targets = [strong, weak, mc]
                        mc_target = 0.5 * (targets[:cfg.u_batch] + targets[cfg.u_batch:])
                        targets = torch.cat(
                            [targets, *[mc_target for _ in range(cfg.multi_crop)]], dim=0)

                    # Generate adversarial noise
                    uni_n = torch.randn_like(anchors_views, requires_grad=True)
                    uni_n = F.normalize(uni_n)
                    anchor_uni = anchors_views + cfg.lamda * uni_n
                    target_uni = targets + cfg.lamda * uni_n
                    target_p = snn(target_uni, target_supports, y_l)
                    anchor_p = snn(anchor_uni, anchor_supports, y_l)
                    # ce = F.cross_entropy(input=anchor_p, target=sharpen(target_p))
                    ce = torch.mean(
                        torch.sum(torch.log(anchor_p ** (-sharpen(target_p))), dim=1))
                    adv_n = torch.autograd.grad(ce, uni_n, retain_graph=True)[0]
                    adv_n = F.normalize(adv_n).detach()

                    # Compute criterion
                    anchor_adv = anchors_views + cfg.anchor_epsilon * adv_n
                    target_adv = targets + cfg.target_epsilon * adv_n
                    p = snn(target_adv, target_supports, y_l)
                    acute_p = snn(anchor_adv, anchor_supports, y_l)
                    # art_loss = F.cross_entropy(input=acute_p, target=sharpen(p))
                    art_loss = torch.mean(torch.sum(torch.log(acute_p ** (-sharpen(p))), dim=1))

                    if cfg.me_max:
                        avg_probs = torch.mean(sharpen(acute_p), dim=0)
                        me_loss = -torch.sum(torch.log(avg_probs ** (-avg_probs)))
                    else:
                        me_loss = 0.
                    loss = art_loss + me_loss
            loss.backward()
            optim.step()
            sched.step()

            if cfg.ema > 0:
                ema_factor = min(1 - 1 / (ep + 1), cfg.ema)
                for emp_p, p in zip(teacher.parameters(), student.parameters()):
                    emp_p.data = ema_factor * emp_p.data + (1 - ema_factor) * p.data

            if idx % args.log_freq == 0:
                logging.info(tag + f'{ep+1}E-[{idx+1:03d}/{ipe:03d}]it')
                logging.info(f"loss: {loss:.4f} ({art_loss:.4f}, {me_loss:.4f}), lr: {lr:.4f}")

        save_dict = {'model': student.state_dict(),
                     'opt': optim.state_dict(),
                     'epoch': ep + 1,
                     'lr': lr}
        torch.save(save_dict, latest_path)

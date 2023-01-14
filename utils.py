import os
import yaml
import math
import torch
from collections import OrderedDict


def init_model(
        device,
        model_name='resnet50',
        use_pred=False,
        output_dim=128
):
    if 'wide_resnet' in model_name:
        import models.wide_resnet as wide_resnet
        encoder = wide_resnet.__dict__[model_name](dropout_rate=0.0)
        hidden_dim = 128
    else:
        import models.resnet as resnet
        encoder = resnet.__dict__[model_name]()
        hidden_dim = 2048
        if 'w2' in model_name:
            hidden_dim *= 2
        elif 'w4' in model_name:
            hidden_dim *= 4

    # -- projection head
    encoder.fc = torch.nn.Sequential(OrderedDict([
        ('fc1', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn1', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu1', torch.nn.ReLU(inplace=True)),
        ('fc2', torch.nn.Linear(hidden_dim, hidden_dim)),
        ('bn2', torch.nn.BatchNorm1d(hidden_dim)),
        ('relu2', torch.nn.ReLU(inplace=True)),
        ('fc3', torch.nn.Linear(hidden_dim, output_dim))
    ]))

    # -- prediction head
    encoder.pred = None
    if use_pred:
        mx = 4  # 4x bottleneck prediction head
        pred_head = OrderedDict([])
        pred_head['bn1'] = torch.nn.BatchNorm1d(output_dim)
        pred_head['fc1'] = torch.nn.Linear(output_dim, output_dim // mx)
        pred_head['bn2'] = torch.nn.BatchNorm1d(output_dim // mx)
        pred_head['relu'] = torch.nn.ReLU(inplace=True)
        pred_head['fc2'] = torch.nn.Linear(output_dim // mx, output_dim)
        encoder.pred = torch.nn.Sequential(pred_head)
    encoder.to(device)
    return encoder


class Config:
    def __init__(self, path, name):
        filename = os.path.join(path, name + '.yaml')
        # self.cfg = defaultdict(lambda: None)
        with open(filename, 'r') as f:
            self.cfg = yaml.full_load(f)

    def __getitem__(self, item):
        return self.cfg[item]

    def __setitem__(self, key, value):
        self.cfg[key] = value

    def __delitem__(self, key):
        del self.cfg[key]

    def __getattr__(self, item):
        return self.cfg.get(item, None)

    def __str__(self):
        return str(self.cfg)


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        super(WarmupCosineSchedule, self).__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
            return new_lr / self.ref_lr

        # -- progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.T_max))
        new_lr = max(self.final_lr,
                     self.final_lr
                     + (self.ref_lr - self.final_lr)
                     * 0.5
                     * (1. + math.cos(math.pi * progress)))
        return new_lr / self.ref_lr


def load_checkpoint(
        path,
        model,
        opt,
):
    checkpoint = torch.load(path, map_location='cpu')
    epoch = checkpoint['epoch']
    # -- loading encoder
    model.load_state_dict(checkpoint['model'])
    print(f'loaded model from epoch {epoch}')
    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    print(f'loaded optimizers from epoch {epoch}')
    print(f'read-path: {path}')
    del checkpoint
    return model, opt, epoch

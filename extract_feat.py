import argparse
import os
import numpy as np
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from utils import AverageMeter, load_ckpt, bin_loader
from datasets import BinDataset


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Feature Extractor')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--input-size', default=112, type=int)
parser.add_argument('--feature-dim', default=256, type=int)
parser.add_argument('--load-path', type=str)
parser.add_argument('--bin-file', type=str)
parser.add_argument('--output-path', default='dump.npy', type=str)


class IdentityMapping(nn.Module):
    def __init__(self, base):
        super(IdentityMapping, self).__init__()
        self.base = base
    def forward(self, x):
        x = self.base(x)
        return x


def main():
    global args
    args = parser.parse_args()

    assert args.output_path.endswith('.npy')

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](feature_dim=args.feature_dim)
    model = IdentityMapping(model)

    model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    if args.load_path:
        classifier_keys = ['module.logits.weight', 'module.logits.bias']
        load_ckpt(args.load_path, model, ignores=classifier_keys, strict=True)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.25, 0.25, 0.25])

    test_loader = DataLoader(
        BinDataset(args.bin_file,
            transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    features = extract(test_loader, model)
    assert features.shape[1] == args.feature_dim

    print('saving extracted features to {}'.format(args.output_path))
    folder = os.path.dirname(args.output_path)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    np.save(args.output_path, features)


def extract(test_loader, model):
    batch_time = AverageMeter(10)
    model.eval()
    features = []
    with torch.no_grad():
        end = time.time()
        for i, input in enumerate(test_loader):
            # compute output
            output = model(input)
            features.append(output.data.cpu().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return np.vstack(features)


if __name__ == '__main__':
    main()

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
from utils import AverageMeter, load_ckpt, write_feat
from datasets import FileListDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Feature Extractor')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=1, type=int)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--input-size', default=112, type=int)
parser.add_argument('--feature-dim', default=256, type=int)
parser.add_argument('--load-path', type=str)
parser.add_argument('--prefix', type=str)
parser.add_argument('--filelist', type=str)
parser.add_argument('--strict', dest='strict', action='store_true')
parser.add_argument('--output-path', default='feat_name.bin', type=str)


class IdentityMapping(nn.Module):
    def __init__(self, base):
        super(IdentityMapping, self).__init__()
        self.base = base

    def forward(self, x):
        x = self.base(x)
        return x


class AxisSwap(object):
    def __init__(self, swap=[2, 1, 0]):
        self.swap = swap

    def __call__(self, img):
        img = np.array(img)
        img = img[:, :, [2, 1, 0]]
        return img


def main():
    global args
    args = parser.parse_args()

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](feature_dim=args.feature_dim)
    model = IdentityMapping(model)

    model = torch.nn.DataParallel(model).cuda()

    if args.load_path:
        if args.strict:
            classifier_keys = []
            load_ckpt(args.load_path,
                      model,
                      ignores=classifier_keys,
                      strict=True)
        else:
            load_ckpt(args.load_path, model, strict=False)

    cudnn.benchmark = True

    # normalize input to [-1.6, 1.6]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.3125, 0.3125, 0.3125])

    if args.filelist is not None and os.path.isfile(args.filelist):
        dataset = FileListDataset(
            args.filelist, args.prefix,
            transforms.Compose([
                transforms.Resize(args.input_size),
                AxisSwap(),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        raise FileNotFoundError('Please specify filelist')

    test_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.workers,
                             pin_memory=True)

    features = extract(test_loader, model)
    assert features.shape[1] == args.feature_dim

    print('saving extracted features to {}'.format(args.output_path))
    folder = os.path.dirname(args.output_path)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    if args.output_path.endswith('.bin'):
        write_feat(args.output_path, features)
    elif args.output_path.endswith('.npy'):
        np.save(args.output_path, features)
    else:
        np.savez(args.output_path, features)


def extract(test_loader, model):
    batch_time = AverageMeter(10)
    model.eval()
    features = []
    with torch.no_grad():
        end = time.time()
        for i, x in enumerate(test_loader):
            # compute output
            output = model(x)
            features.append(output.data.cpu().numpy())
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return np.vstack(features)


if __name__ == '__main__':
    main()

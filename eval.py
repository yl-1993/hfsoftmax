import os
import argparse
import numpy as np
import models
from utils import bin_loader, normalize
from evaluation import evaluate

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet50',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('--input-size', default=112, type=int)
parser.add_argument('--feature-dim', default=256, type=int)
parser.add_argument('--load-path', type=str)
parser.add_argument('--bin-file', type=str)
parser.add_argument('--output-path', default='dump.npy', type=str)
parser.add_argument('--nfolds', default=10, type=int)


def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        comm = 'python extract_feat.py \
                --arch {} \
                --batch-size {} \
                --input-size {} \
                --feature-dim {} \
                --load-path {} \
                --bin-file {} \
                --output-path {}'\
                .format(args.arch, args.batch_size, args.input_size, args.feature_dim,
                        args.load_path, args.bin_file, args.output_path)
        print(' '.join(comm.split()))
        os.system(comm)

    features = np.load(args.output_path).reshape(-1, args.feature_dim)
    _, lbs = bin_loader(args.bin_file)
    print('feature shape: {}'.format(features.shape))
    assert features.shape[0] == 2 * len(lbs), "{} vs {}".format(
        features.shape[0], 2 * len(lbs))

    features = normalize(features)
    _, _, acc, val, val_std, far = evaluate(features,
                                            lbs,
                                            nrof_folds=args.nfolds,
                                            distance_metric=0)
    print("accuracy: {:.4f}({:.4f})".format(acc.mean(), acc.std()))


if __name__ == '__main__':
    main()

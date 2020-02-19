import os
import random
import argparse


def w2f(ofn, lst):
    with open(ofn, 'w') as of:
        for x in lst:
            of.write('{} {}\n'.format(*x))


def gen_train_val_list(folder, ofolder, val_ratio, ext='jpg'):
    train_lst = []
    val_lst = []
    for _, dirs, _ in os.walk(folder):
        for d in dirs:
            lb = int(d)
            fns = os.listdir(os.path.join(folder, d))
            random.shuffle(fns)
            n = int(len(fns) * val_ratio)
            n = max(len(fns) - n, 1)
            train_lst.extend([(os.path.join(d, fn), lb) for fn in fns[:n]])
            val_lst.extend([(os.path.join(d, fn), lb) for fn in fns[n:]])
    w2f(os.path.join(ofolder, 'train.txt'), train_lst)
    w2f(os.path.join(ofolder, 'val.txt'), val_lst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',
                        type=str,
                        help="image folder",
                        required=True)
    parser.add_argument('--ofolder',
                        type=str,
                        default='./',
                        help="output folder to save train/val list")
    parser.add_argument('--val-ratio', type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    gen_train_val_list(args.folder, args.ofolder, args.val_ratio)

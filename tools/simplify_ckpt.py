import argparse
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simplify Model')
    parser.add_argument('--in-path',
                        type=str,
                        required=True,
                        help='input pytorch model')
    parser.add_argument('--out-path',
                        type=str,
                        default='',
                        help='output simplified pytorch model')
    parser.add_argument('--ignores',
                        type=str,
                        default='module.logits.weight,module.logits.bias',
                        help='ignored weights')
    args = parser.parse_args()

    ignores = []
    if args.ignores != '':
        ignores = [ignore for ignore in args.ignores.split(',')]
    utils.simplify_ckpt(args.in_path, args.out_path, ignores=ignores)

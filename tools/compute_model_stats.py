import torch
import argparse
import numpy as np

import models
'''
    Usage:
        PYTHONPATH=. python tools/compute_model_stats.py
'''


def compute_param_number(model, bit_len=4):
    bits = 0
    for k, v in model.state_dict().items():
        if k.endswith('num_batches_tracked') or k.endswith(
                'running_mean') or k.endswith('running_var'):
            continue
        bits += np.prod(np.array(v.size())) * bit_len

    return bits / 1024 / 1024


def compute_flops(model, input_size, bs=1):
    conv_ops = []
    linear_ops = []

    def conv_hook(self, x, output):
        bs, in_c, in_h, in_w = x[0].size()
        out_c, out_h, out_w = output[0].size()

        assert in_c == self.in_channels

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (
            self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = out_c * (kernel_ops + bias_ops)
        flops = bs * params * out_h * out_w
        conv_ops.append(flops)

    def linear_hook(self, x, output):
        bs, in_c = x[0].size()
        out_c, = output[0].size()

        weight_ops = self.weight.nelement()
        bias_ops = self.bias.nelement()

        assert weight_ops == in_c * out_c
        assert bias_ops == out_c

        flops = bs * (weight_ops + bias_ops)
        linear_ops.append(flops)

    def register_hook(model):
        sub_models = list(model.children())
        if not sub_models:
            if isinstance(model, torch.nn.Conv2d):
                model.register_forward_hook(conv_hook)
            elif isinstance(model, torch.nn.Linear):
                model.register_forward_hook(linear_hook)
            return
        for m in sub_models:
            register_hook(m)

    register_hook(model)
    x = torch.rand(bs, 3, input_size, input_size)
    model(x)
    flops = sum(conv_ops) + sum(linear_ops)
    return flops / 1e9 / bs, len(conv_ops), len(linear_ops)


def main():
    parser = argparse.ArgumentParser(description='Compute Model FLOPs and PN')
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--feature_dim', default=256, type=int)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()

    model = models.__dict__[args.arch](feature_dim=args.feature_dim)
    flops, n_conv, n_linear = compute_flops(model, args.input_size,
                                            args.batch_size)
    bits = compute_param_number(model)

    print('[{} ({} conv, {} linear)] FLOPs: {:.2f} G, PN: {:.2f} M'.format(
        args.arch, n_conv, n_linear, flops, bits))


if __name__ == '__main__':
    main()

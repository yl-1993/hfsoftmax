import os
import argparse
import mxnet as mx
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--in-folder",
                    type=str,
                    default='./',
                    help="where stores .rec and .idx")
parser.add_argument(
    "--out-folder",
    type=str,
    default='./images/',
    help=
    "output image folder and each subfolder stores images with same identity")
parser.add_argument("--intvl",
                    type=int,
                    default=10000,
                    help="interval of displaying processing progress")
args = parser.parse_args()

if __name__ == '__main__':
    path_imgidx = os.path.join(args.in_folder, 'train.idx')
    path_imgrec = os.path.join(args.in_folder, 'train.rec')

    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    if header.flag > 0:
        imgnum, totnum = map(int, header.label)
        imgidx = range(1, imgnum)
        label2range = {}
        seq_label = range(imgnum, totnum)
        for label in seq_label:
            s = imgrec.read_idx(label)
            header, _ = mx.recordio.unpack(s)
            a, b = map(int, header.label)
            label2range[label - imgnum] = (a, b)
        assert len(label2range) == totnum - imgnum
    else:
        raise AttributeError('file with incorrect format')

    for idx in imgidx:
        s = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = int(header.label)
        img = mx.image.imdecode(img).asnumpy()
        rmin, rmax = label2range[label]
        if idx >= rmin and idx <= rmax:
            name = '{}/{}.jpg'.format(label, idx - rmin)
            path = os.path.join(args.out_folder, name)
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            plt.imsave(path, img)
        else:
            raise ValueError('something wrong')
        if idx % args.intvl == 0:
            print('handling {}/{} ...'.format(idx, len(imgidx)))

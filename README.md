# Accelerated Training for Massive Classification via Dynamic Class Selection (HF-Softmax) [![pdf](https://img.shields.io/badge/Arxiv-pdf-orange.svg?style=flat)](https://arxiv.org/abs/1801.01687)

## Paper
[Accelerated Training for Massive Classification via Dynamic Class Selection](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17244), AAAI 2018 (**Oral**)


## Training
1. Install [PyTorch](http://pytorch.org/). (Better to install the latest master from source)
2. Follow the instruction of [InsightFace](https://github.com/deepinsight/insightface) and download training data.
3. Decode the data(`.rec`) to images and generate training/validation list.

```bash
python tools/rec2img.py --in-folder xxx --out-folder yyy
```

4. Try normal training. It uses `torch.nn.DataParallel`(multi-thread) for parallel.

```bash
sh scripts/train.sh dataset_path
```

5. Try sampled training. It uses one GPU for training and default sampling number is `1000`.

```bash
python paramserver/paramserver.py
sh scripts/train_hf.sh dataset_path
```

## Distributed Training
For distributed training, there is one process on each GPU.

Some [backends](https://pytorch.org/docs/stable/distributed.html) are provided for PyTroch Distributed training.
If you want to use `nccl` as backend for distributed training,
please follow the [instructions](https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html) to install `NCCL2`.

You can test your distributed setting by executing

```bash
sh scripts/test_distributed.sh
```

When `NCCL2` is installed, you should re-compile PyTorch from source.

```bash
python setup.py clean install
```

In our case, we use `libnccl2=2.2.13-1+cuda9.0 libnccl-dev=2.2.13-1+cuda9.0` and the master of PyTorch `0.5.0a0+e31ab99`

## Hashing Forest
We use [Annoy](https://github.com/spotify/annoy) to approximate the hashing forest.
You can adjust `sample_num`, `ntrees` and `interval` to balance performance and cost.

## Parameter Sever
Parameter server is decoupled with PyTorch. A client is developed to communicate with the server.
Other platforms can integrate the parameter server via the communication API.
Currently, it only supports syncronized SGD updater.

## Evaluation

```bash
./scripts/eval.sh arch model_path dataset_path outputs
```

It uses `torch.nn.DataParallel` to extract features and saves it as `.npy`.
The features will subsequently be used to perform the verification test.

If you use distributed training, set `strict=False` during feature extraction.

Note that the bin file from InsightFace, `lfw.bin` for example, is pickled by Python2. It cannot be processed by Python 3.0+.
You can either use Python2 for evaluation or re-pickle the bin file by Python3 first.

## Feature Extraction

```bash
./scripts/extract_feat.sh prefix filelist load_path
```

## Citation
Please cite the following paper if you use this repository in your reseach.

```
@inproceedings{zhang2018accelerated,
  title     = {Accelerated Training for Massive Classification via Dynamic Class Selection},
  author    = {Xingcheng Zhang and Lei Yang and Junjie Yan and Dahua Lin},
  booktitle = {AAAI},
  year      = {2018},
}
```

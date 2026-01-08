# F<sup>4</sup>-CKM
## PyTorch implementation of [F<sup>4</sup>-CKM: Learning Channel Knowledge Map with Radio Frequency Radiance Field Rendering](https://arxiv.org/abs/2601.03601)

This repository is built upon [NeWRF](https://github.com/LuHaofan/NeWRF), thanks very much!

We would gradually upload the full-version of the implementation.

## Citation (Preprint Version)
``` bash
@article{ f4ckm,
  title={F4-CKM: Learning Channel Knowledge Map with Radio Frequency Radiance Field Rendering},
  author={Kequan Zhou and Guangyi Zhang and Hanlei Li and Yunlong Cai and Shengli Liu and Guanding Yu},
  journal={arXiv preprint arXiv:2601.03601},
  year={2026},}
```

## Clone
Clone this repository and enter the directory using the commands below:
```bash
git clone https://github.com/kqzzzz/F4CKM.git
cd F4CKM/
```

### Requirements
`Python 3.10.16` is recommended.

Install the required packages with:
```bash
pip install -r requirements.txt
```
If you're having issues with installing PyTorch compatible with your CUDA version, we strongly recommend this [documentation page](https://pytorch.org/get-started/previous-versions/).

## Usage
+ simulator/ contains the code for dataset generation. To generate the datasets, run `generateDataset.m` in MATLAB (R2024a or later recommended).
+ Use `mat2pkl.py` script in the simulator/ to convert the dataset from .mat to the .pkl.
+ Example of training the F<sep>4<sep>-CKM model:
```bash
bash train.sh
```
+ Example of testing the F<sep>4<sep>-CKM model:
```bash
bash test.sh
```

## Datasets and Pretrained Models
Datasets and pretrained models are available [Here](https://pan.baidu.com/s/1jpfs9kOIqpf8FGgvwRDLfQ?pwd=173e)

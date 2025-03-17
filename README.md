# EasyTSF

This project was upgraded from KAN4TSF to EasyTSF (**E**xperiment **as**sistant for **y**our **T**ime-**S**eries **F**orecasting).

🚩 **News** (2025.02) We add MMK, a Multi-layer Mixture-of-KAN model for MTSF.

🚩 **News** (2024.11) KAN4TSF -> EasyTSF, we will support more time series forecasting models.

🚩 **News** (2024.09) Model Zoo: RMoK, NLinear, DLinear, RLinear, PatchTST, iTransformer, STID, TimeLLM

🚩 **News** (2024.09) [Introduction and Reproduction (in Chinese)](https://mp.weixin.qq.com/s/bSwAbKBxON7FPebAiqltWg)

## Usage

### Environment
Step by Step with Conda:
```shell
conda create -n kan4tsf python=3.10
conda activate kan4tsf
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
python -m pip install lightning
```

or you can just:
```shell
pip install -r requirements.txt
```

### Code and Data
ETTh1 and ETTm1 can be downloaded within this project, and other datasets can be downloaded from [Baidu Drive](https://pan.baidu.com/s/18NKge4dsMIuGQFom7n2S2w?pwd=zumh) or 
[Google Drive](https://drive.google.com/file/d/17JYLHDPIdLv9haLiDF9G_eGewhmMhGbq/view?usp=sharing).

### Running
```shell
python train.py -c config/reproduce_conf/RMoK/ETTh1_96for96.py
```

```shell
 python ray_tune.py --config config/reproduce_conf/MMK/MMK_ETTh1_96for96.py --param_space config/reproduce_conf/MMK/MMK_search_space.py 
```

## Cite
If you find this repo useful, please cite our paper:
```
@inproceedings{han2023are,
  title={KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting?},
  author={Xiao Han, Xinfeng Zhang, Yiling Wu, Zhenduo Zhang and Zhe Wu},
  booktitle={arXiv},
  year={2024},
}
```

# Reproducibility Report: Auto White-Balance Correction for Mixed-Illuminant Scenes

## Replicating our results

This repository is built in PyTorch 1.8.1 and tested on Ubuntu 16.04 environment (Python3.7, CUDA10.2, cuDNN7.6). Follow these intructions

```
conda create -n pytorch181 python=3.7
conda activate pytorch181
conda install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips
```
### Dataset

We download the dataset provided using the original links.

Train: We train our model on the Rendered WB dataset. Links for [input](https://ln2.sync.com/dl/21ce83f60/v36jwspj-e4mw2vtb-s6ifkgmv-jzc8mvya/view/default/9967673690008) and [target](https://ln2.sync.com/dl/1f607c380/ypyw5z4p-q765pviu-rc8tzi2n-4pyyep8h/view/default/9967673010008)

Val: We cross validate our results on the Cube Dataset. Links for [input](https://ln2.sync.com/dl/fb7ef1c90/v2pbm9jm-wq5p8z6b-rrg2g4ez-r4c5wixg/view/default/9967668600008) and [target](https://ln2.sync.com/dl/fb7ef1c90/v2pbm9jm-wq5p8z6b-rrg2g4ez-r4c5wixg/view/default/9967668600008)

Test: We test our model on the mixed-illuminant test set. We already add the test set in this repository.

### Training

To start training, you should first download the Rendered WB dataset, which includes 65K sRGB images rendered with different color temperatures. Each image in this dataset has the corresponding ground-truth sRGB image that was rendered with an accurate white-balance correction. From this dataset, we selected 9,200 training images that were rendered with the "camera standard" photofinishing and with the following white-balance settings: `tungsten` (or `incandescent`), `fluorescent`, `daylight`, `cloudy`, and `shade`. To get this set, you need to only use images ends with the following parts: `_T_CS.png`, `_F_CS.png`, `_D_CS.png`, `_C_CS.png`, `_S_CS.png` and their associated ground-truth image (that ends with `_G_AS.png`).

Use `process_dataset.py` to
- Copy all training input images to `../mixedill/train/input` and copy all ground truth images to `../mixedill/train/target`.
- Copy all validation input images to `../cube/input` and copy all ground truth images to `../cube/target`.


To train the model with `D` `S` `T` version, run

```
python train.py --training training_3pairs.yml
```

To train the model with `D` `S` `T` `F` `C` version, run

```
python train.py --training training_5pairs.yml
```


### Testing

To test the model with `D` `S` `T` version, run

```
python test.py --result_dir ./results/Gridnet1 --weights ./checkpoints/WB/models/Gridnet1/model_best.pth --wb_settings D S T
```

To test the model with `D` `S` `T` `F` `C` version, run

```
python test.py --result_dir ./results/Gridnet2 --weights ./checkpoints/WB/models/Gridnet2/model_best.pth --wb_settings D S T F C
```


### Evaluation

To evaluate the model with `D` `S` `T` version, run

```
python evaluate.py --prd_dir ./results/Gridnet1
```

To evaluate the model with `D` `S` `T` `F` `C` version, run

```
python evaluate.py --prd_dir ./results/Gridnet2
```




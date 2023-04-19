# DLProject

## Replicating our results

### Training

To start training, you should first download the Rendered WB dataset, which includes 65K sRGB images rendered with different color temperatures. Each image in this dataset has the corresponding ground-truth sRGB image that was rendered with an accurate white-balance correction. From this dataset, we selected 9,200 training images that were rendered with the "camera standard" photofinishing and with the following white-balance settings: `tungsten` (or `incandescent`), `fluorescent`, `daylight`, `cloudy`, and `shade`. To get this set, you need to only use images ends with the following parts: `_T_CS.png`, `_F_CS.png`, `_D_CS.png`, `_C_CS.png`, `_S_CS.png` and their associated ground-truth image (that ends with `_G_AS.png`).

Copy all training input images to `./data/images` and copy all ground truth images to `./data/target`.


To train the model with `D` `S` `T` version, run

```
python train.py --training training_3pairs.yml
```

To train the model with `D` `S` `T` `F` `C` version, run

```
python train.py --training training_5pairs.yml
```


### Testing

```
python test.py
```

### Evaluation

```
python eval.py
```


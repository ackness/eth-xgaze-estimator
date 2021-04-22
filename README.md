# eth-xgaze-estimator

Some scripts for me to train and test easily on
[Competition: ETH-XGaze](https://competitions.codalab.org/competitions/28930).

## Train

```bash
#python main.pys
accelerate config  # This will create a config file on your server
accelerate launch ./main.py --cfg=configs/resnet50-xgaze-within-yml
```

## Test

```bash
python main.py
```

Requires

```
torch
timm
h5py
opencv-python
tqdm
tensorboard
accelerate
```


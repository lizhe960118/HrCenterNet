# HrNet backbone in CenterNet
**This code is use to train and evaluate the hornet backbone in CenterNet. For more technical details, please refer to the [arXiv paper](https://arxiv.org/abs/1904.08189).**

**CenterNet is an one-stage detector which gets trained from scratch. On the MS-COCO dataset, CenterNet achieves an AP of 47.0%, which surpasses all known one-stage detectors, and even gets very close to the top-performance two-stage detectors.**

## Architecture

![Network_Structure](https://github.com/Duankaiwen/CenterNet/blob/master/Network_Structure.jpg)

## Preparation
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create --name CenterNet --file conda_packagelist.txt
```

After you create the environment, activate it.
```
source activate CenterNet
```

## Compiling Corner Pooling Layers
```
cd <CenterNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

## Compiling NMS
```
cd <CenterNet dir>/external
make
```

## Installing MS COCO APIs
```
cd <CenterNet dir>/data/coco/PythonAPI
make
```

## Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CenterNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CenterNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training

To train HrCenterNet:
```
python train.py HRNet
```
We provide the configuration file (`HRNet.json`) and the model file (`HRNet-104.py`) for CenterNet in this repo. 

To continue training:

1. modify the `pretrained` in `HRNet.json`
2. `python train.py HRNet --iter 10000`

To train DLANet:
```
python train.py DLANet
```
## Evaluation

To test HRNet:

```
python test.py HRNet --testiter 480000 --split validation
```

To test DLANet:
```
python test.py DLANet --testiter <iter> --split <split>
```
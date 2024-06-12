# patchcore
Implementation of anomaly detection method PatchCore

## Installation
### Local
* torch==1.13
* torchvision==0.14
* opencv-python
* scikit-learn
* torchmetrics==0.9.0
* omegaconf
* tqdm

### Docker
#### GPU
```bash
source setting/init_docker.sh
./build
```

#### CPU
```bash
source setting/init_docker.sh
./build.cpu
```

#### Login container
##### GPU
```bash
./login
```

##### CPU
```bash
./login.cpu
```

## Sample dataset
sample dataset here

```
src/data/images/wood/
  train/
  val/
  test/
```

## Training
```
python train.py cfg/train/wide_resnet50_wood.yaml
```

[train config file document](src/cfg/train/README.md)

## Test
```
python test.py cfg/test/wide_resnet50_wood.yaml
```

[test config file document](src/cfg/test/README.md)

## Sample dataset 

* backborn Wide-ResNet50
* 

|| Precision | Recall | F1-score |
|---|---|---|---|
| Normal | 0.8889 | 1.0000 | 0.9412 |
| Abnormal | 1.0000 | 0.8750 | 0.9333 |
| macro-mean | 0.9444 | 0.9375 | 0.9373 |

<img src="data/sample01.png" width="800">

<img src="data/sample02.png" width="800">

<img src="data/sample03.png" width="800">

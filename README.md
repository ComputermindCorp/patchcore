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

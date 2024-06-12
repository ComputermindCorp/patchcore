# patchcore
Implementation of anomaly detection method PatchCore

## Installation
### docker
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

#### login container
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

## Test
```
python test.py cfg/test/wide_resnet50_wood.yaml
```

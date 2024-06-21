# Train config
Train script（train.py） configuration file.Format is yaml.<br>
学習スクリプトの設定ファイル

# coreset_sampling_ratio
Float between 0 and 1. Coreset sampling ratio to subsample.<br>
コアサンプリング比率(0以上, 1.0以下)
抽出した特徴ベクトルのサンプル比率
（0.1の場合、全特徴ベクトルの10%をサンプリング）

# num_neighbors
Int, Number of nearest neighbors. Defaults to 9.<br>
近傍法の抽出数

# input_size
Tuple, Size of input image size. <br>
入力画像サイズのリスト

# backborn_id
String, backborn id.<br>
Backbornの指定
[resnet18 | resnet50 | wide_resnet50]

## device
String or None. Device id to select. If None, automatic selection.<br>
デバイスを指定。Noneの場合は自動選択。
[cpu | cuda | None]

## batch_size
Int. Size of thre batches of data.<br>
バッチサイズ.

## train
Train Data Settings<br>
Trainデータの設定

### data_paths
List of image directory paths.<br>
For example, if the following is set up, all images under the OK1 and OK2 directories will be used as Train data.<br>

画像ディレクトリパスのリスト。
例えば以下のように設定した場合は、OK1とOK2ディレクトリ下のすべての画像をTrainデータとして使用する。

```
train:
  data_paths:
    - ./data/images/wood/train/OK1
    - ./data/images/wood/train/OK2
```

## val
Validation Data Settings<br>
Validationデータの設定

## data_paths, labels
List of image paths and labels<br>
A label of 0 means normal and 1 means abnormal.<br>

For the following
./data/images/wood/val/OK -> label 0 (normal)
./data/images/wood/val/NG -> label 1 (abnormal)


画像パスとラベルのリスト
ラベルは0が正常, 1が異常となる。

以下の場合は
./data/images/wood/val/OK -> ラベル0 (正常)
./data/images/wood/val/NG -> ラベル1 (異常)
となる

```yaml
test_data_paths:
 - ./data/images/wood/val/OK
 - ./data/images/wood/val/NG

labels: [0, 1]
```

## test
Test Data Settings<br>
Testデータの設定

## data_paths, labels
Same as train.<br>
Validationと同様

## save_weights_root_path
String, path where to save the weights file.<br>
重みファイルの保存先ディレクトリパス

## auto_save_weights_path, save_weights_path_suffix, save_weights_filename
Save weights file name setting.
重みファイルの保存ファイル名の設定。

### If "auto_save_weights_path" is True. auto_save_weights_pathがTrueの場合
The save file name is automatically generated with the following contents.<br>
保存ファイル名は以下内容で自動生成される。
```
{backborn_id}_size{input_size}_param_{coreset_sampling_ratio}_{num_neighbors}.pth
```
ex
```
resnet50_size224_param_0.1_9.pth
```
If "save_weights_path_suffix" is set, the string specified in "save_weights_path_suffix" is added to the file name as a suffix.<br>
For example, if "save_weights_path_suffix" is set to "test", the save file name will be<br>

save_weights_path_suffixが設定されている場合、ファイル名にsave_weights_path_suffixで指定した文字列がsuffixとして付与される。
例えばsave_weights_path_suffixが"test"と設定されている場合は保存ファイル名は以下となる。

```
resnet50_size224_param_0.1_9_test.pth
```

### If "auto_save_weights_path" is False. auto_save_weights_pathがFalseの場合
Save the file with the file name set in "save_weights_filename".<br>
save_weights_filenameで設定したファイル名で保存する。


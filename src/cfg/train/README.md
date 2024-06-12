# Train config
学習スクリプトの設定ファイル

※値を何も記載しないとNoneとなる

# coreset_sampling_ratio
コアサンプリング比率(0以上, 1.0以下)
抽出した特徴ベクトルのサンプル比率
（0.1の場合、全特徴ベクトルの10%をサンプリング）

# num_neighbors
近傍法の抽出数(1〜)

# input_size
入力画像サイズのリスト

# backborn_id
Backbornの指定
[resnet18 | resnet50 | wide_resnet50]

## device
デバイスを指定
[cpu | cuda | None]

## batch_size
バッチサイズ（1〜)

## train
Trainデータの設定

### data_pathes
画像ルートパスのリスト。

例えば以下のように設定した場合は、OK1とOK2ディレクトリ下のすべての画像をTrainデータとして使用する。

```
train:
  data_pathes:
    - ./data/images/wood/train/OK1
    - ./data/images/wood/train/OK2
```

## val
Validationデータの設定

## data_pathes, labels
画像パスとラベルのリスト
ラベルは0が正常, 1が異常となる。

以下の場合は
./data/images/wood/val/OK -> ラベル0 (正常)
./data/images/wood/val/NG -> ラベル1 (異常)
となる

```yaml
test_data_pathes:
 - ./data/images/wood/val/OK
 - ./data/images/wood/val/NG

labels: [0, 1]
```

## test
Testデータの設定

## data_pathes, labels
Validationと同様


## save_weights_root_path
重みファイルの保存先ディレクトリパス
（存在しない場合は自動生成する）

## auto_save_weights_path, save_weights_path_suffix, save_weights_filename
重みファイルの保存ファイル名の設定。

### auto_save_weights_pathがTrueの場合
保存ファイル名は以下内容で自動生成される。
```
{backborn_id}_size{input_size}_param_{coreset_sampling_ratio}_{num_neighbors}.pth
```
ex
```
resnet50_size224_param_0.1_9.pth
```

save_weights_path_suffixが設定されている場合、ファイル名にsave_weights_path_suffixで指定した文字列がsuffixとして付与される。

例えばsave_weights_path_suffixが"test"と設定されている場合は保存ファイル名は以下となる。

```
resnet50_size224_param_0.1_9_test.pth
```


### auto_save_weights_pathがFalseの場合
save_weights_filenameで設定したファイル名で保存する。


# Test config
Test script configuration file.Format is yaml.<br>
テストスクリプトの設定ファイル。フォーマットはyamlとなる。

## device
String or None. Device id to select.
デバイスを指定。Noneの場合は自動選択となる。
[cpu | cuda | None]

## weights_path
String. path to the trained model file.<br>
学習済モデルファイル（.pth）のパス

## th
Float between 0 and 1. The value to threshold. Defaults to 0.5.<br>
推論時の判定しきい値（0.0〜1.0）。デフォルト値は0.5

## test_data_pathes, labels
画像パスとラベルのリスト
ラベルは0が正常, 1が異常となる。

以下の場合は
./data/images/wood/test/OK -> ラベル0 (正常)
./data/images/wood/test/NG -> ラベル1 (異常)
となる

```yaml
test_data_pathes:
 - ./data/images/wood/test/OK
 - ./data/images/wood/test/NG

labels: [0, 1]
```

## output_root_path
String. Path where to save the test result file(csv, image). <br>
結果csvファイルや画像ファイルの保存先ディレクトリパス

## heatmap
Result visualization image settings.<br>
結果可視化画像の設定

### ng_dir
Boolean. Save incoreect images in a separate directory as well.<br>
結果不正解の可視化画像を別フォルダ（NGフォルダ）にも保存する

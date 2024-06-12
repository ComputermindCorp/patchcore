# Test config
テストスクリプトの設定ファイル

※値を何も記載しないとNoneとなる

## device
デバイスを指定
[cpu | cuda | None]

## weights_path
重みデータ（.pth）のパス

## th
推論時の判定しきい値（0.0〜1.0）
通常は0.5を指定

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
結果csvファイルや画像ファイルの保存先ディレクトリパス
（存在しない場合は自動生成する）


## heatmap
結果可視化画像の設定
### ng_dir
結果不正解の可視化画像を別フォルダ（NGフォルダ）にも保存する



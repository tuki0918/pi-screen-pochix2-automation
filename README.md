# pi-screen-pochix2-automation

Raspberry Pi Camera を利用してリアルタイムでスマホ画面を
予め学習したCNNモデルに識別させ、特定の条件下のみ画面を自動でポチポチします

#### ハードウェア

+ Raspberry Pi
+ Raspberry Pi Camera Module V2
+ リレータッチボード（ドライバ有り）

#### 環境

+ picamera
+ cv2
+ keras
+ tensorflow

-----

#### モデルの作成

事前に「resources/datasets」以下に識別したいフォルダを作成し、画像を用意しておく

```
python3 resources/train.py
```

#### 自動ポチポチモード

事前に学習モデル「resources/model-screen-3ch.h5」が必要

```
python3 pochix2.py
```

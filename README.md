Medical Deep Learning Homework 1: SPECT Image Classification
============================================================
# Environment
```
conda create --name [env] python=3.9
conda activate [env]

pip install -r requirements.txt
```

# Data
```mkdir data```
cp source images folder [DICOM] to ```data/```

# Record
## 20221029-20221030003: VGG
- 訓練一直有些問題
    - Logits (output) 輸出一直為 NaN (clipgrad 好像可以解決)
    - Loss 很怪
- 決定先用 resnet
## 20221030004 - ...
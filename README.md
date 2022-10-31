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
## 20221030004 - 20221030006: ResNet
- 可以用
## 20221031001
- 重寫 vgg: 寫法跟 resnet 類似
- lr 重設

## ResNet50: Summary ver-1
- inference 每次結果不同？
    - load constant model file
    - model eval check
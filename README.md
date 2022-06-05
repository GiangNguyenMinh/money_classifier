![](https://vaytienaz.com/wp-content/uploads/2020/10/me%CC%A3%CC%82nh-gia%CC%81-tie%CC%82%CC%80n-vie%CC%A3%CC%82t-nam.jpg)
# VietNamese Money Classifier

## Install 
### Environment
#### op1: install serializion
[anaconda-environment](https://www.anaconda.com/products/individual) with python>=3.8


[torch](https://pytorch.org/)


Activate anaconda-environment then


Install opencv: 
```bash
$ pip install opencv-python
```


Install sklearn: 
```bash
pip3 install -U scikit-learn
```
#### op2: install requirement.txt
```bash
$ pip install -r requirement.txt
```
### Setup 
```bash
$ git clone https://github.com/GiangNguyenMinh/money_classifier.git
$ cd money_classifier
```

## Create data
```bash
$ python MakeData.py --value 0000
                             10000
                             50000
                             200000
                             500000
```
## Training
```bash
$ python train.py --use-weights --lr 0.0001 --batch-size 32 --n-epochs 100 --n-worker 16
```

## Inference
* Inference with pytorch model
```bash
$ python predict.py --thread-hold 0.6
```
* Inference with onnx model in onnxruntime
```bash
$ python export.py --onnx
$ python predict_onnx.py
```

## Train colab
#### From scratch
Colab code [click here](https://colab.research.google.com/drive/15aTHA5HJFVxIv1HLv3zdQu1sqbvCsa3b?usp=sharing)
#### With weight pretrain
Colab code [click here](https://colab.research.google.com/drive/1p3PH9AuupS4HRhVIDpc_hGSiZg6L3T6K?usp=sharing)
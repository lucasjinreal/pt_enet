# ENet 

一个十分快速的语义分割网络，在GTX1080
TI上速度可以达到22fps。其训练之后的权重也非常小，仅有20M左右。这用于实际的生产环境是非常好的网络。它在cityscapes上meaniOU可以达到60
（最佳的时候），目前这个实现已经添加了cfr后续处理，效果非常不错。

<div align=center><img src="https://s1.ax1x.com/2018/12/07/F1OKLF.gif"/></div>

## 运行


该实现依赖与alfred-py，使用前请安装：

```
sudo pip3 install alfred-py
```

运行非常简单：

```
python3 demo.py
```

要训练自己的数据集，准备好cityscapes数据，同时设置相应的路径。

```
python3 train.py
```

即开始了训练，该实现同时支持CPU和GPU。

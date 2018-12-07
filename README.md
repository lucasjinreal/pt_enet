# Enet

This is a realtime segmentation net with almost 22 fps on GTX1080 ti, and the model size is very
small with only 28M. 

This repo contains the inference demo with already trained weights for production. the weights
trained on cityscapes and camvid. the result like below:

<div align=center><img src="https://s1.ax1x.com/2018/12/07/F1OKLF.gif"/></div>


## Runing

To run the demo, simply:

```
python3 demo.py
```

this demo requires `alfred-py`, you should install first with:

```
sudo pip3 install alfred-py
```





## Training



If you want training with your own dataset or fine-tune on cityscapes, you can get the full version codes from http://strangeai.pro .  contact me to get the full codes, you can find me on that website.
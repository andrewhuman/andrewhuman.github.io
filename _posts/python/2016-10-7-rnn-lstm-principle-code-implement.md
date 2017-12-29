---
published: true
layout: post
title: lstm内部原理详解与代码实现(numpy)
description: rnn
---

## 前言
lstm全名Long Short Term Memory，即长短记忆网络，它是对平凡rnn模型的一种改进，可在一定程度上避免平凡rnn模型层数过高时网络的梯度弥散问题,并有更好的拟合效果.在tensorflow或其他框架中，lstm都是封装好的，只对外暴露输入输出等，不过如果你想弄明白网络发生了什么，以及为什么这么设计，还是有必要把封装撕开，看一看内部的实现的.以下大部分图片都摘自[CS231N课程](http://cs231n.stanford.edu/slides/2016/winter1516_lecture10.pdf).
## 警告
在开始之前，我们先看一张图，但是放在这里是建议大家不要看这张图及类似的变种:
![](../images/rnn_lstm_pipe_donot_look.png)
这张图也是LSTM的流程图，网络上也有很多这种弯弯曲曲的图片的变种，如果你是刚开始接触LSTM并且仔细看了，我只能说不知道你看没看懂反正我到现在也没看太懂,而且第一次看浪费了2个小时......，因为它画的实在有点复杂了.
## 1. LSTM原理
### 1.1 RNN和LSTM比较
平凡rnn和lstm的结构如下图：
![](../images/rnn_lstm_simple_bijiao.png)
#### . 平凡RNN的计算公式:
![](../images/rnn_simple_rnn_alghrithem.png)
如果设
$ h_\t $




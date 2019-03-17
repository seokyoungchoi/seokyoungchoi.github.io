---
layout: post
title: Introduction to Deep Learning @ corsera, Andrew Ng
use_math: true
---

오늘부터 다음의 링크에서 볼 수 있는 coursera에 업데이트 된 Andrew Ng 교수님의 딥러닝 강의를 정리해서 올려보려고 합니다.

## What you will learn from this course

첫번째 주차는 딥러닝에 대한 전반적인 이해를 돕는 내용을 설명하고 있습니다.
이 강의에서 우리가 알아갈 수 있는 것은 다음과 같다고 합니다.

	1. Neural Networks and Deep Learning
	2. Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
	3. Structuring your ML project
	4. Convolutional Neural Networks
  5. Natural Language Processing: Building sequence models
		a. Sequence models includes RNN, LSTM, …


## What is a Neural Network?

![_config.yml]({{seokyoungchoi.github.io}}/images/Week1_1.png)

위에서 보이는 model은 우리가 흔히 아는 Linear regression과는 조금 다릅니다.
가격은 무조건 0보다 높다는 사실을 이용해, negative값을 가질 수 있는 부분을 0으로 대체하였기 때문입니다.

이렇게 집의 크기(x)를 이용해 집의 가격(y)를 예측하는 것이 어떤 뉴런(neuron)을 통해서 모델로 구현된 것을 Neural Network라고 합니다.
더 정확히 말하자면 위의 그림의 예시는 activation model 중 ReLU(Rectified Linear Units)라고 불리우는 모델입니다.

위와 같이 하나의 뉴런으로 이뤄진 경우는 가장 작은 형태의 neural network라고 할 수 있고, 보통은 이런 single neuron을 'stacking together'함으로써 더 큰 neural network를 만들게 됩니다.


![_config.yml]({{ site.baseurl }}/images/Week1_2.png)

더 stacking된 neural network의 예시를 봅시다.
위에서 *Input Neuron*은 $x_1$, $x_2$, $x_3$, $x_4$, *Output Neuron*은 $y$가 있습니다.

여기서 짚고 넘어갈 점은, Neural Network에서 우리가 수동으로 해야하는 일은 Examples를, 즉 input neuron과 output neuron을 포함한 데이터 셋을 입력해주는 것 뿐이라는 것입니다. 가운데에 있는 숨겨진 유닛들은 Neural Network가 알아서 판단하고 완성하게 놔둡니다. 
놀라운 것은 충분한 데이터가 제공되었다는 가정 하에, Neural Network가 찾아내는 $x$에서 $y$로의 map function이 정말 정확하다는 것입니다.
이는 Neural Network가 Supervised Learning에서 강력한 도구가 될 수 있는 이유이기도 합니다.

## Supervised Learning with Neural Network

![_config.yml]({{ site.baseurl }}/images/Week1_3.png)

Supervised Learning에서 어떤 input과 output이 입력되느냐에 따라 사용되는 Neural Networks의 종류가 다릅니다.

먼저, 제일 처음 봤던 House price 예시에서 쓰이는 것은 *Standard Nerual Networks*입니다.
*CNN(Convolution on neural networks)*은 image data에서 많이 쓰입니다.
*RNN(Recurrent nerual networks)*는 sequence data에서 쓰이는데, sequence data의 대표적인 예로는 음성자료와 번역자료가 있겠습니다.

Standard CNN과 RNN의 구조를 도식화하면 다음과 같습니다.
![_config.yml]({{ site.baseurl }}/images/Week1_4.png)

더 자세한 것은 앞으로의 강의에서 설명될 예정이라고 합니다.

음성 data와 같이 unstructured data는 컴퓨터가 해석하기에 어려웠다는 점을 극복했다는 점이 neural network의 또 다른 공헌입니다. 
사실 이 부분이 언론에서 딥러닝이 각광받은 가장 큰 이유이지만, 사실은 structured data에서의 경제적인 공헌도 못지 않게 크다는 점을 잊지 말아야합니다.


## Why is Deep Learning taking off?
![_config.yml]({{ site.baseurl }}/images/Week1_5.png)

딥러닝의 이론적인 개념은 예전부터 정립되어 있었지만 갑자기 각광을 받기 시작한 것은 활용할 수 있는 데이터의 크기가 점점 커지면서라고 보는 것이 맞습니다.
빅데이터 시대라는 말이 같이 맞물리면서 각광을 받게 되었다고 보면 쉽게 이해할 수 있는 현상이라고 합니다.
위의 그림에서 볼 수 있듯이, 데이터의 크기가 커질수록 performance(accuracy 등)의 극대화는 Neural Network가 커짐으로써 달성됩니다.

일반적으로 사용되는 머신러닝의 sigmoid function의 문제점은 gradient가 0으로 가는 sigmoid function의 경우 learning의 속도가 매우 느려진다는 것입니다. 이 sigmoid function 자체를 수정함으로써 gradient descent를 좀 더 빠르게 작동시킬 수는 있지만 근본적으로 Neural Network보다는 느립니다.

이런 fast computation이 중요한 이유는, 빠르게 결과를 돌림으로써 시행착오를 통해 모델을 더 발전시키는 속도를 훨씬 빠르게 할 수 있기 때문입니다. 

이렇듯, 다룰 수 있는 데이터가 점점 더 커지는 지금의 환경에서 딥러닝은 중요한 데이터 해석툴이라고 볼 수 있겠습니다. 








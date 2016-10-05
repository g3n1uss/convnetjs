
# ConvNetJS

This project is a redesigned, improved (in my opinion) version of the [original project](https://github.com/karpathy/convnetjs). For now it contains only a single example - the MNIST classifier.

## Introduction

A very nice introduction to convolutional networks can be found [here](http://cs231n.github.io/convolutional-networks/) and [here](https://en.wikipedia.org/wiki/Convolutional_neural_network). To summarize convolutional neural networks are used to solve the [overfitting problem](https://en.wikipedia.org/wiki/Overfitting#Machine_learning), when one has too many weights to fit the curve. A single fully-connected neuron (used in ordinary NN) taking as input small colorful images, let’s say 32x32 pixels, would require 32x32x3=3072 weights. It seems ok, but usually we want several such neurons, also the input images might be bigger, which would drastically increase the number of weights (parameters), which definitely would lead to overfitting. In convolutional networks the neurons in a layer are connected only to a small region of the previous layer, which eliminates overfitting.

Let's compare the regular deep nueral network and the convolutional one (images are taken from [here](http://cs231n.github.io/convolutional-networks/)) 

  <img src="http://cs231n.github.io/assets/nn1/neural_net2.jpeg" width="40%" />
  <img src="http://cs231n.github.io/assets/cnn/cnn.jpeg" width="48%" style="border-left: 1px solid black;"/>
  
  Left: A regular 3-layer Neural Network. Right: A ConvNet arranges its neurons in three dimensions (width, height, depth), as visualized in one of the layers. Every layer of a ConvNet transforms the 3D input volume to a 3D output volume of neuron activations. In this example, the red input layer holds the image, so its width and height would be the dimensions of the image, and the depth would be 3 (Red, Green, Blue channels).

Other resources:
* [Original paper](http://www.mitpressjournals.org/doi/abs/10.1162/neco.1989.1.4.541) by LeCun et al - application of convolutional NN to recognition of handwritten zip codes https://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf  
* Applying of convolutional neural networks to classify colorful images https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf 
* Introducing visualization of each level to see how and why convolutional networks introduced in the previous paper work http://www.matthewzeiler.com/pubs/arxive2013/arxive2013.pdf 

Recent developments:
* [Conditional image generation](https://arxiv.org/abs/1606.05328 )
* [Text-to-speech generation](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

## Example Code

Here's a minimum example of defining a **2-layer neural network** and training
it on a single data point:

```javascript
// species a 2-layer neural network with one hidden layer of 20 neurons
var layer_defs = [];
// input layer declares size of input. here: 2-D data
// ConvNetJS works on 3-Dimensional volumes (sx, sy, depth), but if you're not dealing with images
// then the first two dimensions (sx, sy) will always be kept at size 1
layer_defs.push({type:'input', out_sx:1, out_sy:1, out_depth:2});
// declare 20 neurons, followed by ReLU (rectified linear unit non-linearity)
layer_defs.push({type:'fc', num_neurons:20, activation:'relu'}); 
// declare the linear classifier on top of the previous hidden layer
layer_defs.push({type:'softmax', num_classes:10});

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

// forward a random data point through the network
var x = new convnetjs.Vol([0.3, -0.5]);
var prob = net.forward(x); 

// prob is a Vol. Vols have a field .w that stores the raw data, and .dw that stores gradients
console.log('probability that x is class 0: ' + prob.w[0]); // prints 0.50101

var trainer = new convnetjs.SGDTrainer(net, {learning_rate:0.01, l2_decay:0.001});
trainer.train(x, 0); // train the network, specifying that x is class zero

var prob2 = net.forward(x);
console.log('probability that x is class 0: ' + prob2.w[0]);
// now prints 0.50374, slightly higher than previous 0.50101: the networks
// weights have been adjusted by the Trainer to give a higher probability to
// the class we trained the network with (zero)
```

and here is a small **Convolutional Neural Network** if you wish to predict on images:

```javascript
var layer_defs = [];
layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3}); // declare size of input
// output Vol is of size 32x32x3 here
layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
// the layer will perform convolution with 16 kernels, each of size 5x5.
// the input will be padded with 2 pixels on all sides to make the output Vol of the same size
// output Vol will thus be 32x32x16 at this point
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 16x16x16 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 16x16x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
// output Vol is of size 8x8x20 here
layer_defs.push({type:'pool', sx:2, stride:2});
// output Vol is of size 4x4x20 here
layer_defs.push({type:'softmax', num_classes:10});
// output Vol is of size 1x1x10 here

net = new convnetjs.Net();
net.makeLayers(layer_defs);

// helpful utility for converting images into Vols is included
var x = convnetjs.img_to_vol(document.getElementById('#some_image'))
var output_probabilities_vol = net.forward(x)
```

## Getting Started

To run the demo just open `mnist.html`, it will be launched automatically. It works on Firefox, however, it will not work on Chrome. Due to the following problem:
> In some cases, if you are trying to load images or other data dynamically, you might run into issues with running local html files and cross-origin policies. For example, the MNIST or CIFAR demos will not work locally because they load images dynamically. A simple work-around is to run a dummy local web server in your folder. On Ubuntu for example, cd into it, start up one: python -m SimpleHTTPServer, and then navigate to the local address that python prints for you in your browser to see your files in that folder.

So to run the demo in Chome:
1. navigate to convnetjs folder `cd convnetjs/`
2. run the python server in that directory `python -m SimpleHTTPServer` 
3. open Chrome and go to `http://localhost:8000/`


## License
MIT

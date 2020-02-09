
## digit_NN

This repository is a personal project to help me understand the fundamentals of neural networks and demonstrate my ability to write beautiful code. With this code I have produced a neural network trained on data from <u>The MNIST Database of handwritten digits</u> (http://yann.lecun.com/exdb/mnist/) able to identify handwritten digits with a 95.86% success rate during validation.

&nbsp;

Here are the important scripts:

**NN.py**, the fundamental script in this repository, defines the NN class used to create neural network objects. These objects are given a learning rate, momentum constant, and a numpy array shape (eg [784,300,100,10]; 784 input nodes, 10 output nodes, and 2 hidden layers with 300 and 100 nodes respectively).

**network_trainer.py** is used to iterate through the training data (kept in the data directory) and call the .train() method of a neural network object. This script is currently set up to create a new neural network and train it once against all 60,000 training examples.

**network_validator.py** is used to load a neural network object and run it against the validation data (also kept in the data directory) to determine the error rate of the network. This script is currently set up to validate *values_4thFullRun_alpha1e-1_4Levels_2020-02-08_10_05.pckl* with the 10,000 validation examples, which yields a 4.14% error rate.

**nn_tools.py** contains supporting tools for network_trainer.py and network_validator.py

&nbsp;

Dependencies of this repository are quite standard (os, numpy, matplotlib.pyplot, datetime, pickle) except for idx2numpy, which is easily installed with *pip install idx2numpy* (https://pypi.org/project/idx2numpy/).

&nbsp;

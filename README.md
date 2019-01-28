# Sample implementations of Privacy Preserving DL from Literature
## Getting started with PyTorch
[mnist.py](https://github.com/debjyoti0891/privacypreDL/blob/master/mnist.py) uses a CNN for classification of the MNIST dataset. It also includes details of saving network state to file and reloading.
The code is based on the following tutorials.
* [CNN in PyTorch](https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/)
* [CNN in PyTorch](https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/)
* [Computing the output dimensions of network layers](https://discuss.pytorch.org/t/explaination-of-conv2d/8082/4)
* [MNIST in PyTorch](https://nextjournal.com/gkoehler/pytorch-mnist)

## [Privacy Preserving Deep Learning](https://dl.acm.org/citation.cfm?id=2813687) 
[dssgd.py](https://github.com/debjyoti0891/privacypreDL/blob/master/dsgd.py)
 implements Distributed Selective Stochastic Gradient Descent implemented on the MNIST dataset. 

Currently supported parameters are as follows :-
1. *trainers* : number of participants in DSSGD
2. *partition_ratio* : part of data sets available to individual participant.
3. *n_parts* : number of times global update happens per epoch
4. *theta* : fraction to total parameters updated to global state
5. *n_epochs* : number of epochs used for training
The usual SGD parameters are also present : *batch_size_train, batch_size_test, learning_rate, momentum* and  
*log_interval*. 

Create a folder *results* in the same directory  to store the trained models and optimizers. 
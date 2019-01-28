# Sample implementations of Privacy Preserving DL from Literature


dssgd.py implements Distributed Selective Stochastic Gradient Descent based on the [Privacy Preserving Deep Learning](https://dl.acm.org/citation.cfm?id=2813687) on the MNIST dataset. 

Currently supported parameters are as follows :-
1. *trainers* : number of participants in DSSGD
2. *partition_ratio* : part of data sets available to individual participant.
3. *n_parts* : number of times global update happens per epoch
4. *theta* : fraction to total parameters updated to global state
5. *n_epochs* : number of epochs used for training
The usual SGD parameters are also present : *batch_size_train, batch_size_test, learning_rate, momentum* and  
*log_interval*. 

Create a folder *results* in the same directory  to store the trained models and optimizers. 
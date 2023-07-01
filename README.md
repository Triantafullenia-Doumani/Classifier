# Classifier

Classification neural network based on multilayer perceptron with three hidden layers in C.


Compile dataset:
 `gcc dataset1.c -o dataset -lm`

Compile classifier:
`gcc classifier.c -o cliassifier`

## Activation functions

Hidden Layers:

* Sigmoid
* Relu
* Tanh

We define the Activation function as Macro definition.

```c
#define ACTIVATION_FUNCTION 0 //0 for "tanh", 1 for "relu", 2 for "logistic"

```

Output Layer:

* Sigmoid

For Multiclass Classification Problems we use sigmoid(logistic) as activation function for the output layer.

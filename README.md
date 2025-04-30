# Simple Neural Network in C for MNIST

This project implements a basic feedforward neural network in C from scratch to classify handwritten digits from the MNIST dataset. It demonstrates the core concepts of neural networks, including forward propagation, backpropagation, and gradient descent.

This project was a deeply rewarding experience, significantly enhancing my C programming skills through the implementation of a custom matrix math library. It also provided hands-on experience with deep learning fundamentals, particularly in developing the backpropagation algorithm from scratch. Debugging the network's failure to learn proved to be a significant challenge, one that led me to set the project aside for nine months. I'm incredibly pleased to have finally identified the root cause: an incorrect application of the reverse chain rule in the squared error loss function derivative, where I mistakenly retained the squared term instead of using the simple difference.

## Project Structure

The project is organized into several C source and header files:

* `main.c`: Contains the main function to load data, initialize the network, and start the training process.

* `neural.c` / `neural.h`: Implements the neural network structure, including layers, forward pass, backward pass, and the training loop.

* `matrixmath.c` / `matrixmath.h`: Provides basic matrix operations (addition, multiplication, transpose, element-wise operations) necessary for neural network calculations.

* `sigmoid.c` / `sigmoid.h`: Implements the sigmoid activation function and its derivative.

* `mnist.c` / `mnist.h`: Contains functions to load and preprocess the MNIST dataset.

## Getting Started

### Prerequisites

* A C compiler (like GCC).

* The MNIST dataset files (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`). You can typically download these from the [official MNIST website](http://yann.lecun.com/exdb/mnist/).

### Data Setup

Place the four MNIST data files into a directory named `data` at the root of your filesystem (i.e., `/data/`). The project is configured to look for the data files in this specific location.
```
/
└── data/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

### Building the Project

1. Navigate to the project directory in your terminal.

2. Compile the source files using a C compiler. If you are using GCC, you can use the following command:

`gcc main.c matrixmath.c mnist.c neural.c sigmoid.c -o mnist_neural_network -lm`

This command compiles all the necessary `.c` files and links the math library (`-lm`), creating an executable file named `mnist_neural_network`.

### Running the Project

After successfully building the project and setting up the data directory, you can run the executable from your terminal:

`./mnist_neural_network`

The program will load the MNIST data, initialize the neural network, and begin the training process. You will see output in the console indicating the training progress, including batch accuracy.

## Current Features

* Basic feedforward neural network architecture.

* Sigmoid activation function.

* Squared error cost function.

* Mini-batch gradient descent for training.

* Loading and preprocessing of the full MNIST training dataset.

## Potential Future Improvements

* Implement other activation functions (e.g., ReLU, Leaky ReLU).

* Implement cross-entropy loss for better classification performance.

* Add support for different optimizers (e.g., Adam, RMSprop).

* Implement a separate evaluation function for the test dataset.

* Add command-line arguments for configuring network parameters (learning rate, epochs, batch size, etc.).

* Improve memory management and efficiency.

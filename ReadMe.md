# Practical5 :
### Why GPUs Are Used for Neural Networks:
1. Parallel Processing: GPUs have thousands of cores that can handle multiple operations simultaneously, ideal for matrix-heavy computations in neural networks.
2. Efficiency: GPUs process large datasets faster due to their architecture and high memory bandwidth, making them more efficient than CPUs for deep learning tasks.
### What Are Batches:
1. Batches: Small groups of data processed at a time during training. Instead of updating weights after the entire dataset, batches allow more frequent, memory-efficient updates, leading to faster and more stable training.

### Matrix Multiplication
1. If matrix A is of size m×n and matrix B is n×p, their product will be a matrix of size m×p
2. The number of columns of Matrix A must match the number of rows of Matrix B.
3. Therefore, sometimes we have to do Transpose of matrix

### Why we use Activation Functions?
Ans: Activation functions are used in neural networks to introduce non-linearity into the model. This non-linearity allows the network to learn and represent complex patterns and relationships in the data, which enables it to solve more complex problems. Without activation functions, a neural network would essentially be just a linear model, limiting its capability.

### What is `nnfs` package?
The nnfs package is used for educational purposes to simplify and illustrate the implementation of neural networks. It provides a set of functions and classes that help you build and understand neural networks from scratch, focusing on core concepts and algorithms without relying on high-level libraries like TensorFlow or PyTorch. This makes it easier to learn and experiment with the fundamentals of neural networks.



the "dying neurons" problem, which commonly occurs in neural networks, particularly with activation functions like the ReLU (Rectified Linear Unit). When neurons in the network produce a constant zero output, they are said to have "died." This can happen when weights are updated in such a way that the input to the ReLU activation becomes negative, causing it to output zero (because ReLU outputs zero for negative inputs). Over time, these neurons stop contributing to the learning process.
To prevent neurons from dying (outputting zeros), we initialize biases to small positive values, use proper weight initialization (like He), or use activations like Leaky ReLU. Techniques like batch normalization also help keep neurons active during training.




###### Softmax is used at the output layer for multi-class classification because it converts the raw network output into probabilities that sum to 1, making it suitable for predicting multiple classes.

Difference between Sigmoid and Softmax:

* Sigmoid: Used for binary classification. It outputs a probability for a single class (between 0 and 1).
* Softmax: Used for multi-class classification. It outputs probabilities for each class, where the sum of probabilities is 1.




##### If we do not use a loss function in a neural network, the model will have no way to measure the error or difference between its predictions and the actual target values. Without this feedback, the network cannot learn or improve during training, as there would be no basis for adjusting the weights through backpropagation. The loss function is crucial for guiding the learning process.
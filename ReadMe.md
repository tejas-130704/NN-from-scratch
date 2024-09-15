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
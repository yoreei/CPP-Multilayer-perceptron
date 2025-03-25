# Multilayer Perceptron with C++ from Scratch

This is a Multilayer Perceptron (MLP) from scratch in C++. The MLP achieves 97%+ accuracy on the MNIST dataset after 80 epochs.

This is my original work for the ML@Chaos (Chaos Camp Vol.2) course.

You can access the final presentation slides here: https://www.canva.com/design/DAGe6p4vc_I/1G5lmcWNNLghCjH7JtL0qQ/edit?utm_content=DAGe6p4vc_I&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
(they cover only the Eigen project, the other projects were added after the course)

## Implementations

The MLP is implemented using three different libraries:
- Eigen
- OpenBlas
- cuBlas

These are split into 3 separate VS projects (all in one solution)

## Benchmarks

Benchmark based on 10 epochs:

Eigen project uses .noalias(), memory preallocation & 32 threads to maximize performance

All projects comiled for AVX2 & /fp:fast

### Ryzen Bench
Ryzen ???, Rtx 3070 Ti

Mini-batch size: 64
- Eigen : 5.9s
- openBlas & custom SIMD: 5.1s
- cuBlas: ?

Mini-batch size: 128
- Eigen\*: 111s
- openBlas & custom SIMD:  3.2s
- cuBlas: ?

### Intel Laptop Bench

### Eigen Remark

Notice that Eigen has a problem with larger minibatches, specifically, matrices with more than 9984 elements are handled extremely slowly.
This is not the case for openBlas & my custom SIMD operations, which are faster for larger minibatches (as they should be!)


## Usage

Run the VS2022 project. All library dependencies are in the git repo and VS should find them automatically.
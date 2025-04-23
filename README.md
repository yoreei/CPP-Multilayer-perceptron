# Multilayer Perceptron with C++ from Scratch

This is a Multilayer Perceptron (MLP) written from scratch in C++ with a portable C interface. The MLP achieves 97%+ accuracy on the MNIST dataset after 80 epochs.

This is my original work for the ML@Chaos (Chaos Camp Vol.2) course.

You can access the final presentation slides here: [on canva](https://www.canva.com/design/DAGe6p4vc_I/1G5lmcWNNLghCjH7JtL0qQ/edit?utm_content=DAGe6p4vc_I&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
(they cover only the Eigen project, the other projects were added after the project submission)

## Implementations

The MLP is implemented using three different libraries:
- Eigen
- OpenBlas
- cuBlas

These are split into 3 separate Visual Studio 2022 projects (all in one solution)

The *cuBlas* solution also builds as a library with a portable C interface, see **cuBlasMlp/cublas_mlp_api.h**. The Eigen and OpenBlas projects do not expose a C interface.

## GUI

There is a Qt6 GUI project in folder **qt-frontend/** that lets you draw while the cuBlasMlp library analyzes your image in real-time. It also lets you browse the MNIST dataset.
Video Demonstration of the GUI:

[https://youtu.be/9dwyB3GEBBo](https://youtu.be/9dwyB3GEBBo)

## Benchmarks

Benchmark is based on the average time for 1 epoch across 10 epochs:

Eigen project uses .noalias(), memory preallocation & 32 threads to maximize performance

All projects compiled for AVX2 & /fp:fast

### Ryzen Desktop Bench

AMD Ryzen 9 5950X 16-Cores 3401Mhz, Rtx 3070 Ti

Mini-batch size: 64
- Eigen : 0.6s
- openBlas & custom SIMD: 0.55s

Mini-batch size: 128
- Eigen\*: 9.6s
- openBlas & custom SIMD: 0.33s
- cuBlas: 0.095s

### Intel Laptop Bench

Intel Core Ultra 7 165H 16-Cores, Rtx 4070

Mini-batch size: 64
- Eigen : 170s
- openBlas & custom SIMD: 0.67s

Mini-batch size: 128
- Eigen\*: 208s
- openBlas & custom SIMD:  0.4s
- cuBlas: 0.090s

### Eigen Remark

Notice that Eigen has a problem with larger minibatches, specifically, matrices with more than 9984 elements are handled extremely slowly.
This is not the case for openBlas & my custom SIMD operations, which are faster for larger minibatches (as they should be!). For the Eigen project, there is also a huge difference between the Ryzen and the Intel benchmarks, favoring Ryzen.

## Usage

Run the VS2022 project. All library dependencies are in the git repo and VS should find them automatically.


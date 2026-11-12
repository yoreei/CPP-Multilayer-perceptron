# Multilayer Perceptron in C++ and CUDA, from Scratch

A neural network (MLP) for handwritten-digit recognition, written from scratch three times: with **Eigen**, with **OpenBLAS + hand-written AVX2 SIMD**, and with **cuBLAS + custom CUDA kernels**. The CUDA version is also a DLL with a portable C API, which a **Qt 6 GUI** uses to classify digits as you draw them.

<a href="https://www.youtube.com/shorts/9dwyB3GEBBo">
  <img src="docs/demo-thumbnail.jpg" width="300" alt="Qt GUI classifying a hand-drawn 4 with 1.00 confidence. Click to watch the demo video on YouTube.">
</a>

▶ [Watch the demo on YouTube](https://www.youtube.com/shorts/9dwyB3GEBBo): drawing digits in the GUI while the network predicts them live.

## Highlights

- **About 96.5% test accuracy on MNIST** (cuBLAS backend, 100 epochs, one hidden layer of 128 units).
- **0.09 s per training epoch on a laptop GPU**: about 4× faster than the OpenBLAS + AVX2 version and over 2,000× faster than the Eigen version on the same laptop (see [Benchmarks](#benchmarks)).
- **No ML frameworks.** Forward pass, backpropagation and mini-batch SGD are all written by hand.
- **Custom CUDA kernels** for softmax, ReLU, one-hot encoding, bias gradients (with CUB block reductions) and fused loss gradients, next to cuBLAS for the matrix multiplications.
- **Hand-written AVX2 code** in the OpenBLAS version: data is stored directly as `__m256` vectors, and `exp` is vectorized with SLEEF.
- **Self-checking:** at startup, the CUDA and OpenBLAS versions test their math against Eigen (see [Tests](#tests)).

This started as my project for the ML@Chaos course (Chaos Camp Vol. 2). The [presentation slides](https://www.canva.com/design/DAGe6p4vc_I/1G5lmcWNNLghCjH7JtL0qQ/edit?utm_content=DAGe6p4vc_I&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) cover the Eigen version; the OpenBLAS, CUDA and GUI parts came after the course.

## The network

784 inputs (28×28 pixels) → 128 hidden units (ReLU) → 10 outputs (softmax), trained with cross-entropy loss and mini-batch SGD (learning rate 0.01).

| Version | Folder | Linear algebra | Extras |
|---|---|---|---|
| Eigen | [`eigenMlp/`](eigenMlp) | Eigen | `.noalias()`, preallocated buffers, 32 threads |
| OpenBLAS | [`openBlasMlp/`](openBlasMlp) | OpenBLAS `sgemm` | Hand-written AVX2 kernels, SLEEF `exp` |
| cuBLAS | [`cuBlasMlp/`](cuBlasMlp) | cuBLAS / cuBLASLt | Custom CUDA kernels, C API ([`cublas_mlp_api.h`](cuBlasMlp/cublas_mlp_api.h)), builds as a DLL |
| GUI | [`qt-frontend/`](qt-frontend) | uses the cuBLAS DLL | Draw a digit and see live predictions, browse MNIST |

## Benchmarks

Average time per training epoch, over 10 epochs. All versions are built for AVX2 with `/fp:fast`.

| Version | Mini-batch | Ryzen 9 5950X + RTX 3070 Ti | Core Ultra 7 165H + RTX 4070 Laptop |
|---|---|---|---|
| Eigen | 64 | 0.6 s | 170 s |
| OpenBLAS + AVX2 | 64 | 0.55 s | 0.67 s |
| Eigen | 128 | 9.6 s | 208 s |
| OpenBLAS + AVX2 | 128 | 0.33 s | 0.4 s |
| **cuBLAS + CUDA kernels** | 128 | **0.095 s** | **0.090 s** |

**Why Eigen is so slow here:** Eigen becomes extremely slow once a matrix has more than 9,984 elements, which is why its time explodes with larger mini-batches. It is also far slower on the Intel laptop than on the Ryzen desktop. The OpenBLAS + AVX2 version gets faster with larger batches, as it should.

### Optimization log (cuBLAS version)

Training time for 110 epochs (mini-batch 128), from [`docs/benchCublas.txt`](docs/benchCublas.txt):

| Step | Training time |
|---|---|
| First working version | 43.4 s |
| Move semantics instead of copies | 30.9 s |
| Preallocated "ones" vector | 25.6 s |
| Fixed a reallocation bug in `dupRows` | 14.9 s |
| Fused kernel for "softmax output minus one-hot" | 12.2 s |
| Bias gradients in a custom kernel with CUB block reductions | ≈11.5 s |
| Final | **10.2 s** (96.6% accuracy) |

A CPU profile taken during this work showed that the CPU spent most of its time inside the CUDA driver, launching and waiting for many small GPU operations, and only about 9% in the project's own code. Fusing work into fewer custom kernels is what brought the time down. Details: [`docs/profile-2025-03-02-cublas.md`](docs/profile-2025-03-02-cublas.md).

## Tests

The cuBLAS and OpenBLAS versions run their unit tests every time they start. Each test runs one operation on every combination of small matrix sizes (1 to 5) and compares the result with the same math done in Eigen:

- matrix multiplication with all four transpose combinations
- ReLU, softmax, one-hot encoding, row broadcasting, positive mask
- column sums and GPU memory management (cuBLAS version only)

The cuBLAS version prints `Passed all unit tests`. A failed check prints `Test failed` and stops the program in Debug builds (Release builds compile out the `assert`).

## Building and running

### Requirements

- Windows with Visual Studio 2022 (MSVC v143, C++20)
- An AVX2-capable CPU
- CUDA Toolkit 12.x and an NVIDIA GPU (cuBLAS version and GUI)
- Qt 6 and CMake 3.16+ (GUI only)

Eigen, OpenBLAS and SLEEF are included in the repository ([`include/`](include), [`lib/`](lib), [`dll/`](dll)).

Tested with MSVC 14.44, CUDA 12.8 and Qt 6.9.

### 1. Unpack MNIST

The programs read MNIST from `assets.ignored/`. Unpack the included zip there:

```powershell
Expand-Archive MNIST_ORG.zip -DestinationPath assets.ignored
```

### 2. Build and run the three versions

Open `cppMlp.sln` in Visual Studio, select **Release | x64**, build, and run any of the three projects. Their working directory is already set to the repository root, where `assets.ignored/` is.

From a Developer Command Prompt instead:

```powershell
msbuild cppMlp.sln /p:Configuration=Release /p:Platform=x64
.\x64\Release\cuBlasMlp.exe
```

The other two are `.\x64\Release\openBlasMlp.exe` and `.\x64\Release\eigenMlp.exe`. Run them from the repository root.

### 3. Build and run the GUI

[`qt-frontend/`](qt-frontend) is a CMake project; open it in Qt Creator or build it with CMake. It links a prebuilt copy of the cuBLAS DLL from [`qt-frontend/dll/`](qt-frontend/dll).

- Run it with **`qt-frontend/` as the working directory**, because it loads MNIST from `../assets.ignored`.
- At startup it trains the network (about 10 seconds on a recent GPU) before predictions appear.
- In the drawing tab, **Space** clears the canvas and **Ctrl+S** saves the 28×28 input to `assets.ignored/tabletOut.png`.

To use your own build of the DLL instead, build `cuBlasMlp` in the **dllRelease** configuration and copy `x64\dllRelease\cuBlasMlp.dll` and `cuBlasMlp.lib` into `qt-frontend/dll/`.

## License

[MIT](LICENSE)

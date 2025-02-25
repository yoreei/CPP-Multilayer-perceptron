#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <random>
#include <immintrin.h>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <type_traits>
#include <cassert>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "common/cuUtil.h"
#include "common/cppUtil.h"
#include "common/simdUtil.h"

#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX23_DEPRECATION_WARNINGS
#include "Eigen/Dense"
#undef _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#undef _SILENCE_ALL_CXX23_DEPRECATION_WARNINGS

#define SLEEF_STATIC_LIBS
#include "sleef/sleef.h"
#include <openblas/cblas.h>

// define to compare my implementation with an eigen implementation (known to work) (slow)
#define COMPARE_MLP_WITH_EIGEN
// compute loss function (slows down epochs)
#define EVAL_EPOCH

/*
CUP = Cuda (multilayer) Perceptron
*/

using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenRowVectorf = Eigen::RowVector<float, Eigen::Dynamic>;
using EigenVectorf = Eigen::Vector<float, Eigen::Dynamic>;
// BLOCKDIM [1 to 1024]: Number of threads per block in the CUDA kernel
constexpr size_t BLOCK_DIM = 256;

__global__ void relu(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        data[idx] = data[idx] > 0 ? data[idx] : 0;
}

uint32_t swapEndian(uint32_t val) {
    return ((val >> 24) & 0xff) |
        ((val << 8) & 0xff0000) |
        ((val >> 8) & 0xff00) |
        ((val << 24) & 0xff000000);
}

enum CUPTransMask {
    CUPNoneTrans = 0,
    CUPATrans = 1,
    CUPBTrans = 2,
    CUPABTrans = 3,
};

template <typename T>
class CUPRAII {
public:
    CUPRAII(const std::vector<T>& cpuVec) : size(cpuVec.size()) {
        if (ptr) { free(ptr); }
        ptr = cuAllocCpyFromHost(cpuVec);
    }

    CUPRAII(int size) : size(size) {
        if (ptr) { free(ptr); }
        std::vector<T>& cpuVec(size, 0xBADBAD);
        ptr = cuAllocCpyFromHost(cpuVec);
    }

    __host__ __device__
        ~CUPRAII() {
#ifdef __CUDA_ARCH__
        // do not free
#else
        free();
#endif
    }
    CUPRAII(const CUPRAII& rhs) {
        if (this->size != rhs.size) {
            *this = CUPRAII(rhs.size);
            std::cout << "reallocating CUPRAII\n";
        }
        CUDA_CHECK(cudaMemcpy(ptr, rhs.ptr, sizeof(T) * size, cudaMemcpyDeviceToDevice));
    }
    CUPRAII& operator=(const CUPRAII& rhs) {
        if (this->size != rhs.size) {
            *this = CUPRAII(rhs.size);
            std::cout << "reallocating CUPRAII\n";
        }
        CUDA_CHECK(cudaMemcpy(ptr, rhs.ptr, sizeof(T) * size, cudaMemcpyDeviceToDevice));
        return *this;
    }

    __host__ void free() {
        CHECK_CUDA(cudaFree(ptr));
    }
    T* ptr = nullptr;
    size_t size;
};

/* non-owning type */
template <typename T>
struct CUPMatrix {
    CUPMatrix() = default;
    __host__ CUPMatrix(const std::vector<T>& cpuVec, int rows, int cols) : rows(rows), cols(cols), raii(cpuVec), {
        assert(rows * cols == cpuVec.size());
    }
        __host__ CUPMatrix(int rows, int cols) : rows(rows), cols(cols), raii(rows* cols), {}

        size_t get(int row, int col) {
        assert(row < rows && col < cols);
        return row * cols + col;
    }
    T* end() {
        return data + rows * cols;
    }
    std::vector<T> cpyFromDevice() const {
        std::vector<T> cpuVec(rows * cols, 0xBADBAD);
        cuCpyFromDevice(cpuVec, data);
        return cpuVec;
    }

    int size() {
        assert(rows * cols <= raii.size);
        return rows * cols;
    }

    void setView(int rowOffset, int rowSpan) {
        T* newData = _dataStart + rowOffset * cols;
        if (newData < raii.ptr || newData + rowSpan * cols >= raii.ptr + raii.size) {
            throw std::runtime_error("wrong offset & span");
        }
        data = newData;
        rows = rowSpan;

    }

    int raiiRows() const {
        assert(raii.size % cols == 0);
        return raii.size() / cols;
    }

    template <CUPTransMask transMask = CUPNoneTrans>
    void gemm(const CUPMatrix<T>& aMatrix, const CUPMatrix<T>& bMatrix, float alpha = 1.f, float beta = 0.f);
    void positive_mask(const CUPMatrix<T>& mask);
    void dup_rows(const CUPMatrix<T>& row);
    void softmax();

    T* data = nullptr; // device pointer!
    int rows = 0;
    int cols = 0;
private:
    CUPRAII<T> raii;
};


// C = αA ∗ B ∗ +βC
template <typename T>
template <CUPTransMask transMask>
void CUPMatrix<T>::gemm(const CUPMatrix& aMatrix, const CUPMatrix& bMatrix, float alpha, float beta) {
    float* C = data;
    const float* A = aMatrix.data;
    const float* B = bMatrix.data;
    int M, N, K;
    int lda, ldb;
    CBLAS_TRANSPOSE aTransOpt;
    CBLAS_TRANSPOSE bTransOpt;

    if constexpr (transMask == MlpNoneTrans) {
        if (aMatrix.cols != bMatrix.gemmRows) {
            throw std::runtime_error("wrong dim");
        }
        M = aMatrix.gemmRows;
        K = aMatrix.cols;
        N = bMatrix.cols;
        lda = K;
        ldb = N;
        aTransOpt = CblasNoTrans;
        bTransOpt = CblasNoTrans;
    }
    else if constexpr (transMask == MlpATrans) {
        if (aMatrix.gemmRows != bMatrix.gemmRows) {
            throw std::runtime_error("wrong dim");
        }
        M = aMatrix.cols; // A trans!
        K = aMatrix.gemmRows; // A trans!
        N = bMatrix.cols;
        lda = M;
        ldb = N;
        aTransOpt = CblasTrans;
        bTransOpt = CblasNoTrans;
    }
    else if constexpr (transMask == MlpBTrans) {
        if (aMatrix.cols != bMatrix.cols) {
            throw std::runtime_error("wrong dim");
        }
        M = aMatrix.gemmRows;
        K = aMatrix.cols;
        N = bMatrix.gemmRows; // B trans!
        lda = K;
        ldb = K;
        aTransOpt = CblasNoTrans;
        bTransOpt = CblasTrans;
    }
    else if constexpr (transMask == (MlpATrans | MlpBTrans)) {
        if (aMatrix.gemmRows != bMatrix.cols) {
            throw std::runtime_error("wrong dim");
        }
        M = aMatrix.cols; // A trans!
        K = aMatrix.gemmRows; // A trans!
        N = bMatrix.gemmRows; // B trans!
        lda = M;
        ldb = K;
        aTransOpt = CblasTrans;
        bTransOpt = CblasTrans;
    }
    else {
        static_assert(false, "wrong mask");
    }

    if (M != rows || N != cols) {
        throw std::runtime_error("wrong C dim");
    }

    cblas_sgemm(CblasRowMajor,
        aTransOpt,
        bTransOpt,
        M,
        N,
        K,
        alpha,
        A,
        lda,	          // lda, leading dimension of A (num col in A)
        B,
        ldb,    // ldb, leading dimension of B (num col in B)
        beta,             // beta
        C,	  // C (OUT, result)
        N);   // ldc, leading dimension of C (num col in C)
}


//v dim(mask) = dim(*this)
template <typename T>
void CUPMatrix<T>::positive_mask(const CUPMatrix<T>& mask) {
    __m256 zeros = _mm256_setzero_ps();
    for (size_t i = 0; i < size256(); ++i) {
        __m256 mask8 = _mm256_cmp_ps(mask.data256[i], zeros, _CMP_GT_OS); // mask > 0 ? 1 : 0
        data256[i] = _mm256_and_ps(data256[i], mask8); // mask == 1 ? elem = elem : elem = 0
    }

}

template <typename T>
void CUPMatrix<T>::dup_rows(const CUPMatrix<T>& row) {
    // O(log n) memcpy calls

    if (row.size() != cols) {
        int _rowSize = row.size();
        throw std::runtime_error("wrong size " + _rowSize);
    }

    size_t block_size = sizeof(row.at32(0)) * row.size32();
    std::memcpy(data32(), row.data32(), block_size);

    size_t copied = 1;
    char* d = reinterpret_cast<char*>(data256.data());

    // double copy region every time
    while (copied < rows) {
        size_t blocks_to_copy = std::min(copied, rows - copied);
        std::memcpy(d + copied * block_size, d, blocks_to_copy * block_size);
        copied += blocks_to_copy;
    }
}


template <typename T>
void CUPMatrix<T>::softmax() {
    for (int row = 0; row < x.rows; ++row) {
        real_t* start32 = x.data32(row);
        real_t* end32 = start32 + x.cols;
        const real_t rowMax = (*std::max_element(start32, end32));
        __m256 sub = _mm256_set1_ps(rowMax);

        __m256* start256 = &x.data256[row * (x.cols / 8)];
        __m256 sum256 = _mm256_setzero_ps();

        // subtract rowMax, compute exp, accumulate sum256
        assert(x.cols % 8 == 0);
        for (size_t i = 0; i < x.cols / 8; ++i) {
            assert(start256 + i < x.end256());
            start256[i] = _mm256_sub_ps(start256[i], sub);
            start256[i] = Sleef_expf8_u10avx2(start256[i]);
            sum256 = _mm256_add_ps(sum256, start256[i]);
        }
        float rowSum = sum256f(sum256);
        __m256 rowSum256 = _mm256_set1_ps(rowSum);

        // divide exponential by rowSum
        for (size_t i = 0; i < x.cols / 8; ++i) {
            assert(start256 + i < x.end256());
            start256[i] = _mm256_div_ps(start256[i], rowSum256);
        }

    }
}


template <typename T>
CUPMatrix<T> readIdxXubyte(const std::string& dataFile) {
    // T == float: we are reading data
    // T == int: we are reading labels
    constexpr int numDim = std::is_same_v(T, int) ? 1 : 3;
    std::cout << "loading " << dataFile << std::endl;
    std::ifstream dataIfstream(dataFile, std::ios::binary);
    if (!dataIfstream) {
        std::cerr << "Unable to open file: " << dataFile << std::endl;
        exit(-1);
    }

    // Read header: magic number, number of images, rows, and columns.
    uint32_t magicNumber = 0;
    dataIfstream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    magicNumber = swapEndian(magicNumber);
    std::cout << "Magic Number: " << magicNumber << "\n";

    std::array<T, numDim> dim;
    int dims[numDim] = { 0, 0, 0 };
    for (int i = 0; i < numDim, ++i) {
        dataIfstream.read(reinterpret_cast<char*>(&(dims[i])), sizeof(numImages));
        dims[i] = swapEndian(dims[i]);
        std::cout << "Dim" << i << ": " << dims[i] << "\n";
    }

    // Read data:
    int totalElements = std::accumulate(dim.cbegin(), dim.cend(), 1, [](int a, const int& b) {return a * b; });
    std::vector<T> cpuData(totalElements, 0xBADBAD);
    for (int i = 0; i < cpuData.size(); ++i) {
        uint8_t byte;
        dataIfstream.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        if (!dataIfstream) {
            throw std::runtime_error("error reading pos " + i);
        }


        cpuData[i] = T(byte);
        if constexpr (std::is_same_v<T, float>) {
            cpuData[i] /= 255.f;
        }
    }

    // Print out statistics and show the 40th image.

    return CUPMatrix<T>{ cpuData, totalElements };

}

namespace _testEig {
    // EigenType: EigenMatrix or EigenRowVectorF
    template <typename EigenType>
    EigenMatrix fromCUPMatrix(const CUPMatrix<float>& cup);
    EigenRowVectorf fromCUPVector(const CUPMatrix<float>& cup);
    EigenMatrix relu(const EigenMatrix& x);
    EigenMatrix softmax(const EigenMatrix& x);
    EigenMatrix one_hot(const EigenMatrix& y, int maxVal);
    EigenMatrix dup_rows(const EigenMatrix& x, const EigenRowVectorf& y);
    EigenMatrix positive_mask(const EigenMatrix& data, const EigenMatrix& mask);
    void cmpMat(const EigenMatrix& a, const EigenMatrix& b);
}

class MLP {

public:
    MLP(CUPMatrix<float>& x, CUPMatrix<int>& y, size_t hiddenSize, int batchSize, float lr, int epochs) {
        size_t inputSize = x.cols;
        auto d_begin = thrust::device_pointer_cast(y.data);
        auto d_end = thrust::device_pointer_cast(y.end());
        auto max_iter = thrust::max_element(d_begin, d_end);
        size_t outputSize = *max_iter + 1;

        std::vector<float>initVec(inputSize * hiddenSize, 0xBADBAD);
        randSeq(initVec.begin(), initVec.end(), -0.01f, 0.01f);
        weight_1 = CUPMatrix<float>(initVec, inputSize, hiddenSize);

        initVec = std::vector<float>(hiddenSize, 0xBADBAD);
        randSeq(initVec.begin(), initVec.end(), 0.f, 1.f);
        bias_1 = CUPMatrix<float>(initVec, hiddenSize, 1);

        initVec = std::vector<float>(hiddenSize * outputSize, 0xBADBAD);
        randSeq(initVec.begin(), initVec.end(), -0.01f, 0.01f);
        weight_2 = CUPMatrix<float>(initVec, hiddenSize, outputSize);

        initVec = std::vector<float>(outputSize, 0xBADBAD);
        randSeq(initVec.begin(), initVec.end(), 0.f, 1.f);
        bias_2 = CUPMatrix<float>(initVec, outputSize, 1);

        std::cout << "Epoch\tLoss\n";
        // slicing totalRows to align with batchSize (no partial batches)
        int totalRows = batchSize * (x.raiiRows() / batchSize);
        for (int epoch = 0; epoch < epochs; ++epoch) {

            Time begin = getTime();
            for (int i = 0; i < totalRows; i += x.rows) {
                x.setView(i, batchSize);
                y.setView(i, batchSize);
                forward(x);
                backward(x, y, lr);
            }
            Seconds elapsed = getTime() - begin;
            std::cout << "epoch time: " << elapsed << std::endl;

#ifdef EVAL_EPOCH
            x.setView(0, x.raiiRows());
            y.setView(0, y.raiiRows());
            evalEpoch(x, epoch);
#endif
        }
    }

    /*
    input: all training data
    startRow: this batch begins at startRow
    batchRows: this batch ends at endRow = startRow + batchSize
    */
    void forward(const CUPMatrix<float>& x) {
#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix batchEig = _testEig::fromCUPMatrix(x);
        EigenMatrix weight_1Eig = _testEig::fromCUPMatrix(weight_1);
        EigenRowVectorf bias_1Eig = _testEig::fromCUPVector(bias_1);
        EigenMatrix z1Eig = (batchEig * weight_1Eig).rowwise() + bias_1Eig;

        EigenMatrix a1Eig = _testEig::relu(z1Eig);

        EigenMatrix weight_2Eig = _testEig::fromCUPMatrix(weight_2);
        EigenRowVectorf bias_2Eig = _testEig::fromCUPVector(bias_2);
        EigenMatrix z2Eig = (a1Eig * weight_2Eig).rowwise() + bias_2Eig;

        EigenMatrix a2Eig = _testEig::softmax(z2Eig);
#endif // COMPARE_MLP_WITH_EIGEN
        z1.dup_rows(bias_1);
        z1.gemm(x, weight_1, 1.f, 1.f);

        a1 = z1;
        int blocks = (a1.size() + BLOCK_DIM - 1) / BLOCK_DIM;
        relu << <blocks, BLOCK_DIM >> > (a1.data, a1.size());
        cudaDeviceSynchronize();

        //z2 = (a1 * weight_2) +(rowWise) bias_2;
        z2.dup_rows(bias_2);
        z2.gemm(a1, weight_2, 1.f, 1.f);

        a2 = z2;
        softmax(a2);

#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix z1Cmp = _testEig::fromMlp(z1);
        EigenMatrix a1Cmp = _testEig::fromMlp(a1);
        EigenMatrix z2Cmp = _testEig::fromMlp(z2);
        EigenMatrix a2Cmp = _testEig::fromMlp(a2);

        _testEig::cmpMat(z1Cmp, z1Eig);
        _testEig::cmpMat(a1Cmp, a1Eig);
        _testEig::cmpMat(z2Cmp, z2Eig);
        _testEig::cmpMat(a2Cmp, a2Eig);
#endif // COMPARE_MLP_WITH_EIGEN
    }
    /*
    input: all training data
    y: all labels
    startRow: this batch begins at startRow
    batchRows: this batch ends at endRow = startRow + batchSize
    */
    void backward(const CUPMatrix<float>& batch, const CUPMatrix<int>& labels, float lr) {
        float divM = 1.f / batch.rows;
#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix y_one_hotEig = _testEig::one_hot(batchY, a2.cols); // a2.cols = outputSize

        EigenMatrix a2Eig = _testEig::fromMlp(a2);
        EigenMatrix dL_dz2Eig = a2Eig - y_one_hotEig;

        EigenMatrix a1Eig = _testEig::fromMlp(a1);
        EigenMatrix dL_dW2Eig = (a1Eig.transpose() * dL_dz2Eig) * divM;

        EigenMatrix weight_2Eig = _testEig::fromMlp(weight_2);
        weight_2Eig -= lr * dL_dW2Eig;

        EigenMatrix dL_db2Eig = dL_dz2Eig.colwise().sum() * divM;

        EigenMatrix bias_2Eig = _testEig::fromMlp(bias_2);
        bias_2Eig -= lr * dL_db2Eig;

        EigenMatrix dL_da1Eig = dL_dz2Eig * weight_2Eig.transpose();

        EigenMatrix z1Eig = _testEig::fromMlp(z1);
        EigenMatrix dL_dz1Eig = (dL_da1Eig.array() * (z1Eig.array() > 0).cast<real_t>()).matrix();

        EigenMatrix batchXEig = _testEig::fromMlp(batchX);
        EigenMatrix dL_dW1Eig = (batchXEig.transpose() * dL_dz1Eig) * divM;

        EigenMatrix weight_1Eig = _testEig::fromMlp(weight_1);
        weight_1Eig -= lr * dL_dW1Eig;

        EigenMatrix dL_db1Eig = dL_dz1Eig.colwise().sum() * divM;

        EigenMatrix bias_1Eig = _testEig::fromMlp(bias_1);
        bias_1Eig -= lr * dL_db1Eig;
#endif // COMPARE_MLP_WITH_EIGEN
        y_one_hot.one_hot(batchY);

        // 2. Compute gradient at output layer:
        // dL_dz2 = a2 - y_one_hot;
        dL_dz2 = a2;
        cblas_saxpy(     // y = y + alpha * x
            dL_dz2.size32(), //n
            -1.0f,		 //alpha
            y_one_hot.data32(), //x
            1,			//incx
            dL_dz2.data32(),    //y
            1			//incy
        );

        // 3. Gradients for the second (output) layer:
        dL_dW2.gemm<MlpATrans>(a1, dL_dz2, divM);

        // weight_2 -= lr * dL_dW2
        cblas_saxpy( // y = y + alpha * x
            dL_dW2.size32(), // x.size
            -lr,			 // alpha
            dL_dW2.data32(), // x
            1,				 // incx
            weight_2.data32(), // y
            1);				 // incy

        // dL_db2 = dL_dz2.colwise().sum() * dimM
        MlpMatrix ones(1, dL_dz2.rows, 1.f);
        dL_db2.gemm(ones, dL_dz2, divM); // gemv could be faster

        // bias_2 -= lr * dL_db2
        cblas_saxpy(		 // y = y + alpha * x
            dL_db2.size32(), // x.size
            -lr,			 // alpha
            dL_db2.data32(), // x
            1,				 // incx
            bias_2.data32(), // y
            1);				 // incy

        // 4. Backpropagate to the hidden layer:
        dL_da1.gemm<MlpBTrans>(dL_dz2, weight_2);

        dL_dz1 = dL_da1;
        dL_dz1.positive_mask(z1);

        // 5. Gradients for the first (hidden) layer:
        dL_dW1.gemm<MlpATrans>(batchX, dL_dz1, divM);

        // weight_1 -= lr * dL_dW1
        cblas_saxpy( // y = y + alpha * x
            dL_dW1.size32(), // x.size
            -lr,			 // alpha
            dL_dW1.data32(), // x
            1,				 // incx
            weight_1.data32(), // y
            1);				 // incy


        // dL_db1 = dL_dz1.colwise().sum() * divM
        ones = MlpMatrix(1, dL_dz1.rows, 1.f);
        dL_db1.gemm(ones, dL_dz1, divM); // gemv could be faster

        //bias_1 -= lr * dL_db1
        cblas_saxpy(		 // y = y + alpha * x
            dL_db1.size32(), // x.size
            -lr,			 // alpha
            dL_db1.data32(), // x
            1,				 // incx
            bias_1.data32(), // y
            1);				 // incy

#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix y_one_hotCmp = _testEig::fromMlp(y_one_hot);
        _testEig::cmpMat(y_one_hotCmp, y_one_hotEig);
        EigenMatrix dL_dz2Cmp = _testEig::fromMlp(dL_dz2);
        _testEig::cmpMat(dL_dz2Cmp, dL_dz2Eig);
        EigenMatrix dL_dW2Cmp = _testEig::fromMlp(dL_dW2);
        _testEig::cmpMat(dL_dW2Cmp, dL_dW2Eig);
        EigenMatrix weight_2Cmp = _testEig::fromMlp(weight_2);
        _testEig::cmpMat(weight_2Cmp, weight_2Eig);
        EigenMatrix dL_db2Cmp = _testEig::fromMlp(dL_db2);
        _testEig::cmpMat(dL_db2Cmp, dL_db2Eig);
        EigenMatrix bias_2Cmp = _testEig::fromMlp(bias_2);
        _testEig::cmpMat(bias_2Cmp, bias_2Eig);
        EigenMatrix dL_da1Cmp = _testEig::fromMlp(dL_da1);
        _testEig::cmpMat(dL_da1Cmp, dL_da1Eig);
        EigenMatrix dL_dz1Cmp = _testEig::fromMlp(dL_dz1);
        _testEig::cmpMat(dL_dz1Cmp, dL_dz1Eig);
        EigenMatrix dL_dW1Cmp = _testEig::fromMlp(dL_dW1);
        _testEig::cmpMat(dL_dW1Cmp, dL_dW1Eig);
        EigenMatrix weight_1Cmp = _testEig::fromMlp(weight_1);
        _testEig::cmpMat(weight_1Cmp, weight_1Eig);
        EigenMatrix dL_db1Cmp = _testEig::fromMlp(dL_db1);
        _testEig::cmpMat(dL_db1Cmp, dL_db1Eig);
        EigenMatrix bias_1Cmp = _testEig::fromMlp(bias_1);
        _testEig::cmpMat(bias_1Cmp, bias_1Eig);
#endif
    }

    void evalEpoch(const CUPMatrix<float> x, int epoch) {
        std::cout << "evalEpoch not implemented\n";
        //float epsilon = 1.0e-6F;
        //forward(); // output is in `a2` member var
        //__m256 epoch_loss = _mm256_setzero_ps();
        //size_t outSize = a2.cols;
        //int batchSize = a2.rows;
        //for (int row = 0; row < batchSize; row += 8) {
        //    __m256 probs = _mm256_set_ps(
        //        *a2.data32(row, y.at32(y.gemmOffset + row)),
        //        *a2.data32(row + 1, y.at32(y.gemmOffset + row + 1)),
        //        *a2.data32(row + 2, y.at32(y.gemmOffset + row + 2)),
        //        *a2.data32(row + 3, y.at32(y.gemmOffset + row + 3)),
        //        *a2.data32(row + 4, y.at32(y.gemmOffset + row + 4)),
        //        *a2.data32(row + 5, y.at32(y.gemmOffset + row + 5)),
        //        *a2.data32(row + 6, y.at32(y.gemmOffset + row + 6)),
        //        *a2.data32(row + 7, y.at32(y.gemmOffset + row + 7)));
        //    probs = _mm256_add_ps(probs, _mm256_set1_ps(epsilon)); // against log(0)
        //    probs = _mm256_log_ps(probs);
        //    epoch_loss = _mm256_add_ps(epoch_loss, probs);
        //}

        //float epoch_loss_sum = sum256f(epoch_loss);
        //epoch_loss_sum /= -int(batchSize);
        //losses.push_back(epoch_loss_sum);

        //std::cout << (epoch + 1) << "\t" << epoch_loss_sum << std::endl;



#ifdef COMPARE_MLP_WITH_EIGEN
        double epoch_lossEig = 0.0;
        for (size_t row = 0; row < batchSize; ++row) {
            float prob = *a2.data32(row, trainData.y.at32(trainData.y.gemmOffset + row));
            prob += epsilon; // against log(0)
            epoch_lossEig += std::log(prob);
        }
        epoch_lossEig /= -batchSize;
        float lastLoss = losses.back();
        if (std::fabsf(epoch_lossEig - lastLoss) > 0.001f) {
            throw std::runtime_error("wrong loss");
        }
#endif // COMPARE_MLP_WITH_EIGEN

    }

    std::vector<int> predict(const CUPMatrix<float>& testX) {
        forward(testX);

        std::vector<int> predictions{ testX.rows, 1 };
        EigenMatrix a2Eig = _testEig::fromCUPMatrix(a2);
        for (int i = 0; i < a2Eig.rows(); ++i) {
            int maxIndex;
            a2Eig.row(i).maxCoeff(&maxIndex);
            predictions[i] = maxIndex;
            std::cout << i << ": " << maxIndex << "\n";
        }

        return predictions;
    }

    CUPMatrix<float> weight_1; //< dim [inputSize x hiddenSize]
    CUPMatrix<float> bias_1;   //< dim [1 x hiddenSize]
    CUPMatrix<float> weight_2; //< dim [hiddenSize x outputSize]
    CUPMatrix<float> bias_2;   //< dim [1 x outputSize]

    CUPMatrix<float> z1; //< dim [batchSize x hiddenSize]
    CUPMatrix<float> z2; //< dim [batchSize x outputSize]
    CUPMatrix<float> a1; //< dim [batchSize x hiddenSize]
    CUPMatrix<float> a2; //< dim [batchSize x outputSize]

    // temp matrices
    CUPMatrix<float> y_one_hot;
    CUPMatrix<float> dL_dz2;
    CUPMatrix<float> dL_dW2; // dim [hiddenSize x outputSize]
    CUPMatrix<float> dL_db2;
    CUPMatrix<float> dL_da1;
    CUPMatrix<float> dL_dz1;
    CUPMatrix<float> dL_dW1;
    CUPMatrix<float> dL_db1;

    std::vector<float> losses;
};

/* Eigen implementations for comparisons with my implementations */
namespace _testEig {

    EigenMatrix fromCUPMatrix(const CUPMatrix<float>& cup) {
        std::vector<float> cpuVec = cup.cpyFromDevice();
        EigenMatrix eig = EigenMatrix(cup.rows, cup.cols);
        for (int row = 0; row < cup.rows; ++row) {
            for (int col = 0; col < cup.cols; ++col) {
                eig(row, col) = cpuVec[row * cup.cols + col];
            }

        }
        return eig;
    }

    EigenRowVectorf fromCUPVector(const CUPMatrix<float>& cup) {
        std::vector<float> cpuVec = cup.cpyFromDevice();
        EigenMatrix eig = EigenRowVectorf(cup.rows);
        for (int row = 0; row < cup.rows; ++row) {
            eig(row) = cpuVec[row];
        }
        return eig;
    }
    EigenMatrix relu(const EigenMatrix& x) {
        return x.array().max(0.0).matrix();
    };

    EigenMatrix softmax(const EigenMatrix& x) {
        EigenMatrix rowMax = x.rowwise().maxCoeff();
        EigenMatrix x_stable = x - rowMax.replicate(1, x.cols());
        EigenMatrix exp_x = x_stable.array().exp();
        EigenVectorf rowSum = exp_x.rowwise().sum();
        EigenMatrix sm = exp_x.array().colwise() / rowSum.array();
        return sm;
    };

    EigenMatrix one_hot(const EigenMatrix& y, int maxVal) {
        assert(y.cols() == 1);
        EigenMatrix y_one_hot = EigenMatrix(y.rows(), maxVal);
        y_one_hot.setZero();
        for (int i = 0; i < y_one_hot.rows(); ++i) {
            int label = y(i);
            y_one_hot(i, label) = 1.0f;
        }
        return y_one_hot;
    };

    EigenMatrix dup_rows(const EigenMatrix& x, const EigenRowVectorf& y) {
        assert(x.cols() == y.cols());
        return x.rowwise() + y;
    };

    EigenMatrix positive_mask(const EigenMatrix& data, const EigenMatrix& mask) {
        return (data.array() * (mask.array() > 0).cast<float>()).matrix();
    };

    void cmpMat(const EigenMatrix& a, const EigenMatrix& b) {
        if (a.isApprox(b)) {
            //std::cout << "Test passed: matrices are equal." << std::endl;
        }
        else {
            std::cerr << "Test failed: matrices differ." << std::endl;
            int printPrecision = 3;
            Eigen::IOFormat fmt(printPrecision, 0, ", ", "\n", "[", "]"); // up printPrecision if diff hard to spot
            std::cerr << "Matrix A:" << std::endl << a.format(fmt) << std::endl;
            std::cerr << "Matrix B:" << std::endl << b.format(fmt) << std::endl;
            assert(false);
        }
    }

    void statistics(EigenMatrix& mat) {
        float minVal = mat.minCoeff();
        float maxVal = mat.maxCoeff();
        float sumVal = mat.sum();

        std::cout << "Data: min = " << minVal
            << ", max = " << maxVal
            << ", sum = " << sumVal << std::endl;

    }
    void printImage(const EigenMatrix& mat, int imageId, int rows, int cols) {
        for (size_t y = 0; y < rows; ++y) {
            for (size_t x = 0; x < cols; ++x) {
                float val = mat(imageId, y * cols + x);
                std::cout << ASCIIArtFromFloat(val);
            }
            std::cout << std::endl;
        }
    }

}


//// ^TESTEIGEN
///////////////////////////////////////////////////


bool nextPermute(std::vector<int>& in, std::vector<int>& out) {

    int n = in.size();
    int k = out.size();
    for (int i = 0; i < k; i++)
    {
        out[i] = in[i];
    }
    std::reverse(in.begin() + k, in.end());
    return std::next_permutation(in.begin(), in.end());
}




void test_gemm(int m, int n, int k) {
    // Test 1: MlpNoneTrans with multi-dim aMatrix
    {
        MlpMatrix mata = MlpMatrix::Random(m, k * 8, -5.f, 5.f);
        MlpMatrix matb = MlpMatrix::Random(k * 8, n * 8, -5.f, 5.f);
        MlpMatrix matc = MlpMatrix::Random(m, n * 8, -5.f, 5.f);
        matc.gemm<MlpNoneTrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromMlp(mata);
        EigenMatrix eigB = _testEig::fromMlp(matb);
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromMlp(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }


    // Test 3: MlpATrans
    {
        MlpMatrix mata = MlpMatrix::Random(k, m * 8, -5.f, 5.f);
        MlpMatrix matb = MlpMatrix::Random(k, n * 8, -5.f, 5.f);
        MlpMatrix matc = MlpMatrix::Random(m * 8, n * 8, -5.f, 5.f);
        matc.gemm<MlpATrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromMlp(mata).transpose();
        EigenMatrix eigB = _testEig::fromMlp(matb);
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromMlp(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }

    // Test 5: MlpBTrans with multi-dim aMatrix
    {
        MlpMatrix mata = MlpMatrix::Random(m, k * 8, -5.f, 5.f);
        MlpMatrix matb = MlpMatrix::Random(n * 8, k * 8, -5.f, 5.f);
        MlpMatrix matc = MlpMatrix::Random(m, n * 8, -5.f, 5.f);
        matc.gemm<MlpBTrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromMlp(mata);
        EigenMatrix eigB = _testEig::fromMlp(matb).transpose();
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromMlp(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }

    // Test 7: (MlpATrans | MlpBTrans) with multi-dim aMatrix
    {
        MlpMatrix mata = MlpMatrix::Random(k * 8, m * 8, -5.f, 5.f);
        MlpMatrix matb = MlpMatrix::Random(n * 8, k * 8, -5.f, 5.f);
        MlpMatrix matc = MlpMatrix::Random(m * 8, n * 8, -5.f, 5.f);
        matc.gemm<MlpABTrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromMlp(mata).transpose();
        EigenMatrix eigB = _testEig::fromMlp(matb).transpose();
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromMlp(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }
}

void test_relu(int m, int n) {
    n *= 8;
    MlpMatrix mlp = MlpMatrix::Random(m, n * 8, -5.f, 5.f);
    EigenMatrix eig = _testEig::fromMlp(mlp);
    relu(mlp);
    EigenMatrix mlpCmp = _testEig::fromMlp(mlp);
    EigenMatrix eigCmp = _testEig::relu(eig);

    _testEig::cmpMat(eigCmp, mlpCmp);
}
void test_softmax(int m, int n) {

    n *= 8;
    MlpMatrix mlp = MlpMatrix::Random(m, n * 8, -5.f, 5.f);
    EigenMatrix eig = _testEig::fromMlp(mlp);

    softmax(mlp);
    EigenMatrix mlpCmp = _testEig::fromMlp(mlp);
    EigenMatrix eigCmp = _testEig::softmax(eig);

    _testEig::cmpMat(eigCmp, mlpCmp);
}
void test_one_hot(int m, int n) {

    n *= 8;
    MlpMatrix mlp = MlpMatrix(m * 8, n * 8);
    EigenMatrix eig = EigenMatrix(m * 8, n * 8);

    MlpVector<__m256i> y = MlpVector<__m256i>::Random(m * 8, 0, n * 8 - 1);
    int outputSize = n * 8; // ensure one_hot produces correct cols
    mlp.one_hot(y);
    EigenMatrix mlpCmp = _testEig::fromMlp(mlp);
    EigenMatrix eigCmp = _testEig::one_hot(y, outputSize);

    _testEig::cmpMat(eigCmp, mlpCmp);
}
void test_dup_rows(int m, int n) {
    n *= 8;
    MlpMatrix mlp = MlpMatrix::Random(m, n * 8, 0.f, 0.f);
    EigenMatrix eig = _testEig::fromMlp(mlp);

    MlpVector<__m256> y = MlpVector<__m256>::Random(n * 8, -5.f, 5.f);
    EigenRowVectorf eigY = _testEig::fromMlp(y);
    mlp.dup_rows(y);
    EigenMatrix mlpCmp = _testEig::fromMlp(mlp);
    EigenMatrix eigCmp = _testEig::dup_rows(eig, eigY);

    _testEig::cmpMat(eigCmp, mlpCmp);

}

void test_positive_mask(int m, int n) {
    n *= 8;
    MlpMatrix mlp = MlpMatrix::Random(m, n * 8, -5.f, 5.f);
    MlpMatrix mlpMask = MlpMatrix::Random(m, n * 8, -5.f, 5.f);
    EigenMatrix eig = _testEig::fromMlp(mlp);
    EigenMatrix eigMask = _testEig::fromMlp(mlpMask);

    mlp.positive_mask(mlpMask);
    EigenMatrix mlpCmp = _testEig::fromMlp(mlp);
    EigenMatrix eigCmp = _testEig::positive_mask(eig, eigMask);

    _testEig::cmpMat(eigCmp, mlpCmp);
}

void testRun() {
    std::vector<int> in{ 1,2,3,4,5 };
    std::vector<int> out(3, 0);
    while (nextPermute(in, out)) {
        int m = out[0];
        int n = out[1];
        int k = out[2];
        test_gemm(m, n, k);
    }
    in = std::vector<int>{ 1,2,3,4,5 };
    out = std::vector<int>(2, 0);
    while (nextPermute(in, out)) {
        int m = out[0];
        int n = out[1];
        test_relu(m, n);
        test_softmax(m, n);
        test_one_hot(m, n);
        test_dup_rows(m, n);
        test_positive_mask(m, n);
    }
}

int main() {
    enableFpExcept();
    testRun();

    CUPMatrix<float> x = readIdxXubyte<float>("assets.ignored/train-images.idx3-ubyte");
    CUPMatrix<int> y = readIdxXubyte<int>("assets.ignored/train-labels.idx1-ubyte");
    CUPMatrix<float> testX = readIdxXubyte<float>("assets.ignored/t10k-images.idx3-ubyte");
    CUPMatrix<int> testY = readIdxXubyte<int>("assets.ignored/t10k-labels.idx1-ubyte");

    const size_t hiddenSize = 128;
    int miniBatchSize = 128;
    MLP mlp{ x, y, hiddenSize, miniBatchSize, 0.01f };

    Time begin = getTime();

    int epochs = 10;
    mlp.train(epochs);
    MlpVector<__m256i> predictions = mlp.predict(testData.x);

    //double accuracy = (predictions.array() == testData.labels.array()).cast<double>().mean();
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < predictions.size256(); ++i) {

        __m256i mask = _mm256_cmpeq_epi32(predictions.at256(i), testData.y.at256(i));
        __m256i ones = _mm256_srli_epi32(mask, 31); // >> 31 for each 32-bit element
        acc = _mm256_add_epi32(acc, ones);
    }

    std::cout << "Test Accuracy: " << sum256i(acc) / float(testData.y.gemmRows) << std::endl;

    Seconds elapsed = getTime() - begin;
    std::cout << "benchmark: " << elapsed;


    return 0;
}


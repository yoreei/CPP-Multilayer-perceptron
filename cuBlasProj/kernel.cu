#pragma once

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

#define EIGEN_NO_CUDA
#include "Eigen/Dense"

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <cub/cub.cuh>

#define DISABLE_INLINING
#include "common/cppUtil.h"
#include "common/cuUtil.h"
#include "common/simdUtil.h"
#include "common/cublasUtil.h"

#include "cublas_mlp_api.h"

// define to compare my implementation with an eigen implementation (known to work) (slow)
#undef COMPARE_MLP_WITH_EIGEN
// compute loss function (slow!)
#undef EVAL_EPOCH

inline constexpr int EPOCHS = 100;
inline constexpr int BATCH_SIZE = 128;
inline constexpr float LR = 0.01f;

/*
CUP = Cuda (multilayer) Perceptron
*/

using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenRowVectorf = Eigen::RowVector<float, Eigen::Dynamic>;
using EigenVectorf = Eigen::Vector<float, Eigen::Dynamic>;
// BLOCKDIM [1 to 1024]: Number of threads per block in the CUDA kernel
constexpr size_t BLOCK_DIM = 256;

/* a singleton class is not elegant for CublasHandle::get(), but it works for this project */
class CublasHandle {
public:
    static cublasHandle_t& get() {
        if (mustInitCublas) {
            CUBLAS_CHECK(cublasCreate(&cublasHandle));
            mustInitCublas = false;
        }
        return cublasHandle;
    }

    static cublasLtHandle_t& getLt() {
        if (mustInitLt) {
            cublasLtCreate(&ltHandle);
            mustInitLt = false;
        }
        return ltHandle;
    }

    static void free() {
        if (!mustInitCublas) {
            cublasDestroy(cublasHandle);
        }
        if (!mustInitLt) {
            cublasLtDestroy(ltHandle);
        }
    }
private:
    inline static bool mustInitCublas = true;
    inline static cublasHandle_t cublasHandle;
    inline static bool mustInitLt = true;
    inline static cublasLtHandle_t ltHandle;
};

/* host-only, owning class */
template <typename T>
class CUPRAII : public Traceable<CUPRAII<T>> {
public:
    explicit CUPRAII(const std::vector<T>& cpuVec) : size(cpuVec.size()) {
        ptr = cuAllocCpyFromHost(cpuVec);
    }

    explicit CUPRAII(int size) : size(size) {
        if constexpr (DEBUG) {
            std::vector<T> cpuVec(size, std::bit_cast<T>(0xBADDBADD));
            ptr = cuAllocCpyFromHost(cpuVec);
        }
        else {
            int bytes = sizeof(T) * size;
            CU_CHECK(cudaMalloc(&ptr, bytes));
        }
    }

    explicit CUPRAII(int size, const T* cpuData) : size(size) {
        int bytes = sizeof(T) * size;
        CU_CHECK(cudaMalloc(&ptr, bytes));
        CU_CHECK(cudaMemcpy(ptr, cpuData, bytes, cudaMemcpyHostToDevice));
    }

    ~CUPRAII() {
        release();
    }
    CUPRAII(const CUPRAII& rhs) : CUPRAII(rhs.size) {
        size_t bytes = sizeof(T) * rhs.size;
        CU_CHECK(cudaMemcpy(ptr, rhs.ptr, bytes, cudaMemcpyDeviceToDevice));
    }

    CUPRAII& operator=(const CUPRAII& rhs) {
        size_t bytes = sizeof(T) * rhs.size;
        if (this->size != rhs.size) {
            //std::cout << "CUPRAII operator= reallocating!\n";
            release();
            this->size = rhs.size;
            CU_CHECK(cudaMalloc(&ptr, bytes));
        }
        CU_CHECK(cudaMemcpy(ptr, rhs.ptr, bytes, cudaMemcpyDeviceToDevice));
        return *this;
    }

    CUPRAII& operator=(CUPRAII&& rhs) {
        ptr = rhs.ptr;
        size = rhs.size;
        rhs.ptr = nullptr;
        rhs.size = 0;
        return *this;
    }
    CUPRAII(CUPRAII&& rhs) {
        ptr = rhs.ptr;
        size = rhs.size;
        rhs.ptr = nullptr;
        rhs.size = 0;
    }

    void release() {
        CU_CHECK(cudaFree(ptr));
        ptr = nullptr;
        size = 0;
    }
    T* ptr = nullptr;
    size_t size = 0;
};

enum CUPTransMask {
    CUPNoneTrans = 0,
    CUPATrans = 1,
    CUPBTrans = 2,
    CUPABTrans = 3,
};

/* non-owning type, used in Cuda kernels */
template <typename T>
struct PODMatrix {
    int rows = 0;
    int cols = 0;
    T* data = nullptr;
    __host__ size_t getIdx(int row, int col) const {
        assert(row < rows && col < cols);
        return row * cols + col;
    }
    __device__ size_t d_getIdx(int row, int col) const {
        return row * cols + col;
    }

    __host__ __device__ T* end() {
        return data + rows * cols;
    }

    __host__ __device__ int size() const {
        //assert(rows * cols <= raii.size);
        return rows * cols;
    }

};

template <typename T>
struct CPUMatrix : public PODMatrix<T> {
    using PODMatrix<T>::rows;
    using PODMatrix<T>::cols;
    using PODMatrix<T>::data;
    CPUMatrix(int _rows, int _cols) {
        rows = _rows;
        cols = _cols;
        data = new T[rows * cols];
        memset(data, std::bit_cast<T>(0xBA), sizeof(T) * rows * cols);
    }

    ~CPUMatrix() {
        delete[] data;
    }

    CPUMatrix(const CPUMatrix& rhs) = delete;
    CPUMatrix(CPUMatrix&& rhs) {
        rows = rhs.rows;
        cols = rhs.cols;
        data = rhs.data;

        rhs.data = nullptr;
        rhs.rows = 0;
        rhs.cols = 0;
    }

    CPUMatrix& operator=(const CPUMatrix& rhs) = delete;
    CPUMatrix& operator=(CPUMatrix&& rhs) {
        rows = rhs.rows;
        cols = rhs.cols;
        data = rhs.data;

        rhs.data = nullptr;
        rhs.rows = 0;
        rhs.cols = 0;
        return *this;
    }

};

/* owning type, stores data on gpu, used in host only */
template <typename T>
struct GPUMatrix : public PODMatrix<T> {
    using PODMatrix<T>::rows;
    using PODMatrix<T>::cols;
    using PODMatrix<T>::data;
    using PODMatrix<T>::end;
    using PODMatrix<T>::size;
    using PODMatrix<T>::getIdx;
    GPUMatrix() = default;
    explicit GPUMatrix(const CPUMatrix<T>& cpuMatrix) : PODMatrix<T>{ cpuMatrix.rows, cpuMatrix.cols, nullptr }, raii(cpuMatrix.size(), cpuMatrix.data) {
        data = raii.ptr;
    }
    explicit GPUMatrix(const std::vector<T>& cpuVec, int rows, int cols) : PODMatrix<T>{ rows, cols, nullptr }, raii(cpuVec) {
        if (rows * cols != cpuVec.size()) {
            throw std::runtime_error("wrong dims");
        }
        data = raii.ptr;
    }
    explicit GPUMatrix(int rows, int cols) : PODMatrix<T>{ rows, cols, nullptr }, raii(rows* cols) {
        data = raii.ptr;
    }

    explicit GPUMatrix(int rows, int cols, const T val) : PODMatrix<T>{ rows, cols, nullptr } {
        std::vector<T> cpuVec(rows * cols, val);
        raii = CUPRAII{ cpuVec };
        data = raii.ptr;
    }

    explicit GPUMatrix(int rows, int cols, const T* ptr) : PODMatrix<T>{ rows, cols, nullptr } {
        raii = CUPRAII{ rows * cols, ptr };
        data = raii.ptr;
    }

    explicit GPUMatrix(const CUPRAII<T>& raii, int rows, int cols) : PODMatrix<T>{ rows, cols, nullptr }, raii(raii) {
        data = raii.ptr;
    }

    GPUMatrix(const GPUMatrix& rhs) : GPUMatrix(rhs.raii, rhs.rows, rhs.cols) {
        int diff = rhs.data - rhs.raii.ptr;
        data = raii.ptr + diff;
    }

    GPUMatrix& operator=(const GPUMatrix& rhs) {
        raii = rhs.raii;
        rows = rhs.rows;
        cols = rhs.cols;

        int diff = rhs.data - rhs.raii.ptr;
        data = raii.ptr + diff;
        return *this;
    }

    GPUMatrix& operator=(GPUMatrix&& rhs) = default;
    GPUMatrix(GPUMatrix&& rhs) = default;

    const PODMatrix<T> getPod() const {
        return PODMatrix<T>(*this);
    }

    static GPUMatrix Random(int rows, int cols, T minVal, T maxVal) {
        std::vector<T> ranVec(rows * cols, 0xBADDBADD);
        randSeq(ranVec.begin(), ranVec.end(), minVal, maxVal);
        return GPUMatrix(ranVec, rows, cols);
    }

    std::vector<T> cpyFromDevice() const {
        std::vector<T> cpuVec(rows * cols, 0xBADDBADD);
        cuCpyFromDevice<T>(cpuVec, data);
        return cpuVec;
    }

    void setView(size_t _rowOffset, size_t rowSpan) {
        if (_rowOffset + rowSpan > raiiRows()) {
            throw std::runtime_error("wrong view");
        }
        data = raii.ptr + _rowOffset * cols;
        rows = rowSpan;
        assert(data >= raii.ptr && end() <= raii.ptr + raii.size);
    }

    int getRowOffset() {
        int diff = data - raii.ptr;
        assert(diff % cols == 0);
        return diff / cols;
    }

    const CUPRAII<T>& getRaii() const {
        return raii;
    }

    int raiiRows() const {
        assert(raii.size % cols == 0);
        return raii.size / cols;
    }

    INLINING void a2Sub1Hot(const GPUMatrix<float>& a2, const GPUMatrix<int>& batchY) {
        if (rows != a2.rows || cols != a2.cols) {
            *this = GPUMatrix<float>{ a2.rows, a2.cols };
        }

        assert(batchY.rows == rows);

        int blocks = (size() + BLOCK_DIM - 1) / BLOCK_DIM;
        cuA2Sub1Hot << <blocks, BLOCK_DIM >> > (getPod(), a2.getPod(), batchY.getPod());
        cudaDeviceSynchronize();
    }

    template <CUPTransMask transMask = CUPNoneTrans>
    INLINING void gemm(const GPUMatrix<T>& aMatrix, const GPUMatrix<T>& bMatrix, float alpha = 1.f, float beta = 0.f) {

        const float* A = aMatrix.data;
        const float* B = bMatrix.data;
        int M, N, K;
        int lda, ldb;

        cublasOperation_t aTransOpt;
        cublasOperation_t bTransOpt;

        if constexpr (transMask == CUPNoneTrans) {
            if (aMatrix.cols != bMatrix.rows) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.rows;
            K = aMatrix.cols;
            N = bMatrix.cols;
            lda = K;
            ldb = N;
            aTransOpt = CUBLAS_OP_N;
            bTransOpt = CUBLAS_OP_N;
        }
        else if constexpr (transMask == CUPATrans) {
            if (aMatrix.rows != bMatrix.rows) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.cols; // A trans!
            K = aMatrix.rows; // A trans!
            N = bMatrix.cols;
            lda = M;
            ldb = N;
            aTransOpt = CUBLAS_OP_T;
            bTransOpt = CUBLAS_OP_N;
        }
        else if constexpr (transMask == CUPBTrans) {
            if (aMatrix.cols != bMatrix.cols) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.rows;
            K = aMatrix.cols;
            N = bMatrix.rows; // B trans!
            lda = K;
            ldb = K;
            aTransOpt = CUBLAS_OP_N;
            bTransOpt = CUBLAS_OP_T;
        }
        else if constexpr (transMask == (CUPATrans | CUPBTrans)) {
            if (aMatrix.rows != bMatrix.cols) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.cols; // A trans!
            K = aMatrix.rows; // A trans!
            N = bMatrix.rows; // B trans!
            lda = M;
            ldb = K;
            aTransOpt = CUBLAS_OP_T;
            bTransOpt = CUBLAS_OP_T;
        }
        else {
            static_assert(false, "wrong mask");
        }

        if (M != rows || N != cols) {
            raii.release();
            //std::cout << "GPUMatrix<T>::gemm reallocate\n";
            *this = GPUMatrix<T>{ M, N };
        }
        float* C = data;

        int ldc = N;
        size_t workspaceSize = 0;
        void* workspace = nullptr; // or allocate a workspace if desired
        LtSgemm(CublasHandle::getLt(),
            aTransOpt,
            bTransOpt,
            M,
            N,
            K,
            &alpha, /* host pointer */
            A,
            lda,
            B,
            ldb,
            &beta, /* host pointer */
            C,
            ldc,
            workspace,
            workspaceSize);
    }

    INLINING void colwiseSumAlpha(const GPUMatrix<T>& mat, GPUMatrix<T>& ones, float alpha, float beta = 0.f) {
        // resize `ones` if necessary:
        if (ones.cols != 1 || ones.rows < mat.rows) {
            std::cout << "ColwiseSumAlpha reallocate ones\n";
            ones = GPUMatrix<T>(mat.rows, 1, 1.f);
        }
        if (this->rows != 1 || this->cols != mat.cols) {
            std::cout << "ColwiseSumAlpha reallocate C\n";
            *this = GPUMatrix<T>(1, mat.cols);
        }

        int N = mat.rows;
        int M = mat.cols;

        cublasSgemv(CublasHandle::get(),
            CUBLAS_OP_N, // transpose A
            M, N,        // dimensions of A
            &alpha,
            mat.data, M,   // A pointer and leading dimension
            ones.data, 1,       // ones vector
            &beta,
            this->data, 1);       // output vector

    }

    INLINING void positiveMask(const GPUMatrix<T>& mask) {
        assert(cols == mask.cols && rows == mask.rows);
        int blocks = (size() + BLOCK_DIM - 1) / BLOCK_DIM;
        cuPositiveMask << <blocks, BLOCK_DIM >> > (getPod(), mask.getPod());
        cudaDeviceSynchronize();
    }

    INLINING void dupRows(const GPUMatrix<T>& row, int numRows) {
        assert(row.cols == 1); // we are assuming a row vector here!
        // O(log n) memcpy calls
        if (rows != numRows || cols != row.rows) {
            raii.release();
            *this = GPUMatrix<T>{ numRows, row.rows };
        }

        if (row.size() != cols) {
            int _rowSize = row.size();
            throw std::runtime_error("wrong size " + std::to_string(_rowSize));
        }

        int blocks = (size() + BLOCK_DIM - 1) / BLOCK_DIM;
        dim3 blockDim(32, 8);
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x,
            (rows + blockDim.y - 1) / blockDim.y);

        cuDupRows << <gridDim, blockDim >> > (getPod(), row.getPod());
        cudaDeviceSynchronize();
    }

    INLINING void relu() {
        int blocks = (size() + BLOCK_DIM - 1) / BLOCK_DIM;
        cuRelu << <blocks, BLOCK_DIM >> > (getPod());
        cudaDeviceSynchronize();
    }

    INLINING void oneHot(const GPUMatrix<int>& y, int maxVal) {
        assert(y.cols == 1);
        if (raii.size != y.rows * maxVal) {
            *this = GPUMatrix(y.rows, maxVal);
        }
        rows = y.rows;
        cols = maxVal;

        int blocks = (size() + BLOCK_DIM - 1) / BLOCK_DIM;
        cuOneHot << <blocks, BLOCK_DIM >> > (getPod(), y.getPod());
        cudaDeviceSynchronize();
    }

    INLINING void softmax() requires std::same_as<T, float> {
        // matrix.cols is currently hardcoded for this function. TODO unhardcode!
        constexpr int BlockSize = 32; // must be multiple of warp size?
        if (cols > BlockSize) {
            throw std::runtime_error("unhardcode me");
        }

        int blocks = rows;
        cuSoftmax<BlockSize> << <blocks, BlockSize >> > (getPod());
        cudaDeviceSynchronize();
    }

    INLINING void computeBias(const GPUMatrix<float>& dL_dz, float divM, float lr) {
        if (rows != dL_dz.cols || cols != 1) {
            throw std::runtime_error("dims");
        }

        constexpr int ItemsPerThread = 2;
        static_assert(BATCH_SIZE % ItemsPerThread == 0);
        int blocks = rows;

        BenchId benchId = cppBench("cuComputeBias");
        cuComputeBias<BATCH_SIZE / ItemsPerThread, ItemsPerThread> << <blocks, BATCH_SIZE >> > (getPod(), dL_dz.getPod(), -divM * lr);
        cudaDeviceSynchronize();
        cppBenchEnd(benchId);
    }

    CUPRAII<T> raii{ 0 };
};

__global__ void cuA2Sub1Hot(PODMatrix<float> dL_dz2, PODMatrix<float> a2, PODMatrix<int> batchY) {
    //    int outputSize = a2.cols;
    //    y_one_hot.oneHot(batchY, outputSize);
    //    dL_dz2 = a2 - y_one_hot;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = idx / a2.cols;
    int col = idx % a2.cols;
    if (idx < a2.size()) {
        dL_dz2.data[idx] = a2.data[idx];
        if (col == batchY.data[row]) {
            dL_dz2.data[idx] -= 1.f;
        }
    }

}

template <int BlockSize, int ItemsPerThread >
__global__ void cuComputeBias(PODMatrix<float> bias, const PODMatrix<float> dL_dz, float alpha) {
    //EigenMatrix dL_db2Eig = (dL_dz2Eig.colwise().sum() * divM);
    //bias_2Eig -= lr * dL_db2Eig.transpose();

    static_assert(BlockSize % 32 == 0);
    // Each block processes one row
    using BlockReduce = cub::BlockReduce<float, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int rowStart = threadIdx.x * ItemsPerThread;

    float thread_sum = 0.f;
    for (int i = 0; i < ItemsPerThread; ++i) {
        thread_sum = (threadIdx.x < dL_dz.rows) ? dL_dz.data[blockIdx.x + (rowStart + i) * dL_dz.cols] : 0;
    }

    // Only thread_0 in block has the correct result
    float block_sum = BlockReduce(temp_storage).Reduce(thread_sum, cub::Sum());

    if (threadIdx.x == 0) {
        bias.data[blockIdx.x] += block_sum * alpha;
    }

}

//BlockSize given as template because we need it to be constexpr
template <int BlockSize>
__global__ void cuSoftmax(PODMatrix<float> mat) {
    assert(blockIdx.x < mat.rows);
    static_assert(BlockSize % 32 == 0); // BlockReduce requires multiple of warp size
    // Each block processes one row
    using BlockReduce = cub::BlockReduce<float, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float sync_reduce;

    int idx = mat.d_getIdx(blockIdx.x, threadIdx.x);
    float thread_max = (threadIdx.x < mat.cols) ? mat.data[idx] : mat.data[0];

    // Only thread_0 in block has the correct result
    float block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
    //v sync block
    if (threadIdx.x == 0) {
        sync_reduce = block_max;
    }
    __syncthreads();
    block_max = sync_reduce;
    //^ sync block

    if (threadIdx.x < mat.cols) {
        mat.data[idx] -= block_max;
        mat.data[idx] = expf(mat.data[idx]);
    }

    float thread_sum = (threadIdx.x < mat.cols) ? mat.data[idx] : 0;
    __syncthreads(); // required because of temp_storage reuse
    // Only thread_0 in block has the correct result
    float block_sum = BlockReduce(temp_storage).Reduce(thread_sum, cub::Sum());
    //v sync block
    if (threadIdx.x == 0) {
        sync_reduce = block_sum;
    }
    __syncthreads();
    block_sum = sync_reduce;
    //^ sync block

    if (threadIdx.x < mat.cols) {
        mat.data[idx] /= block_sum;
    }
}

__global__ void cuOneHot(PODMatrix<float> lhs, const PODMatrix<int> y) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = idx / lhs.cols;
    int col = idx % lhs.cols;
    if (idx < lhs.size()) {
        lhs.data[idx] = 0;
        if (col == y.data[row]) {
            lhs.data[idx] = 1.0f;
        }
    }
}

__global__ void cuRelu(PODMatrix<float> mat) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat.size())
        mat.data[idx] = mat.data[idx] > 0 ? mat.data[idx] : 0;
}

__global__ void cuDupRows(PODMatrix<float> dst, PODMatrix<float> srcRow) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we're within bounds
    if (col < dst.cols && row < dst.rows) {
        dst.data[row * dst.cols + col] = srcRow.data[col];
    }
}

__global__ void cuPositiveMask(PODMatrix<float> mat, const PODMatrix<float> mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < mat.size()) {
        if (mask.data[idx] <= 0) {
            mat.data[idx] = 0;
        }
    }

}

template <typename T>
CPUMatrix<T> readIdxXubyte(const std::string& dataFile) {
    // T == float: we are reading data
    // T == int: we are reading labels
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

    uint32_t numImages;
    dataIfstream.read(reinterpret_cast<char*>(&numImages), sizeof(uint32_t));
    numImages = swapEndian(numImages);
    std::cout << "Number of Images: " << numImages << "\n";

    size_t colSize = [&]() {
        // 08 bits, 01 dimensions
        if (magicNumber == 	0x00000801) {
            return size_t(1);
        }
        // 08 bits, 03 dimensions
        else if (magicNumber == 0x00000803) {
            uint32_t imgRow;
            dataIfstream.read(reinterpret_cast<char*>(&imgRow), sizeof(uint32_t));
            imgRow = swapEndian(imgRow);

            uint32_t imgCol;
            dataIfstream.read(reinterpret_cast<char*>(&imgCol), sizeof(uint32_t));
            imgCol = swapEndian(imgCol);

            return size_t(imgRow * imgCol);
        }
        else {
            throw std::runtime_error("wrong magic");
        }
    }();

    // Read data:
    CPUMatrix<T> mat(numImages, colSize);
    for (T* i = mat.data; i < mat.end(); ++i) {
        uint8_t byte;
        dataIfstream.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        if (!dataIfstream) {
            throw std::runtime_error("error reading pos " + std::to_string(std::distance(mat.data, i)));
        }

        *i = T(byte);
        if constexpr (std::is_same_v<T, float>) {
            *i /= 255.f;
        }
    }

    return mat;
}

namespace _testEig {
    // EigenType: EigenMatrix or EigenRowVectorF
    template <typename T>
    EigenMatrix fromGPUMatrix(const GPUMatrix<T>& cup);
    EigenRowVectorf fromCUPVector(const GPUMatrix<float>& cup);
    EigenMatrix relu(const EigenMatrix& x);
    EigenMatrix softmax(const EigenMatrix& x);
    EigenMatrix one_hot(const EigenMatrix& y, int maxVal);
    EigenMatrix dup_rows(const EigenMatrix& x, const EigenRowVectorf& y);
    EigenMatrix positive_mask(const EigenMatrix& data, const EigenMatrix& mask);
    void cmpMat(const EigenMatrix& a, const EigenMatrix& b, float prec = Eigen::NumTraits<float>::dummy_precision());
}

class MLP {

public:
    MLP(GPUMatrix<float>& x, GPUMatrix<int>& y, size_t hiddenSize, int batchSize, float lr, int epochs) {

        size_t inputSize = x.cols;
        auto d_begin = thrust::device_pointer_cast(y.data);
        auto d_end = thrust::device_pointer_cast(y.end());
        auto max_iter = thrust::max_element(d_begin, d_end);
        size_t outputSize = *max_iter + 1;

        std::vector<float>initVec(inputSize * hiddenSize, 0xBADDBADD);
        randSeq(initVec.begin(), initVec.end(), -0.01f, 0.01f);
        weight_1 = GPUMatrix<float>(initVec, inputSize, hiddenSize);

        initVec = std::vector<float>(hiddenSize, 0xBADDBADD);
        randSeq(initVec.begin(), initVec.end(), 0.f, 1.f);
        bias_1 = GPUMatrix<float>(initVec, hiddenSize, 1);

        initVec = std::vector<float>(hiddenSize * outputSize, 0xBADDBADD);
        randSeq(initVec.begin(), initVec.end(), -0.01f, 0.01f);
        weight_2 = GPUMatrix<float>(initVec, hiddenSize, outputSize);

        initVec = std::vector<float>(outputSize, 0xBADDBADD);
        randSeq(initVec.begin(), initVec.end(), 0.f, 1.f);
        bias_2 = GPUMatrix<float>(initVec, outputSize, 1);

        std::cout << "Epoch\tLoss\n";
        for (int epoch = 0; epoch < epochs; ++epoch) {

            TimePoint begin = getTimePoint();
            for (int i = 0; i < x.raiiRows(); i += x.rows) {
                int batchRows = std::min((x.raiiRows() - i), batchSize);
                x.setView(i, batchRows);
                y.setView(i, batchRows);
                forward(x);
                backward(x, y, lr);
            }
            Seconds elapsed = getTimePoint() - begin;
            std::cout << "epoch time (" << epoch << "/" << epochs << "): " << elapsed << std::endl;

#ifdef EVAL_EPOCH
            x.setView(0, x.raiiRows());
            y.setView(0, y.raiiRows());
            evalEpoch(x, y, epoch);
#endif
        }
    }


    /*
    input: all training data
    startRow: this batch begins at startRow
    batchRows: this batch ends at endRow = startRow + batchSize
    */
    INLINING void forward(const GPUMatrix<float>& x) {
#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix batchEig = _testEig::fromGPUMatrix<float>(x);
        EigenMatrix weight_1Eig = _testEig::fromGPUMatrix<float>(weight_1);
        EigenRowVectorf bias_1Eig = _testEig::fromCUPVector(bias_1);
        EigenMatrix z1Eig = (batchEig * weight_1Eig).rowwise() + bias_1Eig;

        EigenMatrix a1Eig = _testEig::relu(z1Eig);

        EigenMatrix weight_2Eig = _testEig::fromGPUMatrix<float>(weight_2);
        EigenRowVectorf bias_2Eig = _testEig::fromCUPVector(bias_2);
        EigenMatrix z2Eig = (a1Eig * weight_2Eig).rowwise() + bias_2Eig;

        EigenMatrix a2Eig = _testEig::softmax(z2Eig);
#endif // COMPARE_MLP_WITH_EIGEN
        //z1.dup_rows(bias_1, x.rows);
        z1.dupRows(bias_1, x.rows);
        z1.gemm(x, weight_1, 1.f, 1.f);

        a1 = z1;
        a1.relu();

        //z2 = (a1 * weight_2) +(rowWise) bias_2;
        //z2.dup_rows(bias_2, x.rows);
        z2.dupRows(bias_2, x.rows);
        z2.gemm(a1, weight_2, 1.f, 1.f);

        a2 = z2;
        a2.softmax();

#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix z1Cmp = _testEig::fromGPUMatrix<float>(z1);
        EigenMatrix a1Cmp = _testEig::fromGPUMatrix<float>(a1);
        EigenMatrix z2Cmp = _testEig::fromGPUMatrix<float>(z2);
        EigenMatrix a2Cmp = _testEig::fromGPUMatrix<float>(a2);

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
    INLINING void backward(const GPUMatrix<float>& batchX, const GPUMatrix<int>& batchY, float lr) {
        float divM = 1.f / batchX.rows;
#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix batchYEig = _testEig::fromGPUMatrix<int>(batchY); // note int->float
        EigenMatrix y_one_hotEig = _testEig::one_hot(batchYEig, a2.cols); // a2.cols = outputSize

        EigenMatrix a2Eig = _testEig::fromGPUMatrix<float>(a2);
        EigenMatrix dL_dz2Eig = a2Eig - y_one_hotEig;

        EigenMatrix a1Eig = _testEig::fromGPUMatrix<float>(a1);
        EigenMatrix dL_dW2Eig = (a1Eig.transpose() * dL_dz2Eig) * divM;

        EigenMatrix weight_2Eig = _testEig::fromGPUMatrix<float>(weight_2);
        weight_2Eig -= lr * dL_dW2Eig;

        EigenMatrix dL_db2Eig = (dL_dz2Eig.colwise().sum() * divM);

        EigenMatrix bias_2Eig = _testEig::fromGPUMatrix<float>(bias_2);
        bias_2Eig -= lr * dL_db2Eig.transpose();

        EigenMatrix dL_da1Eig = dL_dz2Eig * weight_2Eig.transpose();

        EigenMatrix z1Eig = _testEig::fromGPUMatrix<float>(z1);
        EigenMatrix dL_dz1Eig = (dL_da1Eig.array() * (z1Eig.array() > 0).cast<float>()).matrix();

        EigenMatrix batchXEig = _testEig::fromGPUMatrix<float>(batchX);
        EigenMatrix dL_dW1Eig = (batchXEig.transpose() * dL_dz1Eig) * divM;

        EigenMatrix weight_1Eig = _testEig::fromGPUMatrix<float>(weight_1);
        weight_1Eig -= lr * dL_dW1Eig;

        EigenMatrix dL_db1Eig = (dL_dz1Eig.colwise().sum() * divM);

        EigenMatrix bias_1Eig = _testEig::fromGPUMatrix<float>(bias_1);
        bias_1Eig -= lr * dL_db1Eig.transpose();
#endif // COMPARE_MLP_WITH_EIGEN
        dL_dz2.a2Sub1Hot(a2, batchY);
        // 3. Gradients for the second (output) layer:
        dL_dW2.gemm<CUPATrans>(a1, dL_dz2, divM);

        // weight_2 -= lr * dL_dW2
        float alpha = -lr;

        CUBLAS_CHECK(
            cublasSaxpy( // y = y + alpha * x
                CublasHandle::get(),
                dL_dW2.size(), // x.size
                &alpha,			 // alpha
                dL_dW2.data,     // x
                1,				 // incx
                weight_2.data,// y
                1 // incy
            )
        );

        // dL_db2 = dL_dz2.colwise().sum() * divM
        // bias_2 -= lr * dL_db2
        bias_2.computeBias(dL_dz2, divM, lr);

        // 4. Backpropagate to the hidden layer:
        dL_da1.gemm<CUPBTrans>(dL_dz2, weight_2);

        dL_dz1 = dL_da1;
        dL_dz1.positiveMask(z1);

        // 5. Gradients for the first (hidden) layer:
        dL_dW1.gemm<CUPATrans>(batchX, dL_dz1, divM);

        // weight_1 -= lr * dL_dW1
        alpha = -lr;
        CUBLAS_CHECK(
            cublasSaxpy( // y = y + alpha * x
                CublasHandle::get(),
                dL_dW1.size(), // x.size
                &alpha,			 // alpha
                dL_dW1.data,// x
                1,				 // incx
                weight_1.data, // y
                1 				 // incy
            )
        );


        // dL_db1 = dL_dz1.colwise().sum() * divM
        bias_1.computeBias(dL_dz1, divM, lr);

#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix dL_dz2Cmp = _testEig::fromGPUMatrix<float>(dL_dz2);
        _testEig::cmpMat(dL_dz2Cmp, dL_dz2Eig);
        EigenMatrix dL_dW2Cmp = _testEig::fromGPUMatrix<float>(dL_dW2);
        _testEig::cmpMat(dL_dW2Cmp, dL_dW2Eig);
        EigenMatrix weight_2Cmp = _testEig::fromGPUMatrix<float>(weight_2);
        _testEig::cmpMat(weight_2Cmp, weight_2Eig);
        EigenMatrix bias_2Cmp = _testEig::fromGPUMatrix<float>(bias_2);
        _testEig::cmpMat(bias_2Cmp, bias_2Eig, 1e-3f);
        EigenMatrix dL_da1Cmp = _testEig::fromGPUMatrix<float>(dL_da1);
        _testEig::cmpMat(dL_da1Cmp, dL_da1Eig);
        EigenMatrix dL_dz1Cmp = _testEig::fromGPUMatrix<float>(dL_dz1);
        _testEig::cmpMat(dL_dz1Cmp, dL_dz1Eig);
        EigenMatrix dL_dW1Cmp = _testEig::fromGPUMatrix<float>(dL_dW1);
        _testEig::cmpMat(dL_dW1Cmp, dL_dW1Eig);
        EigenMatrix weight_1Cmp = _testEig::fromGPUMatrix<float>(weight_1);
        _testEig::cmpMat(weight_1Cmp, weight_1Eig);
        EigenMatrix bias_1Cmp = _testEig::fromGPUMatrix<float>(bias_1);
        _testEig::cmpMat(bias_1Cmp, bias_1Eig, 1e-3f);
#endif
    }



    void evalEpoch(const GPUMatrix<float>& x, const GPUMatrix<int>& y, int epoch) {
        std::cout << "evalEpoch not implemented\n";
        forward(x); // output is in `a2` member var

        float epsilon = 1.0e-6F;
        auto a2ptr = thrust::device_pointer_cast(a2.data);
        EigenMatrix yEig = _testEig::fromGPUMatrix<int>(y);
        assert(yEig.cols() == 1);
        double epoch_lossEig = 0.0;
        for (size_t row = 0; row < a2.rows; ++row) {
            auto elptr = a2ptr + row * a2.cols + yEig(row, 0);
            assert(elptr.get() < a2.end());
            float prob = *elptr;
            prob += epsilon; // against log(0)
            epoch_lossEig += std::log(prob);
        }
        epoch_lossEig /= -a2.rows;
        std::cout << (epoch + 1) << "\t" << epoch_lossEig << std::endl;

    }

    std::vector<int> predictVect(const GPUMatrix<float>& testX) {
        forward(testX);

        std::vector<int> predictions(testX.rows, 0xBADDBADD);
        EigenMatrix a2Eig = _testEig::fromGPUMatrix<float>(a2);
        for (int i = 0; i < a2Eig.rows(); ++i) {
            int maxIndex;
            a2Eig.row(i).maxCoeff(&maxIndex);
            predictions[i] = maxIndex;
        }

        return predictions;
    }

    void predict(const float* sample, float* result) {
        GPUMatrix<float> testX{ 1, 28 * 28, sample };
        forward(testX);

        assert(a2.cols == 10 && a2.rows == 1);

        int bytes = sizeof(float) * a2.cols;
        CU_CHECK(cudaMemcpy(result, a2.raii.ptr, bytes, cudaMemcpyDeviceToHost));
    }

    GPUMatrix<float> weight_1; //< dim [inputSize x hiddenSize]
    GPUMatrix<float> bias_1;   //< dim [1 x hiddenSize]
    GPUMatrix<float> weight_2; //< dim [hiddenSize x outputSize]
    GPUMatrix<float> bias_2;   //< dim [1 x outputSize]

    GPUMatrix<float> z1; //< dim [batchSize x hiddenSize]
    GPUMatrix<float> z2; //< dim [batchSize x outputSize]
    GPUMatrix<float> a1; //< dim [batchSize x hiddenSize]
    GPUMatrix<float> a2; //< dim [batchSize x outputSize]

    // temp matrices
    GPUMatrix<float> dL_dz2;
    GPUMatrix<float> dL_dW2; // dim [hiddenSize x outputSize]
    GPUMatrix<float> dL_da1;
    GPUMatrix<float> dL_dz1;
    GPUMatrix<float> dL_dW1;
    GPUMatrix<float> ones;

    std::vector<float> losses;
};


/* Eigen implementations for comparisons with my implementations */
namespace _testEig {
    void printEig(const EigenMatrix& mat, const std::string& name) {
        int printPrecision = 3;
        Eigen::IOFormat fmt(printPrecision, 0, ", ", "\n", "[", "]"); // up printPrecision if diff hard to spot
        std::cerr << name << std::endl << mat.format(fmt) << std::endl;
    }

    using ::EigenMatrix;
    template <typename T>
    EigenMatrix fromGPUMatrix(const GPUMatrix<T>& cup) {
        std::vector<T> cpuVec = cup.cpyFromDevice();
        EigenMatrix eig = EigenMatrix(cup.rows, cup.cols);
        for (int row = 0; row < cup.rows; ++row) {
            for (int col = 0; col < cup.cols; ++col) {
                eig(row, col) = float(cpuVec[row * cup.cols + col]);
            }

        }
        return eig;
    }

    EigenRowVectorf fromCUPVector(const GPUMatrix<float>& cup) {
        std::vector<float> cpuVec = cup.cpyFromDevice();
        EigenMatrix eig = EigenRowVectorf(cup.rows);
        for (int row = 0; row < cup.rows; ++row) {
            eig(row) = cpuVec[row];
        }
        return eig;
    }

    EigenMatrix relu(const EigenMatrix& x) {
        return x.cwiseMax(0.0);
    }

    EigenMatrix softmax(const EigenMatrix& x) {
        EigenMatrix rowMax = x.rowwise().maxCoeff();
        EigenMatrix x_stable = x - rowMax.replicate(1, x.cols());
        EigenMatrix exp_x = x_stable.array().exp();
        EigenVectorf rowSum = exp_x.rowwise().sum();
        EigenMatrix sm = exp_x.array().colwise() / rowSum.array();
        return sm;
    }

    EigenMatrix one_hot(const EigenMatrix& y, int maxVal) {
        assert(y.cols() == 1);
        EigenMatrix y_one_hot = EigenMatrix(y.rows(), maxVal);
        y_one_hot.setZero();
        for (int i = 0; i < y_one_hot.rows(); ++i) {
            int label = y(i);
            y_one_hot(i, label) = 1.0f;
        }
        return y_one_hot;
    }

    EigenMatrix dup_rows(const EigenMatrix& x, const EigenMatrix& y) {
        assert(x.cols() == y.rows() && y.cols() == 1); // we are assuming a rowvector here!
        return x.rowwise() + EigenRowVectorf(y.transpose());
    }

    EigenMatrix positive_mask(const EigenMatrix& data, const EigenMatrix& mask) {
        return (data.array() * (mask.array() > 0).cast<float>()).matrix();
    }

    void cmpMat(const EigenMatrix& a, const EigenMatrix& b, float prec) {
        if (a.isApprox(b, prec)) {
            //std::cout << "Test passed: matrices are equal." << std::endl;
        }
        else {
            std::cerr << "Test failed: matrices differ." << std::endl;
            printEig(a, "Matrix A");
            printEig(b, "Matrix B");
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

void test_gemm(int m, int n, int k) {
    // Test 1: MlpNoneTrans with multi-dim aMatrix
    {
        //std::cout << m << " " << n << " " << k << "\n";
        GPUMatrix<float> mata = GPUMatrix<float>::Random(m, k * 8, -5.f, 5.f);
        GPUMatrix<float> matb = GPUMatrix<float>::Random(k * 8, n * 8, -5.f, 5.f);
        GPUMatrix<float> matc = GPUMatrix<float>::Random(m, n * 8, -5.f, 5.f);
        matc.gemm<CUPNoneTrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromGPUMatrix<float>(mata);
        EigenMatrix eigB = _testEig::fromGPUMatrix<float>(matb);
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }

    // Test 3: MlpATrans
    {
        GPUMatrix<float> mata = GPUMatrix<float>::Random(k, m * 8, -5.f, 5.f);
        GPUMatrix<float> matb = GPUMatrix<float>::Random(k, n * 8, -5.f, 5.f);
        GPUMatrix<float> matc = GPUMatrix<float>::Random(m * 8, n * 8, -5.f, 5.f);
        matc.gemm<CUPATrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromGPUMatrix<float>(mata).transpose();
        EigenMatrix eigB = _testEig::fromGPUMatrix<float>(matb);
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }

    // Test 5: MlpBTrans with multi-dim aMatrix
    {
        GPUMatrix<float> mata = GPUMatrix<float>::Random(m, k * 8, -5.f, 5.f);
        GPUMatrix<float> matb = GPUMatrix<float>::Random(n * 8, k * 8, -5.f, 5.f);
        GPUMatrix<float> matc = GPUMatrix<float>::Random(m, n * 8, -5.f, 5.f);
        matc.gemm<CUPBTrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromGPUMatrix<float>(mata);
        EigenMatrix eigB = _testEig::fromGPUMatrix<float>(matb).transpose();
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }

    // Test 7: (MlpATrans | MlpBTrans) with multi-dim aMatrix
    {
        GPUMatrix<float> mata = GPUMatrix<float>::Random(k * 8, m * 8, -5.f, 5.f);
        GPUMatrix<float> matb = GPUMatrix<float>::Random(n * 8, k * 8, -5.f, 5.f);
        GPUMatrix<float> matc = GPUMatrix<float>::Random(m * 8, n * 8, -5.f, 5.f);
        matc.gemm<CUPABTrans>(mata, matb, 1.f);

        EigenMatrix eigA = _testEig::fromGPUMatrix<float>(mata).transpose();
        EigenMatrix eigB = _testEig::fromGPUMatrix<float>(matb).transpose();
        EigenMatrix eigCmp = eigA * eigB;
        EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(matc);

        _testEig::cmpMat(eigCmp, mlpCmp);
    }
}

void test_relu(int m, int n) {
    n *= 8;
    GPUMatrix<float> mlp = GPUMatrix<float>::Random(m, n * 8, -5.f, 5.f);
    EigenMatrix eig = _testEig::fromGPUMatrix<float>(mlp);
    mlp.relu();
    EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(mlp);
    EigenMatrix eigCmp = _testEig::relu(eig);

    _testEig::cmpMat(eigCmp, mlpCmp);
}
void test_softmax(int m, int n) {

    // cuda softmax is hardcoded for now. todo unhardcode
    int HARDCODED_COLS = 10;
    GPUMatrix<float> mlp = GPUMatrix<float>::Random(m, HARDCODED_COLS, -5.f, 5.f);
    EigenMatrix eig = _testEig::fromGPUMatrix<float>(mlp);

    mlp.softmax();
    EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(mlp);
    EigenMatrix eigCmp = _testEig::softmax(eig);

    _testEig::cmpMat(eigCmp, mlpCmp);
}
void test_one_hot(int m, int n) {

    n *= 8;
    int outputSize = n * 8; // ensure one_hot produces correct cols
    GPUMatrix<float> mlp = GPUMatrix<float>(m * 8, outputSize);
    EigenMatrix eig = EigenMatrix(m * 8, outputSize);

    GPUMatrix<int> y = GPUMatrix<int>::Random(m * 8, 1, 0, outputSize - 1);
    EigenMatrix yEig = _testEig::fromGPUMatrix<int>(y);
    mlp.oneHot(y, outputSize);
    EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(mlp);
    EigenMatrix eigCmp = _testEig::one_hot(yEig, outputSize);

    _testEig::cmpMat(eigCmp, mlpCmp);
}
void test_dup_rows(int m, int n) {
    n *= 8;
    GPUMatrix<float> mlp = GPUMatrix<float>::Random(m, n * 8, 0.f, 0.f);
    EigenMatrix eig = _testEig::fromGPUMatrix<float>(mlp);

    GPUMatrix<float> row = GPUMatrix<float>::Random(n * 8, 1, -5.f, 5.f);
    EigenMatrix eigRow = _testEig::fromGPUMatrix<float>(row);
    mlp.dupRows(row, mlp.rows);
    EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(mlp);
    EigenMatrix eigCmp = _testEig::dup_rows(eig, eigRow);

    _testEig::cmpMat(eigCmp, mlpCmp);

}

void test_positive_mask(int m, int n) {
    n *= 8;
    GPUMatrix<float> mlp = GPUMatrix<float>::Random(m, n * 8, -5.f, 5.f);
    GPUMatrix<float> mlpMask = GPUMatrix<float>::Random(m, n * 8, -5.f, 5.f);
    EigenMatrix eig = _testEig::fromGPUMatrix<float>(mlp);
    EigenMatrix eigMask = _testEig::fromGPUMatrix<float>(mlpMask);

    mlp.positiveMask(mlpMask);
    EigenMatrix mlpCmp = _testEig::fromGPUMatrix<float>(mlp);
    EigenMatrix eigCmp = _testEig::positive_mask(eig, eigMask);

    _testEig::cmpMat(eigCmp, mlpCmp);
}

void test_raii() {
    GPUMatrix<int> a{ 5,5, 1 };
    a = GPUMatrix<int>{ 6,6, 2 }; // expanding
    a = GPUMatrix<int>{ 4,4, 3 }; // shrinking

    GPUMatrix<int> b;
    b = a;
    assert(b.data != a.data); // deep copy (assignment)
    GPUMatrix<int> copyCtor(a);
    assert(copyCtor.data != a.data); // deep copy (copy ctor)
    PODMatrix<int> pod = a.getPod();
    assert(pod.data == a.data); // weakref copy

    a.setView(1, 2);
    GPUMatrix<int> copyView1 = a;
    assert(copyView1.getRowOffset() == 1);
    assert(copyView1.rows == 2);
    assert(copyView1.raiiRows() == 4);

    GPUMatrix<int> copyView2(copyView1);
    assert(copyView2.getRowOffset() == 1);
    assert(copyView2.rows == 2);
    assert(copyView2.raiiRows() == 4);

    copyView2.setView(0, copyView2.raiiRows());
    auto dataPtr = thrust::device_pointer_cast(copyView2.data);
    assert(*dataPtr == 3);
    auto endPtr = thrust::device_pointer_cast(copyView2.end() - 1);
    assert(*endPtr == 3);
}

void test_colwiseSum(int m, int n, GPUMatrix<float>& ones) {
    GPUMatrix<float> b = GPUMatrix<float>::Random(m, n, -100.f, 100.f);
    EigenMatrix bEig = _testEig::fromGPUMatrix(b);
    GPUMatrix<float> bSum;
    float fAlpha = 1.2f;

    bSum.colwiseSumAlpha(b, ones, fAlpha);
    EigenMatrix bSumEig = (bEig.colwise().sum() * fAlpha);

    assert(bSum.rows == 1 && bSum.cols == b.cols);

    EigenMatrix bSumCmp = _testEig::fromGPUMatrix(bSum);
    _testEig::cmpMat(bSumEig, bSumCmp);

}

void testRun() {
    test_raii();
    std::vector<int> in, out;
    GPUMatrix<float> ones;
    in = { 1,2,3,4,5 };
    out = std::vector<int>(2, 0);
    while (nextPermute(in, out)) {
        int m = out[0];
        int n = out[1];
        test_colwiseSum(m, n, ones);
        test_softmax(m, n);
        test_relu(m, n);
        test_one_hot(m, n);
        test_dup_rows(m, n);
        test_positive_mask(m, n);
    }
    in = { 1,2,3,4,5 };
    out = std::vector<int>(3, 0);
    while (nextPermute(in, out)) {
        int m = out[0];
        int n = out[1];
        int k = out[2];
        test_gemm(m, n, k);
    }
    std::cout << "Passed all unit tests\n";
}

/************************************
**** API DEFINITIONS
*************************************/

void* init() {
	CPUMatrix<float> x = readIdxXubyte<float>("assets.ignored/train-images.idx3-ubyte");
	GPUMatrix<float> xGpu = GPUMatrix<float>(x);
	CPUMatrix<int> y = readIdxXubyte<int>("assets.ignored/train-labels.idx1-ubyte");
	GPUMatrix<int> yGpu = GPUMatrix<int>(y);
	constexpr size_t hiddenSize = 128;

	BenchId trainBench = cppBench("Train MLP");
	void* mlpAsToken = new MLP{ xGpu, yGpu, hiddenSize, BATCH_SIZE, LR, EPOCHS };
	cppBenchEnd(trainBench);

	return mlpAsToken;
}

void destroy(void* hndl) {
	delete reinterpret_cast<MLP*>(hndl);
}

/// output size is 10; sample size is one image
void predict(void* hndl, const float* sample, float* output) {
	reinterpret_cast<MLP*>(hndl)->predict(sample, output);
}

int main() {
    enableFpExcept();
    testRun();

    void* hndl = init();

    BenchId testBench = cppBench("Iterate Test Images");
    CPUMatrix<float> testX = readIdxXubyte<float>("assets.ignored/t10k-images.idx3-ubyte");
    float output[10];
    float numCorrect = 0;
	CPUMatrix<int> testY = readIdxXubyte<int>("assets.ignored/t10k-labels.idx1-ubyte");
    for (size_t i = 0; i < testX.rows; ++i) {
        const float* imagePtr = testX.data + testX.getIdx(i, 0);
		predict(hndl, imagePtr, output);

        // check accuracy
		const auto maxIter = std::max_element(output, output + std::size(output));
        const size_t ourPrediction = std::distance(std::begin(output), maxIter);
        const size_t correctPrediction = *(testY.data + testY.getIdx(i, 0));
        if (ourPrediction == correctPrediction) {
            ++numCorrect;
        }
    }
    cppBenchEnd(testBench);
    std::cout << "Test Accuracy: " << numCorrect / float(testY.size()) << std::endl;
    
    destroy(hndl);
    CublasHandle::free();
    cppBenchPrint();

    return 0;
}


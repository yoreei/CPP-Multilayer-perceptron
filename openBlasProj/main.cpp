#define _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#define _SILENCE_ALL_CXX23_DEPRECATION_WARNINGS
#include "Eigen/Dense"
#undef _SILENCE_CXX23_DENORM_DEPRECATION_WARNING
#undef _SILENCE_ALL_CXX23_DEPRECATION_WARNINGS


#include <iostream>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <vector>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <cassert>


#define SLEEF_STATIC_LIBS
#include "sleef/sleef.h"
#include <openblas/cblas.h>

// define to compare my implementation with an eigen implementation (known to work) (slow)
#undef COMPARE_MLP_WITH_EIGEN
#define COMPARE_EVALEPOCH_MLP_WITH_EIGEN

#include "simdUtil.h"

using EigenMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenRowVectorf = Eigen::RowVector<float, Eigen::Dynamic>;
using EigenVectorf = Eigen::Vector<float, Eigen::Dynamic>;


using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

Time getTime() {
    return std::chrono::high_resolution_clock::now();
}
using Milli = std::chrono::duration<double, std::milli>;
using Seconds = std::chrono::duration<double, std::ratio<1>>;

uint32_t swapEndian(uint32_t val) {
    return ((val >> 24) & 0xff) |
        ((val << 8) & 0xff0000) |
        ((val >> 8) & 0xff00) |
        ((val << 24) & 0xff000000);
}


struct Dataset;
template <typename T256>
class MlpVector : private std::vector<T256> {
    static_assert(std::is_same_v<T256, __m256> || std::is_same_v<T256, __m256i>,
        "T256 must be either __m256 or __m256i");

public:
    int gemmRows;
    int gemmOffset = 0;
    using T32 = std::conditional_t<std::is_same_v<T256, __m256i>, int, float>;

    MlpVector() = default;
    MlpVector(size_t size, T32 val = 0) : std::vector<T256>(size / 8, Set1(val)), gemmRows(size) {
        assert(size % 8 == 0);
    }
    //using std::vector<T256>::vector;
    using std::vector<T256>::clear;
    //using std::vector<T256>::resize;

    using std::vector<T256>::begin;
    using std::vector<T256>::cbegin;
    using std::vector<T256>::rbegin;
    using std::vector<T256>::crbegin;
    using std::vector<T256>::end;
    using std::vector<T256>::cend;
    using std::vector<T256>::rend;
    using std::vector<T256>::crend;
    static MlpVector<T256> Random(int size, T32 minVal, T32 maxVal) {
        assert(size % 8 == 0);
        MlpVector<T256> vec(size, static_cast<T32>(0));
        T32* data = vec.data32();
        std::mt19937 rng(std::random_device{}());
        if constexpr (std::is_same_v<T32, float>) {
            std::uniform_real_distribution<float> dist(minVal, maxVal);
            for (size_t i = 0; i < vec.size32(); ++i) {
                data[i] = dist(rng);
            }
        }
        else {
            std::uniform_int_distribution<int> dist(minVal, maxVal);
            for (size_t i = 0; i < vec.size32(); ++i) {
                data[i] = dist(rng);
            }
        }
        return vec;
    }
    size_t size256() const {
        return std::vector<T256>::size();
    }
    size_t size32() const {
        return size256() * 8;
    }
    T256* data256() {
        return std::vector<T256>::data();
    }
    const T256* data256() const {
        return std::vector<T256>::data();
    }
    T32* data32() {
        return reinterpret_cast<T32*>(data256());
    }
    const T32* data32() const {
        return reinterpret_cast<const T32*>(data256());
    }
    T32* end32() {
        return data32() + size32();
    }
    const T32* end32() const {
        return data32() + size32();
    }

    const T256* end256() const {
        return reinterpret_cast<const T256*>(end32());
    }
    T32& at32(size_t i) {
        return *(data32() + i);
    }
    const T32& at32(size_t i) const {
        return *(data32() + i);
    }
    T256& at256(size_t i) {
        return *(data256() + i);
    }
    const T256& at256(size_t i) const {
        return *(data256() + i);
    }
private:
    constexpr T256 Set1(T32 val) {
        static_assert(std::is_same_v<T256, __m256> || std::is_same_v<T256, __m256i>,
            "T256 must be either __m256 or __m256i");

        if constexpr (std::is_same_v<T256, __m256>) {
            return _mm256_set1_ps(val);
        }
        else {
            return _mm256_set1_epi32(val);
        }
    }
};

enum MlpTransMask {
    MlpNoneTrans = 0,
    MlpATrans = 1,
    MlpBTrans = 2,
    MlpABTrans = 3,
};
using real_t = float;
struct MlpMatrix {
    MlpMatrix() = default;
    MlpMatrix(int _rows, int _cols, float val = 0.f) : rows(_rows) {
        assert(_cols % 8 == 0);
        gemmOffset = 0;
        gemmRows = rows;
        cols = pad8(_cols);
        data256.resize((rows * cols) / 8, _mm256_set1_ps(val));
    }

    static MlpMatrix Random(int _rows, int _cols, float minVal, float maxVal) {
        MlpMatrix mat(_rows, _cols);
        seqRan256(mat.data256.data(), mat.end256(), minVal, maxVal);
        return mat;

    }
    std::string toStr() {
        std::string str = "";
        for (int row = gemmOffset; row < gemmOffset + gemmRows; ++row) {
            for (int col = 0; col < cols; ++col) {
                str += std::to_string(*data32(row, col)) + " ";
            }
            str += "\n";
        }
        return str;
    }

    void one_hot(const MlpVector<__m256i>& y) {
        // Usually need to ensure output matrix has cols = outputSize
        // But Mlp constructor already saw to that, since this one_hot is inplace
        assert(rows == gemmRows && gemmRows == y.gemmRows);
        for (auto& e : data256) {
            e = _mm256_setzero_ps();
        }
        for (size_t i = 0; i < y.gemmRows; ++i) {
            int label = y.at32(y.gemmOffset + i);
            *data32(i, label) = 1.0f;
        }
    }

    // C = αA ∗ B ∗ +βC
    template <MlpTransMask transMask = MlpNoneTrans>
    void gemm(const MlpMatrix& aMatrix, const MlpMatrix& bMatrix, float alpha = 1.f, float beta = 0.f) {
        float* C = data32(gemmOffset);
        const float* A = aMatrix.data32(aMatrix.gemmOffset);
        const float* B = bMatrix.data32(bMatrix.gemmOffset);
        int M, N, K;
        int lda, ldb; // pointers to help M/N swapping
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
    void positive_mask(const MlpMatrix& mask) {
        __m256 zeros = _mm256_setzero_ps();
        for (size_t i = 0; i < size256(); ++i) {
            __m256 mask8 = _mm256_cmp_ps(mask.data256[i], zeros, _CMP_GT_OS); // mask > 0 ? 1 : 0
            data256[i] = _mm256_and_ps(data256[i], mask8); // mask == 1 ? elem = elem : elem = 0
        }

    }
    void dup_rows(const MlpVector<__m256>& row) {
        // O(log n) memcpy calls

        if (row.size32() != cols) {
            int _rowSize = row.size32();
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

    size_t size32() const {
        return rows * cols;
    }
    size_t size256() const {
        return (rows * cols) / 8;
    }
    float* data32(size_t row = 0, size_t col = 0) {
        assert(row < rows && col < cols);
        return reinterpret_cast<float*>(data256.data()) + row * cols + col;
    }
    const float* data32(size_t row = 0, size_t col = 0) const {
        assert(row < rows && col < cols);
        return reinterpret_cast<const float*>(data256.data()) + row * cols + col;
    }
    const float* end32() const {
        return reinterpret_cast<const float*>(data256.data() + size256());
    }
    const __m256* end256() const {
        return data256.data() + size256();
    }

    std::vector<__m256> data256; // needs 32-byte alignment because we will be reading it as __m256
    int rows;
    int cols; // number of floats on x axis, i.e. __m256-width * 8
    int gemmOffset = 0; // Change through Dataset::setGemmView. used only in gemm ops
    int gemmRows;   // Change through Dataset::setGemmView. used only in gemm ops

};

void relu(MlpMatrix& x) {
    __m256 zero = _mm256_setzero_ps();
    for (size_t i = 0; i < x.size256(); ++i) {
        x.data256[i] = _mm256_max_ps(x.data256[i], zero);
    }
}

void softmax(MlpMatrix& x) {
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

struct Dataset {
    // Default constructor.
    Dataset() = default;

    // Constructor that reads MNIST data and label files.
    Dataset(const std::string& dataFile, const std::string& labelFile, bool halveData = false) {
        std::cout << "loading " << dataFile << std::endl;
        std::ifstream dataIfstream(dataFile, std::ios::binary);
        if (!dataIfstream) {
            std::cerr << "Unable to open file: " << dataFile << std::endl;
            exit(-1);
        }

        uint32_t magicNumber = 0;
        uint32_t numImages = 0;
        // Read header: magic number, number of images, rows, and columns.
        dataIfstream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        dataIfstream.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
        dataIfstream.read(reinterpret_cast<char*>(&imageRows), sizeof(imageRows));
        dataIfstream.read(reinterpret_cast<char*>(&imageCols), sizeof(imageCols));

        magicNumber = swapEndian(magicNumber);
        numImages = swapEndian(numImages);
        imageRows = swapEndian(imageRows);
        imageCols = swapEndian(imageCols);

        std::cout << "Magic Number: " << magicNumber << "\n";
        std::cout << "Number of Images in file: " << numImages << "\n";
        numImages = (numImages / 8) * 8;
        std::cout << "Number of Images read: " << numImages << "\n";
        std::cout << "Rows: " << imageRows << "\n";
        std::cout << "Columns: " << imageCols << "\n";

        if ((imageRows * imageCols) % 8 != 0) {
            throw std::runtime_error("numCol must be divisible by 8");
        }
        x = MlpMatrix(numImages, imageRows * imageCols);

        //Read data:
        for (auto& e : x.data256) {
            uint64_t bytes8;
            dataIfstream.read(reinterpret_cast<char*>(&bytes8), sizeof(bytes8));
            if (!dataIfstream) {
                std::ptrdiff_t pos = &e - x.data256.data();
                throw std::runtime_error("error reading pos " + pos);
            }


            __m256 floats8 = cvt8epu8_8ps(bytes8);
            floats8 = _mm256_div_ps(floats8, _mm256_set1_ps(255.f));
            e = floats8;
        }

        // labels
        std::cout << "loading " << labelFile << std::endl;
        std::ifstream labelIfstream(labelFile, std::ios::binary);
        if (!labelIfstream) {
            std::cerr << "Unable to open file: " << labelFile << std::endl;
            exit(-1);
        }

        uint32_t numLabels = 0;
        labelIfstream.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
        labelIfstream.read(reinterpret_cast<char*>(&numLabels), sizeof(numLabels));
        magicNumber = swapEndian(magicNumber);
        numLabels = swapEndian(numLabels);

        std::cout << "Magic Number: " << magicNumber << "\n";
        std::cout << "Number of Labels in file: " << numLabels << "\n";
        numLabels = (numLabels / 8) * 8;
        std::cout << "Number of Labels read: " << numLabels << "\n";

        if (numLabels != numImages) {
            std::cerr << "numLabels != numImages" << std::endl;
            exit(-1);
        }

        y = MlpVector<__m256i>(numLabels);
        for (size_t i = 0; i < y.size32(); ++i) {
            char byte;
            labelIfstream.read(&byte, sizeof(char));
            if (!dataIfstream) {
                throw std::runtime_error("error reading bytes");
            }
            y.at32(i) = static_cast<int>(byte);
        }
    }

    void setGemmView(int _gemmOffset, int _gemmRows) {
        x.gemmOffset = _gemmOffset;
        x.gemmRows = _gemmRows;
        y.gemmOffset = _gemmOffset;
        y.gemmRows = _gemmRows;
        if (_gemmOffset + _gemmRows >= x.size32() &&
            _gemmOffset + _gemmRows >= y.size32()) {
            std::runtime_error("wrong offsets");
        }

    }

    // Convert a pixel (float) to a character for visualization.
    char charFromFloat(float f) const {
        if (f > 0.7f) {
            return '#';
        }
        else if (f > 0.4f) {
            return '!';
        }
        else if (f > 0.1f) {
            return '.';
        }
        else if (f >= 0) {
            return ' ';
        }
        else {
            throw std::runtime_error("wrong f");
        }
    }

    // Print out statistics and show the 40th image.
    void statistics() const {
        std::cout << "Data dim [" << x.rows << ", " << x.cols << "]" << std::endl;
        const auto [minVal, maxVal] = std::minmax_element(x.data32(), x.end32());
        float sumVal = std::accumulate(x.data32(), x.end32(), 0.f);
        std::cout << "Data: min = " << *minVal << ", max = " << *maxVal << ", sum = " << sumVal << std::endl;

        std::cout << "Printing 40th image:" << std::endl;
        for (size_t y = 0; y < imageRows; ++y) {
            for (size_t x = 0; x < imageCols; ++x) {
                std::cout << charFromFloat(getPixel(39, y, x));
            }
            std::cout << std::endl;
        }
        std::cout << "Label: " << y.at32(39) << std::endl;
    }

    // Access a pixel value from the flattened image.
    const float& getPixel(uint64_t imgId, uint64_t yPos, uint64_t xPos) const {
        return *(x.data32() + imgId * x.rows * x.cols + yPos * x.cols + xPos);
    }

    uint32_t imageRows = 0, imageCols = 0; //< must be unit32_t to read from file properly!!!
    MlpMatrix x;  // Data matrix (numImages x (numRows*numCols))
    MlpVector<__m256i> y;        // Label vector (numImages x 1)
};

namespace _testEig {
    EigenMatrix fromMlp(const MlpMatrix& mlp);
    EigenRowVectorf fromMlp(const MlpVector<__m256>& mlp);
    EigenMatrix relu(const EigenMatrix& x);
    EigenMatrix softmax(const EigenMatrix& x);
    EigenMatrix one_hot(const MlpVector<__m256i>& y, int maxVal);
    EigenMatrix dup_rows(const EigenMatrix& x, const EigenRowVectorf& y);
    EigenMatrix positive_mask(const EigenMatrix& data, const EigenMatrix& mask);
    void cmpMat(const EigenMatrix& a, const EigenMatrix& b);
}

class MLP {

public:
    /* arg m: mini-batch size */
    MLP(size_t inputSize, size_t hiddenSize, size_t outputSize, int m, float lr) : m(m), lr(lr) {
        outputSize = pad8(outputSize);

        // Initialize weight_1: values in [-1,1] scaled to [-0.01, 0.01]

        weight_1 = MlpMatrix(inputSize, hiddenSize);
        seqRan256(weight_1.data256.data(), weight_1.end256(), -0.01f, 0.01f);

        // Initialize bias_1: values in [0,1)
        bias_1 = MlpVector<__m256>(hiddenSize);
        seqRan256(bias_1.data256(), bias_1.end256(), 0.f, 1.f);

        // Initialize weight_2: values in [-0.01, 0.01]
        weight_2 = MlpMatrix(hiddenSize, outputSize);
        seqRan256(weight_2.data256.data(), weight_2.end256(), -0.01f, 0.01f);

        // Initialize bias_2: values in [0,1)
        bias_2 = MlpVector<__m256>(outputSize);
        seqRan256(bias_2.data256(), bias_2.end256(), 0.f, 1.f);

        setBatchSize(m);

        dL_dW2 = MlpMatrix(hiddenSize, outputSize);
        dL_db2 = MlpMatrix(1, outputSize);
        dL_dW1 = MlpMatrix(inputSize, hiddenSize);
        dL_db1 = MlpMatrix(1, hiddenSize);

        y_one_hot = MlpMatrix(m, outputSize); // dl_da2

        //std::cout << "weight_1 (first 5 rows):\n" << weight_1.topRows(5) << "\n\n";
        //std::cout << "bias_1:\n" << bias_1 << "\n\n";
        //std::cout << "weight_2 (first 5 rows):\n" << weight_2.topRows(5) << "\n\n";
        //std::cout << "reluWeight_2:\n" << relu(weight_2.topRows(5)) << "\n\n";
        //std::cout << "bias_2:\n" << bias_2 << "\n";

    }

    void setBatchSize(int m) {
        const int& hiddenSize = bias_1.size32();
        const int& outputSize = bias_2.size32();
        z1 = MlpMatrix(m, hiddenSize);
        a1 = MlpMatrix(m, hiddenSize);
        z2 = MlpMatrix(m, outputSize);

        dL_da1 = MlpMatrix(m, hiddenSize);
        dL_dz1 = MlpMatrix(m, hiddenSize);
        dL_dz2 = MlpMatrix(m, outputSize);

        a2 = MlpMatrix(m, outputSize);
    }

    /*
    input: all training data
    startRow: this batch begins at startRow
    batchRows: this batch ends at endRow = startRow + batchSize
    */
    void forward(const MlpMatrix& batch) {
#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix batchEig = _testEig::fromMlp(batch);
        EigenMatrix weight_1Eig = _testEig::fromMlp(weight_1);
        EigenRowVectorf bias_1Eig = _testEig::fromMlp(bias_1);
        EigenMatrix z1Eig = (batchEig * weight_1Eig).rowwise() + bias_1Eig;

        EigenMatrix a1Eig = _testEig::relu(z1Eig);

        EigenMatrix weight_2Eig = _testEig::fromMlp(weight_2);
        EigenRowVectorf bias_2Eig = _testEig::fromMlp(bias_2);
        EigenMatrix z2Eig = (a1Eig * weight_2Eig).rowwise() + bias_2Eig;

        EigenMatrix a2Eig = _testEig::softmax(z2Eig);
#endif // COMPARE_MLP_WITH_EIGEN
        z1.dup_rows(bias_1);
        z1.gemm(batch, weight_1, 1.f, 1.f);

        a1 = z1;
        relu(a1);

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
    void backward(const MlpMatrix& batchX, const MlpVector<__m256i>& batchY) {
        float divM = 1.f / batchX.gemmRows;
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

    void evalEpoch(Dataset& trainData, int epoch) {
        float epsilon = 1.0e-6F;
        trainData.setGemmView(0, trainData.x.rows); // max batch
        setBatchSize(trainData.x.rows); // max batch
        forward(trainData.x); // output is in `a2` member var
        __m256 epoch_loss = _mm256_setzero_ps();
        size_t outSize = a2.cols;
        const auto& y = trainData.y;
        int batchSize = a2.rows;
        for (int row = 0; row < batchSize; row += 8) {
            __m256 probs = _mm256_set_ps(
                *a2.data32(row,     y.at32(y.gemmOffset + row)),
                *a2.data32(row + 1, y.at32(y.gemmOffset + row + 1)),
                *a2.data32(row + 2, y.at32(y.gemmOffset + row + 2)),
                *a2.data32(row + 3, y.at32(y.gemmOffset + row + 3)),
                *a2.data32(row + 4, y.at32(y.gemmOffset + row + 4)),
                *a2.data32(row + 5, y.at32(y.gemmOffset + row + 5)),
                *a2.data32(row + 6, y.at32(y.gemmOffset + row + 6)),
                *a2.data32(row + 7, y.at32(y.gemmOffset + row + 7)));
            probs = _mm256_add_ps(probs, _mm256_set1_ps(epsilon)); // against log(0)
            probs = _mm256_log_ps(probs);
            epoch_loss = _mm256_add_ps(epoch_loss, probs);
        }

        float epoch_loss_sum = sum256f(epoch_loss);
        epoch_loss_sum /= -int(batchSize);
        losses.push_back(epoch_loss_sum);

        std::cout << (epoch + 1) << "\t" << epoch_loss_sum << std::endl;



#ifdef COMPARE_EVALEPOCH_MLP_WITH_EIGEN
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

        setBatchSize(m);
    }

    void train(Dataset& trainData, int epochs = 10) {
        std::cout << "Epoch\tLoss\n";
        // batchSize is already set to m
        auto& x = trainData.x;
        auto& y = trainData.y;
        int maxSamples = m * (x.rows / m);
        for (int epoch = 0; epoch < epochs; ++epoch) {

            Time begin = getTime();
            for (int i = 0; i < maxSamples; i += m) {
                trainData.setGemmView(i, m);
                forward(x);
                backward(x, y);
            }
            Seconds elapsed = getTime() - begin;
            std::cout << "epoch time: " << elapsed << std::endl;

            evalEpoch(trainData, epoch);
        }
    }

    MlpVector<__m256i> predict(const MlpMatrix& test) {
        setBatchSize(test.gemmRows);
        forward(test);

        MlpVector<__m256i> predictions(test.gemmRows);
        for (int row = 0; row < predictions.size32(); ++row) {
            const float* start32 = a2.data32(row, 0);
            const float* end32 = start32 + a2.cols;
            const auto maxIter = std::max_element(start32, end32);
            size_t maxIndex = std::distance(start32, maxIter);
            predictions.at32(row) = maxIndex;
        }

#ifdef COMPARE_MLP_WITH_EIGEN
        EigenMatrix a2Eig = _testEig::fromMlp(a2);
        for (int i = 0; i < a2Eig.rows(); ++i) {
            int maxIndex;
            a2Eig.row(i).maxCoeff(&maxIndex);
            std::cout << "predict.at32(" << i << ")=" << predictions.at32(i) << "; a2Eig.row(i).maxCoeff(&maxIndex) = " << maxIndex << "\n";
            if (predictions.at32(i) != maxIndex) {
                std::cerr << "wrong predict\n";
            }
        }
#endif // COMPARE_MLP_WITH_EIGEN


        return predictions;
    }

    float lr;
    int m; //< mini-batch size

    MlpMatrix weight_1; //< dim [inputSize x hiddenSize]
    MlpVector<__m256> bias_1;   //< dim [1 x hiddenSize]
    MlpMatrix weight_2; //< dim [hiddenSize x outputSize]
    MlpVector<__m256> bias_2;   //< dim [1 x outputSize]

    MlpMatrix z1; //< dim [batchSize x hiddenSize]
    MlpMatrix z2; //< dim [batchSize x outputSize]
    MlpMatrix a1; //< dim [batchSize x hiddenSize]
    MlpMatrix a2; //< dim [batchSize x outputSize]
    std::vector<float> losses;

    // temp matrices
    MlpMatrix y_one_hot;
    MlpMatrix dL_dz2;
    MlpMatrix dL_dW2; // dim [hiddenSize x outputSize]
    MlpMatrix dL_db2;
    MlpMatrix dL_da1;
    MlpMatrix dL_dz1;
    MlpMatrix dL_dW1;
    MlpMatrix dL_db1;
};

///////////////////////////////////////////////////////
// TESTEIGEN

/* Eigen implementations for comparisons with my implementations */
namespace _testEig {
    EigenMatrix fromMlp(const MlpMatrix& mlp) {
        EigenMatrix eig = EigenMatrix(mlp.gemmRows, mlp.cols);
        for (int row = 0; row < mlp.gemmRows; ++row) {
            int mlpRow = row + mlp.gemmOffset;
            for (int col = 0; col < mlp.cols; ++col) {
                eig(row, col) = *mlp.data32(mlpRow, col);
            }

        }
        return eig;
    }

    EigenRowVectorf fromMlp(const MlpVector<__m256>& mlp) {
        EigenMatrix eig = EigenRowVectorf(mlp.gemmRows);
        for (int row = 0; row < mlp.gemmRows; ++row) {
            int mlpRow = row + mlp.gemmOffset;
            eig(row) = mlp.at32(mlpRow);
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

    EigenMatrix one_hot(const MlpVector<__m256i>& y, int maxVal) {
        EigenMatrix y_one_hot = EigenMatrix(y.gemmRows, maxVal);
        y_one_hot.setZero();
        for (int i = 0; i < y_one_hot.rows(); ++i) {
            int label = y.at32(y.gemmOffset + i);
            y_one_hot(i, label) = 1.0f;
        }
        return y_one_hot;
    };

    EigenMatrix dup_rows(const EigenMatrix& x, const EigenRowVectorf& y) {
        assert(x.cols() == y.cols());
        return x.rowwise() + y;
    };

    EigenMatrix positive_mask(const EigenMatrix& data, const EigenMatrix& mask) {
        return (data.array() * (mask.array() > 0).cast<real_t>()).matrix();
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
    bool HALVEDATA = false;
    bool DEBUGSTATISTICS = false;

    Dataset trainData = Dataset("assets.ignored/train-images.idx3-ubyte", "assets.ignored/train-labels.idx1-ubyte", HALVEDATA);
    Dataset testData = Dataset("assets.ignored/t10k-images.idx3-ubyte", "assets.ignored/t10k-labels.idx1-ubyte", HALVEDATA);
    if (DEBUGSTATISTICS) {
        testData.statistics();
        trainData.statistics();
    }

    size_t inputSize = trainData.imageRows * trainData.imageCols;
    size_t hiddenSize = 128;
    const auto maxLabel = std::max_element(trainData.y.data32(), trainData.y.end32());
    size_t outputSize = *maxLabel + 1;
    MLP mlp{ inputSize, hiddenSize, outputSize, 64, 0.01f };

    Time begin = getTime();

    int epochs = 80;
    mlp.train(trainData, epochs);
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

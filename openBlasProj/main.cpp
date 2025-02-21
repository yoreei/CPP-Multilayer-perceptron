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

#define SLEEF_STATIC_LIBS
#include "sleef/sleef.h"
#include <openblas/cblas.h>

#include "simdUtil.h"

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


template <typename T256>
class MlpVector : private std::vector<T256> {
    static_assert(std::is_same_v<T256, __m256> || std::is_same_v<T256, __m256i>,
        "T256 must be either __m256 or __m256i");

public:
    using T32 = std::conditional_t<std::is_same_v<T256, __m256i>, int, float>;

    MlpVector() = default;
    MlpVector(size_t size, T32 val = 0) : std::vector<T256>(pad8(size) / 8, Set1(val)) {}
    //using std::vector<T256>::vector;
    using std::vector<T256>::clear;
    using std::vector<T256>::resize;

    using std::vector<T256>::begin;
    using std::vector<T256>::cbegin;
    using std::vector<T256>::rbegin;
    using std::vector<T256>::crbegin;
    using std::vector<T256>::end;
    using std::vector<T256>::cend;
    using std::vector<T256>::rend;
    using std::vector<T256>::crend;
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
    MlpBTrans = 2
};
using real_t = float;
struct MlpMatrix {
    MlpMatrix() = default;
    MlpMatrix(int _rows, int _cols) {
        gemmOffset = 0;
        rows = pad8(_rows);
        gemmRows = rows;
        cols = pad8(_cols);
        data256.resize((rows * cols) / 8, _mm256_set1_ps(0.f));
        size_t DEBUGcapacity = data256.capacity();
        std::cout << "DEBUGcapacity: " << DEBUGcapacity << std::endl;
    }
    void setGemmView(int _gemmOffset, int _gemmRows) {
        gemmOffset = _gemmOffset;
        gemmRows = _gemmRows;
        if (gemmOffset + gemmRows >= rows) {
            std::runtime_error("wrong offsets");
        }
    }

    // C = αA ∗ B ∗ +βC
    template <MlpTransMask transMask = MlpNoneTrans>
    void gemm(const MlpMatrix& aMatrix, const MlpMatrix& bMatrix, float alpha = 1.f, float beta = 0.f) {
        float* C = data32();
        float* A = aMatrix.data32();
        float* B = bMatrix.data32();
        int M, N, K;
        CBLAS_TRANSPOSE aTransOpt;
        CBLAS_TRANSPOSE bTransOpt;
        if constexpr (transMask == MlpNoneTrans) {
            if (cols != B.rows) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.gemmRows;
            K = aMatrix.cols;
            N = bMatrix.cols;
            aTransOpt = CblasNoTrans;
            bTransOpt = CblasNoTrans;
        }
        else if constexpr (transMask == MlpATrans) {
            if (rows != B.rows) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.cols; // A trans!
            K = aMatrix.gemmRows; // A trans!
            N = bMatrix.cols;
            aTransOpt = CblasTrans;
            bTransOpt = CblasNoTrans;
        }
        else if constexpr (transMask == MlpBTrans) {
            if (cols != B.cols) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.gemmRows;
            K = aMatrix.cols;
            N = bMatrix.gemmRows; // B trans!
            aTransOpt = CblasNoTrans;
            bTransOpt = CblasTrans;
        }
        else if constexpr (transMask == (MlpATrans | MlpBTrans) {
            if (rows != B.cols) {
                throw std::runtime_error("wrong dim");
            }
            M = aMatrix.cols; // A trans!
            K = aMatrix.gemmRows; // A trans!
            N = bMatrix.gemmRows; // B trans!
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
            K,	          // lda, leading dimension of A (num col in A)
            B,
            N,    // ldb, leading dimension of B (num col in B)
            beta,             // beta
            C,	  // C (OUT, result)
            N);   // ldc, leading dimension of C (num col in C)
    }

    //v dim(mask) = dim(*this)
    void positive_mask(const MlpMatrix& mask) {
        __m256 zeros = _mm256_setzero_ps();
        for (size_t i = 0; i < size256(); ++i) {
            __m256 mask8 = _mm256_cmp_ps(mask.data256[i], zeros, _CMP_GT_OS); // elem > 0 ? 1 : 0
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
    float* data32() {
        return reinterpret_cast<float*>(data256.data());
    }
    const float* data32() const {
        return reinterpret_cast<const float*>(data256.data());
    }
    const float* end32() const {
        return reinterpret_cast<const float*>(data256.data() + size256());
    }
    const __m256* end256() const {
        return data256.data() + size256();
    }
    float& at32(size_t row, size_t col) {
        return *(reinterpret_cast<float*>(data256.data()) + row * cols + col);
    }
    const float& at32(size_t row, size_t col) const {
        return *(reinterpret_cast<const float*>(data256.data()) + row * cols + col);
    }

    std::vector<__m256> data256; // needs 32-byte alignment because we will be reading it as __m256
    int rows;
    int cols; // number of floats on x axis, i.e. __m256-width * 8
    int gemmOffset = 0; // used only in gemm ops
    int gemmRows;   // used only in gemm ops

};

void relu(MlpMatrix& x) {
    __m256 zero = _mm256_setzero_ps();
    for (size_t i = 0; i < x.size256(); ++i) {
        x.data256[i] = _mm256_max_ps(x.data256[i], zero);
    }
}

void softmax(MlpMatrix& x) {
    for (int row = 0; row < x.rows; ++row) {
        real_t* start32 = &x.at32(row, 0);
        real_t* end32 = start32 + x.cols;
        const real_t rowMax = (*std::max_element(start32, end32));
        __m256 sub = _mm256_set1_ps(rowMax);

        __m256* start256 = &x.data256[row * (x.cols / 8)];
        __m256 sum256 = _mm256_setzero_ps();

        // subtract rowMax, compute exp, accumulate sum256
        for (size_t i = 0; i <= x.cols; ++i) {
            start256[i] = _mm256_sub_ps(start256[i], sub);
            start256[i] = Sleef_expf8_u10avx2(start256[i]);
            sum256 = _mm256_add_ps(sum256, start256[i]);
        }
        float rowSum = sum256f(sum256);
        __m256 rowSum256 = _mm256_set1_ps(rowSum);

        // divide exponential by rowSum
        for (size_t i = 0; i <= x.cols; ++i) {
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
        dataIfstream.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));
        dataIfstream.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

        magicNumber = swapEndian(magicNumber);
        numImages = swapEndian(numImages);
        numRows = swapEndian(numRows);
        numCols = swapEndian(numCols);

        std::cout << "Magic Number: " << magicNumber << "\n";
        std::cout << "Number of Images: " << numImages << "\n";
        std::cout << "Rows: " << numRows << "\n";
        std::cout << "Columns: " << numCols << "\n";

        if ((numRows * numCols) % 8 != 0) {
            throw std::runtime_error("numCol must be divisible by 8");
        }
        x = MlpMatrix(numImages, numRows * numCols);

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
        std::cout << "Number of Labels: " << numLabels << "\n";

        if (numLabels != numImages) {
            std::cerr << "numLabels != numImages" << std::endl;
            exit(-1);
        }

        y.resize(numImages);
        for (size_t i = 0; i < numImages; ++i) {
            char byte;
            labelIfstream.read(&byte, sizeof(char));
            if (!dataIfstream) {
                throw std::runtime_error("error reading bytes");
            }
            y.at32(i) = static_cast<int>(byte);
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
        for (size_t y = 0; y < numRows; ++y) {
            for (size_t x = 0; x < numCols; ++x) {
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

    uint32_t numRows = 0, numCols = 0; //< must be unit32_t to read from file properly!!!
    MlpMatrix x;  // Data matrix (numImages x (numRows*numCols))
    MlpVector<__m256i> y;        // Label vector (numImages x 1)
};

class MLP {
public:
    /* arg m: mini-batch size */
    MLP(size_t inputSize, size_t hiddenSize, size_t outputSize, int m, float lr) : m(m), lr(lr) {

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

        z1 = MlpMatrix(m, hiddenSize);
        a1 = MlpMatrix(m, hiddenSize);
        z2 = MlpMatrix(m, outputSize);
        a2 = MlpMatrix(m, outputSize);

        // init temps

        dL_da1 = MlpMatrix(m, hiddenSize);
        dL_dz1 = MlpMatrix(m, hiddenSize);
        dL_dz2 = MlpMatrix(m, outputSize);
        y_one_hot = MlpMatrix(m, outputSize); // dl_da2

        //std::cout << "weight_1 (first 5 rows):\n" << weight_1.topRows(5) << "\n\n";
        //std::cout << "bias_1:\n" << bias_1 << "\n\n";
        //std::cout << "weight_2 (first 5 rows):\n" << weight_2.topRows(5) << "\n\n";
        //std::cout << "reluWeight_2:\n" << relu(weight_2.topRows(5)) << "\n\n";
        //std::cout << "bias_2:\n" << bias_2 << "\n";

    }

    /*
    input: all training data
    startRow: this batch begins at startRow
    batchRows: this batch ends at endRow = startRow + batchSize
    */
    void forward(const MlpMatrix& input) {

        //z1 = (batch * weight_1) +(rowWise) bias_1;
        z1.dup_rows(bias_1);
        .gemm()
        cblas_sgemm(CblasRowMajor,
            CblasNoTrans,     // A: no transpose
            CblasNoTrans,     // B: no transpose
            batchRows,		  // M (A.rows == C.rows)
            weight_1.cols,    // N (B.cols == C.cols)
            ACols,			  // K (A.cols == B.rows)
            1.0f,             // alpha
            A,
            ACols,	          // lda, leading dimension of A (num col in A)
            weight_1.data32(),// B
            weight_1.cols,    // ldb, leading dimension of B (num col in B)
            0.0f,             // beta
            z1.data32(),	  // C (OUT, result)
            weight_1.cols);   // ldc, leading dimension of C (num col in C)

        a1 = z1;
        relu(a1);

        //z2 = (a1 * weight_2) +(rowWise) bias_2;
        z2.dup_rows(bias_2);
        cblas_sgemm(CblasRowMajor,
            CblasNoTrans,     // A: no transpose
            CblasNoTrans,     // B: no transpose
            a1.rows,          // M (A.rows == C.rows)
            weight_2.cols,    // N (B.cols == C.cols)
            a1.cols,	      // K (A.cols == B.rows)
            1.0f,             // alpha
            a1.data32(),   // A
            a1.cols,          // lda (num col in A)
            weight_2.data32(), // B
            weight_2.cols,    // ldb (num col in B)
            0.0f,             // beta
            z2.data32(),   // C (OUT, result)
            weight_2.cols);   // ldc (num col in C)

        a2 = z2;
        softmax(a2);
    }

    /*
    input: all training data
    y: all labels
    startRow: this batch begins at startRow
    batchRows: this batch ends at endRow = startRow + batchSize
    */
    void backward(const MlpMatrix& input, const MlpVector<__m256i>& y, size_t startRow, size_t batchRows) {
        y_one_hot = MlpMatrix(y_one_hot.rows, y_one_hot.cols); // fill with 0
        for (size_t i = 0; i < batchRows; ++i) {
            int label = y.at32(startRow + i);
            y_one_hot.at32(i, label) = 1.0f;
        }

        // 2. Compute gradient at output layer:
        //dL_dz2.noalias() = a2 - y_one_hot;
        dL_dz2 = a2;
        cblas_saxpy(     // y = y + alpha * x
            a2.size32(), //n
            -1.0f,		 //alpha
            y_one_hot.data32(), //x
            1,			//incx
            dL_dz2.data32(),    //y
            1			//incy
        );

        // 3. Gradients for the second (output) layer:
        //dL_dW2 = (a1.transpose() * dL_dz2) / m;
        cblas_sgemm(CblasRowMajor,  // C = alpha*A*B + beta*C
            CblasTrans,       // A: transpose
            CblasNoTrans,     // B: no transpose
            a1.cols,          // M (A^T.rows == C.rows)
            dL_dz2.cols,      // N (B.cols == C.cols)
            a1.rows,	      // K (A^T.cols == B.rows)
            1.f / batchRows,          // alpha
            a1.data32(),      // A
            a1.rows,          // lda (num col in A^T)
            dL_dz2.data32(),  // B
            dL_dz2.cols,      // ldb (num col in B)
            0.0f,             // beta
            dL_dW2.data32(),  // C (OUT, result)
            dL_dz2.cols);     // ldc (num col in C)

        //weight_2 -= lr * dL_dW2;
        cblas_saxpy( // y = y + alpha * x
            dL_dW2.size32(), // x.size
            -lr,			 // alpha
            dL_dW2.data32(), // x
            1,				 // incx
            weight_2.data32(), // y
            1);				 // incy

        //dL_db2 = dL_dz2.colwise().sum() / m;
        std::vector<float> ones(dL_dz2.rows, 1.f);
        cblas_sgemv(CblasRowMajor, // y = αA ∗ x + βy
            CblasTrans,		// transpose: A
            dL_dz2.cols,	// M (rows in A^T)
            dL_dz2.rows,	// N (cols in A^T)
            1.0f / batchRows,		// α
            dL_dz2.data32(),// A
            dL_dz2.rows,	// lda: A^T row stride
            ones.data(),	// x
            1,              // incx
            0.0f,			// beta
            dL_db2.data32(),// y
            1);				// incy

        //bias_2 -= lr * dL_db2;
        cblas_saxpy(		 // y = y + alpha * x
            dL_db2.size32(), // x.size
            -lr,			 // alpha
            dL_db2.data32(), // x
            1,				 // incx
            bias_2.data32(), // y
            1);				 // incy

        // 4. Backpropagate to the hidden layer:
        //dL_da1.noalias() = dL_dz2 * weight_2.transpose();
        const auto& A = dL_dz2;
        const auto& B = weight_2;
        cblas_sgemm(CblasRowMajor,  // C = alpha*A*B + beta*C
            CblasNoTrans,       // A: no transpose
            CblasTrans,			// B: transpose
            A.rows,		// M (A.rows == C.rows)
            B.rows,		// N (B^T.cols == C.cols)
            A.cols,			// K (A.cols == B.rows)
            1.f,			// alpha
            A.data32(),		// A
            A.cols,			// lda (num col in A)
            B.data32(),	// B
            B.rows,		// ldb (num col in B^T)
            0.0f,				// beta
            dL_da1.data32(),	// C (OUT, result)
            B.rows);		// ldc (num col in C)


        //dL_dz1.noalias() = (dL_da1.array() * (z1.array() > 0).cast<real_t>()).matrix();
        dL_dz1 = dL_da1;
        dL_dz1.positive_mask(z1);

        // 5. Gradients for the first (hidden) layer:
        //dL_dW1 = (X.transpose() * dL_dz1) / m;
        const float* batch32 = &input.at32(startRow, 0);
        cblas_sgemm(CblasRowMajor,  // C = alpha*A*B + beta*C
            CblasTrans,       // A: transpose
            CblasNoTrans,     // B: no transpose
            input.cols,       // M (A^T.rows == C.rows)
            dL_dz1.cols,      // N (B.cols == C.cols)
            batchRows,	      // K (A^T.cols == B.rows)
            1.f / batchRows,          // alpha
            batch32,	      // A
            batchRows,        // lda (num col in A^T)
            dL_dz1.data32(),  // B
            dL_dz1.cols,      // ldb (num col in B)
            0.0f,             // beta
            dL_dW1.data32(),  // C (OUT, result)
            dL_dz1.cols);     // ldc (num col in C)


        //weight_1 -= lr * dL_dW1;
        cblas_saxpy( // y = y + alpha * x
            dL_dW1.size32(), // x.size
            -lr,			 // alpha
            dL_dW1.data32(), // x
            1,				 // incx
            weight_1.data32(), // y
            1);				 // incy


        //dL_db1 = dL_dz1.colwise().sum() / m;
        ones.resize(dL_dz1.rows, 1.f);
        cblas_sgemv(CblasRowMajor, // y = αA ∗ x + βy
            CblasTrans,		// transpose: A
            dL_dz1.cols,	// M (rows in A^T)
            dL_dz1.rows,	// N (cols in A^T)
            1.0f / batchRows,		// α
            dL_dz1.data32(),// A
            dL_dz1.rows,	// lda: A^T row stride
            ones.data(),	// x
            1,              // incx
            0.0f,			// beta
            dL_db1.data32(),// y
            1);				// incy


        //bias_1 -= lr * dL_db1;
        cblas_saxpy(		 // y = y + alpha * x
            dL_db1.size32(), // x.size
            -lr,			 // alpha
            dL_db1.data32(), // x
            1,				 // incx
            bias_1.data32(), // y
            1);				 // incy

    }

    void evalEpoch(const Dataset& trainData, int epoch) {
        forward(trainData.x, 0, trainData.x.rows);
        __m256 epoch_loss = _mm256_setzero_ps();
        size_t outSize = a2.cols;
        const auto& y = trainData.y;
        for (int i = 0; i < y.size32(); i += 8) {
            __m256 probs = _mm256_set_ps(
                a2.at32(i, y.at32(i)),
                a2.at32(i, y.at32(i + 1)),
                a2.at32(i, y.at32(i + 2)),
                a2.at32(i, y.at32(i + 3)),
                a2.at32(i, y.at32(i + 4)),
                a2.at32(i, y.at32(i + 5)),
                a2.at32(i, y.at32(i + 6)),
                a2.at32(i, y.at32(i + 7)));
            probs = _mm256_log_ps(probs);
            epoch_loss = _mm256_add_ps(epoch_loss, probs);
        }

        float epoch_loss_sum = sum256f(epoch_loss);
        epoch_loss_sum /= -int(y.size32());
        losses.push_back(epoch_loss_sum);

        std::cout << (epoch + 1) << "\t" << epoch_loss_sum << std::endl;
    }

    void train(Dataset& trainData, int epochs = 10) {
        std::cout << "Epoch\tLoss\n";
        auto& x = trainData.x;
        auto& y = trainData.y;
        int maxSamples = m * (x.rows / m);
        for (int epoch = 0; epoch < epochs; ++epoch) {

            Time begin = getTime();
            for (int i = 0; i < maxSamples; i += m) {
                x.setGemmView(i, m);
                y.setGemmView(i, m);
                forward(x);
                backward(x, y, i, m);
            }
            Seconds elapsed = getTime() - begin;
            std::cout << "epoch time: " << elapsed << std::endl;

            //evalEpoch(trainData, epoch);
        }
    }

    MlpVector<__m256i> predict(const MlpMatrix& test) {
        forward(test, 0, test.rows);

        MlpVector<__m256i> predictions(test.rows / 8);
        for (int i = 0; i < predictions.size32(); ++i) {
            const real_t* start32 = a2.data32();
            const real_t* end32 = start32 + a2.cols;
            const auto maxIter = std::max_element(start32, end32);
            size_t maxIndex = std::distance(start32, maxIter);
            predictions.at32(i) = maxIndex;
        }

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
    MlpMatrix dL_dW2;
    MlpMatrix dL_db2;
    MlpMatrix dL_da1;
    MlpMatrix dL_dz1;
    MlpMatrix dL_dW1;
    MlpMatrix dL_db1;
};

int main() {
    bool HALVEDATA = false;
    bool DEBUGSTATISTICS = false;

    Dataset trainData = Dataset("assets.ignored/train-images.idx3-ubyte", "assets.ignored/train-labels.idx1-ubyte", HALVEDATA);
    Dataset testData = Dataset("assets.ignored/t10k-images.idx3-ubyte", "assets.ignored/t10k-labels.idx1-ubyte", HALVEDATA);
    if (DEBUGSTATISTICS) {
        testData.statistics();
        trainData.statistics();
    }

    size_t inputSize = trainData.numRows * trainData.numCols;
    size_t hiddenSize = 128;
    const auto maxLabel = std::max_element(trainData.y.data32(), trainData.y.end32());
    size_t outputSize = *maxLabel + 1;
    MLP mlp{ inputSize, hiddenSize, outputSize, 60, 0.01f };

    Time begin = getTime();

    mlp.train(trainData, 8);
    MlpVector<__m256i> predictions = mlp.predict(testData.x);

    //double accuracy = (predictions.array() == testData.labels.array()).cast<double>().mean();
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < predictions.size256(); ++i) {

        __m256i mask = _mm256_cmpeq_epi32(predictions.at256(i), testData.y.at256(i));
        __m256i ones = _mm256_srli_epi32(mask, 31); // >> 31 for each 32-bit element
        acc = _mm256_add_epi32(acc, ones);
    }

    std::cout << "Test Accuracy: " << sum256i(acc) << std::endl;

    Seconds elapsed = getTime() - begin;
    std::cout << "benchmark: " << elapsed;


    return 0;
}

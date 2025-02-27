#pragma once
#include <iostream>
#include <vector>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cublas_v2.h>

#define CUBLAS_CHECK(call) {                                    \
    cublasStatus_t status = call;                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                      \
        std::cout << cublasGetErrorString(status) << " " << __FILE__ << " " << __LINE__ << "\n"; \
        printf("cuBLAS error %s:%d\n", __FILE__, __LINE__);     \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

#define CU_CHECK(val) cu_check( (val), #val, __FILE__, __LINE__ )

const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    // Add other cases as needed.
    default:
        return "Unknown cuBLAS error";
    }
}

void cu_check(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

/* Allocate on Device & Copy data from host to device. Returns device pointer */
template <typename T>
T* cuAllocCpyFromHost(const std::vector<T>& h_vector) {

    T* d_array;
    size_t bytes = sizeof(T) * h_vector.size();
    CU_CHECK(cudaMalloc(&d_array, bytes));
    CU_CHECK(cudaMemcpy(d_array, h_vector.data(), bytes, cudaMemcpyHostToDevice));
    return d_array;

}

/* Copy data from device to host */
template <typename T>
void cuCpyFromDevice(std::vector<T>& h_vector, const T* d_array) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        std::exit(-1);
    }

    size_t bytes = sizeof(T) * h_vector.size();
    CU_CHECK(cudaMemcpy(h_vector.data(), d_array, bytes, cudaMemcpyDeviceToHost));
}

/* Wait for active kernels to complete. Report any errors. Returns time of execution derived from parameters */
double cuHostJoinDevice(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaEventRecord(stop, 0);
    cudaError_t err = cudaEventSynchronize(stop);
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        std::exit(-1);
    }

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds / 1000;
}

void cuStartTimer(cudaEvent_t& start, cudaEvent_t& stop) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
}


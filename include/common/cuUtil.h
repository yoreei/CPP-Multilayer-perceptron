#pragma once
#include <iostream>
#include <vector>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <cublas_v2.h>

#define CHECK_CUBLAS(call) {                                    \
    cublasStatus_t status = call;                               \
    if (status != CUBLAS_STATUS_SUCCESS) {                      \
        printf("cuBLAS error %s:%d\n", __FILE__, __LINE__);     \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

#define CHECK_CUDA(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
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
    CHECK_CUDA(cudaMalloc(&d_array, bytes));
    CHECK_CUDA(cudaMemcpy(d_array, h_vector.data(), bytes, cudaMemcpyHostToDevice));
    return d_array;

}

/* Copy data from device to host */
void cuCpyFromDevice(std::vector<float>& h_vector, float* d_array) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
        std::exit(-1);
    }

    size_t bytes = sizeof(float) * h_vector.size();
    CHECK_CUDA(cudaMemcpy(h_vector.data(), d_array, bytes, cudaMemcpyDeviceToHost));
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


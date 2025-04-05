#pragma once 

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

inline cublasStatus_t LtSgemm(cublasLtHandle_t ltHandle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha, /* host pointer */
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta, /* host pointer */
    float* C,
    int ldc,
    void* workspace,
    size_t workspaceSize) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

    // Create operation descriptor; see cublasLtMatmulDescAttributes_t
    // for details about defaults; here we just set the transforms for
    // A and B.
    status = cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

    // Create matrix descriptors. Not setting any extra attributes.
    status = cublasLtMatrixLayoutCreate(
        &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutCreate(
        &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

    status = cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatrixLayoutSetAttribute(Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;


    // Create preference handle; In general, extra attributes can be
    // used here to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C. However, for simplicity
    // here we assume A,B,C are always well aligned (e.g., directly
    // come from cudaMalloc)
    status = cublasLtMatmulPreferenceCreate(&preference);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

    // We just need the best available heuristic to try and run matmul.
    // There is no guarantee that this will work. For example, if A is
    // badly aligned, you can request more (e.g. 32) algos and try to
    // run them one by one until something works.
    status = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
    if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

    if (returnedResults == 0) {
        status = CUBLAS_STATUS_NOT_SUPPORTED;
        goto CLEANUP;
    }

    status = cublasLtMatmul(ltHandle,
        operationDesc,
        alpha,
        A,
        Adesc,
        B,
        Bdesc,
        beta,
        C,
        Cdesc,
        C,
        Cdesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0);

CLEANUP:
    // Descriptors are no longer needed as all GPU work was already
    // enqueued.
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    return status == CUBLAS_STATUS_SUCCESS ? static_cast<cublasStatus_t>(0) : static_cast<cublasStatus_t>(1);
}

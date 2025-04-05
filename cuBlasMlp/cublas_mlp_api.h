#pragma once

#define CUMLP_EXPORTS
#ifdef CUMLP_EXPORTS
#define CUMLP_API __declspec(dllexport)
#else
#define CUMLP_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

    /// init & train
    CUMLP_API void* cppmlp_init(const char* directory);

    CUMLP_API void cppmlp_destroy(void* hndl);

    /// output size is 10; sample size is one image
    CUMLP_API void cppmlp_predict(void* hndl, const float* sample, float* output);

#ifdef __cplusplus
} // extern "C"
#endif

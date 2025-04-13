#pragma once

#ifdef BUILDING_CUMLP_EXPORTS
#define CUMLP_API __declspec(dllexport)
#else
#define CUMLP_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

	// Error code definitions.
	typedef int CppMlpErrorCode;
	#define CPPMLP_GOOD             0  // Good
	#define CPPMLP_UNKNOWN_ERROR    1  // Unknown error
	#define CPPMLP_WRONG_DIRECTORY  2  // Wrong directory
	#define CPPMLP_WRONG_SAMPLE     3  // Wrong sample
	#define CPPMLP_CUDA_ERROR       4  // CUDA error

	typedef void* CppMlpHndl;

    /// init & train
    CUMLP_API CppMlpErrorCode cppmlp_init(CppMlpHndl* hndl, const char* directory);

    CUMLP_API void cppmlp_destroy(CppMlpHndl hndl);

    /// output size is 10; sample size is one image
    CUMLP_API CppMlpErrorCode cppmlp_predict(CppMlpHndl hndl, const float* sample, float* output);

#ifdef __cplusplus
} // extern "C"
#endif

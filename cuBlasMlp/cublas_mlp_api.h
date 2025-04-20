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
	#define CPPMLP_GOOD             0
	#define CPPMLP_UNKNOWN_ERROR    1  
	#define CPPMLP_CANNOT_OPEN_FILE 2
	#define CPPMLP_BAD_READ			3
	#define CPPMLP_WRONG_SAMPLE     4 
	#define CPPMLP_CUDA_ERROR       5
	#define CPPMLP_WRONG_ARGUMENT   6

	struct CppMlpReadDims {
		int numImages = 0;
		int imageRows = 0; // rows per 1 image
		int imageCols = 0; // cols per 1 image
	};

    CUMLP_API CppMlpErrorCode cppmlp_read_mnist_meta(const char* filename, CppMlpReadDims* dims);

	#define CPPMLP_READTYPE_LABELS 1
	#define CPPMLP_READTYPE_IMAGES 2

	CUMLP_API CppMlpErrorCode cppmlp_read_mnist(const char* filename, void* outputPtr, const int cppmlp_readtype);

	typedef void* CppMlpHndl;

    /// init & train
    CUMLP_API CppMlpErrorCode cppmlp_init(CppMlpHndl* hndl, const char* directory);

    CUMLP_API void cppmlp_destroy(CppMlpHndl hndl);

    /// output size is 10; sample size is one image
    CUMLP_API CppMlpErrorCode cppmlp_predict(CppMlpHndl hndl, const float* sample, float* output);

#ifdef __cplusplus
} // extern "C"
#endif

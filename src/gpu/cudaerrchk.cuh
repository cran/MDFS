#ifndef CUDAERRCHK_CUH
#define CUDAERRCHK_CUH

#include <iostream>

#define CUDA(value) {\
	cudaError_t _m_cudaStat = value;\
	if (_m_cudaStat != cudaSuccess) {\
		std::cerr << "Error " << cudaGetErrorString(_m_cudaStat) << \
		" at line " << __LINE__ << "in file " << __FILE__ << std::endl;\
		exit(1);\
	}\
}

#endif

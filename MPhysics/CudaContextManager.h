#pragma once
#include "Common.h"
#include "Manager.h"

class CCudaContextManager
{
	Manager(CCudaContextManager)

public:
	void initCudaContext();
	void freeCudaContext();

	cublasHandle_t getCublasHandle() const;
	cusparseHandle_t getCusparseHandle() const;

	bool getIsInit() const;

	UInt getMaxThreadNumberEachBlock() const;
	UInt getMaxBlocksPerMultiProcessor() const;
	UInt getMultiProcessorCount() const;
	UInt getLuidDeviceNodeMask() const;

	void fetchPropBlockGridSize1D(UInt vTotalThreadNum, UInt& voBlockSize, UInt& voGridSize, Real vBlockScaleRate = 1.0);

private:
	bool m_IsInit = false;
	cublasHandle_t m_CublasHandle = nullptr;
	cusparseHandle_t m_CusparseHandle = nullptr;

	cudaDeviceProp m_DeviceProp;
};
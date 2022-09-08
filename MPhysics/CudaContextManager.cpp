#include "CudaContextManager.h"

void CCudaContextManager::initCudaContext()
{
	_ASSERT(m_IsInit == false);
	CHECK_CUBLAS(cublasCreate(&m_CublasHandle));
	CHECK_CUSPARSE(cusparseCreate(&m_CusparseHandle));

	CHECK_CUDA(cudaGetDeviceProperties(&m_DeviceProp, 0));

	m_IsInit = true;
}

void CCudaContextManager::freeCudaContext()
{
	_ASSERT(m_IsInit == true);

	CHECK_CUBLAS(cublasDestroy(m_CublasHandle));
	CHECK_CUSPARSE(cusparseDestroy(m_CusparseHandle));
	m_CublasHandle = nullptr;
	m_CusparseHandle = nullptr;

	m_IsInit = false;
}

cublasHandle_t CCudaContextManager::getCublasHandle() const
{
	_ASSERT(m_IsInit == true);
	_ASSERT(m_CublasHandle != nullptr);
	return m_CublasHandle;
}

cusparseHandle_t CCudaContextManager::getCusparseHandle() const
{
	_ASSERT(m_IsInit == true);
	_ASSERT(m_CusparseHandle != nullptr);
	return m_CusparseHandle;
}

bool CCudaContextManager::getIsInit() const
{
	return m_IsInit;
}

UInt CCudaContextManager::getMaxThreadNumberEachBlock() const
{
	_ASSERT(m_IsInit == true);
	return m_DeviceProp.maxThreadsPerBlock;
}

UInt CCudaContextManager::getMaxBlocksPerMultiProcessor() const
{
	_ASSERT(m_IsInit == true);
	return 	m_DeviceProp.maxBlocksPerMultiProcessor;
}

UInt CCudaContextManager::getMultiProcessorCount() const
{
	_ASSERT(m_IsInit == true);
	return m_DeviceProp.multiProcessorCount;
}

UInt CCudaContextManager::getLuidDeviceNodeMask() const
{
	return m_DeviceProp.luidDeviceNodeMask;
}

void CCudaContextManager::fetchPropBlockGridSize1D(UInt vTotalThreadNum, UInt & voBlockSize, UInt & voGridSize, Real vBlockScaleRate)
{
	_ASSERT(m_IsInit == true);
	_ASSERT(vTotalThreadNum != 0);
	Int MaxBlockSize = m_DeviceProp.maxThreadsPerBlock * 0.5* vBlockScaleRate;
	if (vTotalThreadNum < MaxBlockSize)
	{
		voGridSize = 1;
		voBlockSize = vTotalThreadNum;
	}
	else
	{
		voBlockSize = MaxBlockSize;
		if(vTotalThreadNum % MaxBlockSize == 0)
			voGridSize = vTotalThreadNum / MaxBlockSize;
		else
			voGridSize = vTotalThreadNum / MaxBlockSize + 1;
	}
}

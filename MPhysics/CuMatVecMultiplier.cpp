#include "CuMatVecMultiplier.h"
#include "CudaContextManager.h"

void CCuMatVecMultiplier::resetMultiplier()
{
	CHECK_CUDA(cudaFree(m_CuSpMVBuffer));
	

	m_CuSpMVBuffer = nullptr;
	m_CuSpMVBufferSize = 0;
	m_IsInitialized = false;
}
void CCuMatVecMultiplier::mulMatVec(const CCuSparseMatrix& vM, const CCuDenseVector& vV, CCuDenseVector& voMV, Real vAlpha, Real vBeta)
{
	if (!m_IsInitialized)
	{
		__initMultiplier(vM, vV, voMV, vAlpha, vBeta);
	}

	CHECK_CUSPARSE(cusparseSpMV
	(
		CCudaContextManager::getInstance().getCusparseHandle(),
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		&vAlpha,
		vM.getCuSparseMatrixDescr(),
		vV.getCuDenseVectorDescr(),
		&vBeta,
		voMV.getCuDenseVectorDescr(),
		CUDA_REAL_TYPE,
		CUSPARSE_MV_ALG_DEFAULT,
		m_CuSpMVBuffer
	));
	
}

void CCuMatVecMultiplier::__initMultiplier(const CCuSparseMatrix& vM, const CCuDenseVector& vV, CCuDenseVector& voMV, Real vAlpha, Real vBeta)
{
	CHECK_CUSPARSE(cusparseSpMV_bufferSize
	(
		CCudaContextManager::getInstance().getCusparseHandle(),
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		&vAlpha,
		vM.getCuSparseMatrixDescr(),
		vV.getCuDenseVectorDescr(),
		&vBeta,
		voMV.getCuDenseVectorDescr(),
		CUDA_REAL_TYPE,
		CUSPARSE_MV_ALG_DEFAULT,
		&m_CuSpMVBufferSize
	));
	
	CHECK_CUDA(cudaMallocM(&m_CuSpMVBuffer, m_CuSpMVBufferSize));
	
	m_IsInitialized = true;
}
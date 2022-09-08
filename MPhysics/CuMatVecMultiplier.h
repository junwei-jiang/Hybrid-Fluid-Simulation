#pragma once
#include "CuDenseVector.h"
#include "CuSparseMatrix.h"

class CCuMatVecMultiplier
{
public:
	//潜在的问题：Alpha和Beta改变时是否会导致BufferSize的改变。
	void resetMultiplier();
	void mulMatVec(const CCuSparseMatrix& vM, const CCuDenseVector& vV, CCuDenseVector& voMV, Real vAlpha = 1.0, Real vBeta = 0.0);

private:
	void __initMultiplier(const CCuSparseMatrix& vM, const CCuDenseVector& vV, CCuDenseVector& voMV, Real vAlpha, Real vBeta);

	void* m_CuSpMVBuffer = nullptr;
	size_t m_CuSpMVBufferSize = 0;
	bool m_IsInitialized = false;
};

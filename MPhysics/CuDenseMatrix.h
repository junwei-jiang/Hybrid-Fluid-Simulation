#pragma once
#include "Common.h"

class CCuDenseMatrix
{
public:
	CCuDenseMatrix() = default;
	CCuDenseMatrix(const Real* vMatrixValue, Int vNumOfRow, Int vNumOfCol);
	~CCuDenseMatrix();

	cusparseDnMatDescr_t getCuDenseMatrixDescr() const;
	const Real* getConstGPUMatrixValuePtr() const;
	Real* getGPUMatrixValuePtr();
	Int getNumOfRow() const;
	Int getNumOfCol() const;

private:
	cusparseDnMatDescr_t m_CuDenseMatrixDescr = nullptr;
	thrust::device_vector<Real> m_MatrixValue;
	Int m_NumOfRow;
	Int m_NumOfCol;
};

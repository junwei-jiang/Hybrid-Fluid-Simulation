#pragma once
#include "Common.h"
#include "CuDenseMatrix.h"

class CCuSparseMatrix
{
public:
	CCuSparseMatrix() = default;
	CCuSparseMatrix
	(
		const Real* vNonZeroValue,
		const Int* vRowIndices,
		const Int* vColIndices,
		Int vNumOfNonZero, 
		Int vNumOfRow, 
		Int vNumOfCol
	);
	CCuSparseMatrix(const CCuDenseMatrix& vDenseMatrix);
	~CCuSparseMatrix();

	cusparseSpMatDescr_t getCuSparseMatrixDescr() const;
	const Real* getConstMatrixValueGPUPtr() const;
	const Int* getConstRowIndGPUPtr() const;
	const Int* getConstColIndGPUPtr() const;
	Real* getMatrixValueGPUPtr();
	Int* getRowIndGPUPtr();
	Int* getColIndGPUPtr();

private:
	cusparseSpMatDescr_t m_CuSparseMatrixDescr = nullptr;
	thrust::device_vector<Real> m_MatrixValue;
	thrust::device_vector<Int> m_RowIndices;
	thrust::device_vector<Int> m_ColIndices;
	Int m_NumOfNonZero = 0;
	Int m_NumOfRow;
	Int m_NumOfCol;
};

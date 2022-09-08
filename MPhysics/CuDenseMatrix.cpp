#include "CuDenseMatrix.h"
#include "CudaContextManager.h"
#include "ThrustWapper.cuh"

CCuDenseMatrix::CCuDenseMatrix(const Real* vMatrixValue, Int vNumOfRow, Int vNumOfCol)
{
	m_NumOfRow = vNumOfRow;
	m_NumOfCol = vNumOfCol;

	assignDeviceVectorReal(m_MatrixValue, vMatrixValue, vMatrixValue + vNumOfRow * vNumOfCol);

	CHECK_CUSPARSE(cusparseCreateDnMat
	(
		&m_CuDenseMatrixDescr,
		m_NumOfRow,
		m_NumOfCol,
		m_NumOfCol,
		getRawDevicePointerReal(m_MatrixValue),
		CUDA_REAL_TYPE,
		CUSPARSE_ORDER_ROW
	));
}

CCuDenseMatrix::~CCuDenseMatrix()
{
	CHECK_CUSPARSE(cusparseDestroyDnMat(m_CuDenseMatrixDescr));
}

cusparseDnMatDescr_t CCuDenseMatrix::getCuDenseMatrixDescr() const
{
	return m_CuDenseMatrixDescr;
}

const Real* CCuDenseMatrix::getConstGPUMatrixValuePtr() const
{
	return getReadOnlyRawDevicePointer(m_MatrixValue);
}

Real* CCuDenseMatrix::getGPUMatrixValuePtr()
{
	return getRawDevicePointerReal(m_MatrixValue);
}

Int CCuDenseMatrix::getNumOfRow() const
{
	return m_NumOfRow;
}

Int CCuDenseMatrix::getNumOfCol() const
{
	return m_NumOfCol;
}
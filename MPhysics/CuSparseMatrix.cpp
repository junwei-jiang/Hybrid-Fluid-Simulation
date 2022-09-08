#include "CuSparseMatrix.h"
#include "CudaContextManager.h"
#include "ThrustWapper.cuh"

CCuSparseMatrix::CCuSparseMatrix
(
	const Real* vNonZeroValue,
	const Int* vRowIndices,
	const Int* vColIndices,
	Int vNumOfNonZero, 
	Int vNumOfRow, 
	Int vNumOfCol
)
{
	m_NumOfNonZero = vNumOfNonZero;
	m_NumOfRow = vNumOfRow;
	m_NumOfCol = vNumOfCol;
	
	assignDeviceVectorReal(m_MatrixValue, vNonZeroValue, vNonZeroValue + vNumOfNonZero);
	assignDeviceVectorInt(m_RowIndices, vRowIndices, vRowIndices + vNumOfNonZero);
	assignDeviceVectorInt(m_ColIndices, vColIndices, vColIndices + vNumOfNonZero);
	
	CHECK_CUSPARSE(cusparseCreateCoo
	(
		&m_CuSparseMatrixDescr, 
		m_NumOfRow,
		m_NumOfCol,
		m_NumOfNonZero, 
		getRawDevicePointerInt(m_RowIndices), 
		getRawDevicePointerInt(m_ColIndices), 
		getRawDevicePointerReal(m_MatrixValue), 
		CUSPARSE_INDEX_32I, 
		CUSPARSE_INDEX_BASE_ZERO, 
		CUDA_REAL_TYPE
	));	
}

CCuSparseMatrix::CCuSparseMatrix(const CCuDenseMatrix& vDenseMatrix)
{
	void* CuConvertBuffer = nullptr;
	size_t CuConvertBufferSize = 0;

	CHECK_CUSPARSE(cusparseCreateCoo
	(
		&m_CuSparseMatrixDescr,
		vDenseMatrix.getNumOfRow(),
		vDenseMatrix.getNumOfCol(),
		0,
		NULL,
		NULL,
		NULL,
		CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO,
		CUDA_REAL_TYPE
	));

	cusparseDenseToSparse_bufferSize
	(
		CCudaContextManager::getInstance().getCusparseHandle(),
		vDenseMatrix.getCuDenseMatrixDescr(),
		m_CuSparseMatrixDescr,
		CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
		&CuConvertBufferSize
	);

	CHECK_CUDA(cudaMallocM(&CuConvertBuffer, CuConvertBufferSize));

	cusparseDenseToSparse_analysis
	(
		CCudaContextManager::getInstance().getCusparseHandle(),
		vDenseMatrix.getCuDenseMatrixDescr(),
		m_CuSparseMatrixDescr,
		CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
		CuConvertBuffer
	);

	int64_t TempNumRow;
	int64_t TempNumCol;
	int64_t TempNumNnz;

	CHECK_CUSPARSE(cusparseSpMatGetSize(m_CuSparseMatrixDescr, &TempNumRow, &TempNumCol, &TempNumNnz));

	m_NumOfRow = static_cast<Int>(TempNumRow);
	m_NumOfCol = static_cast<Int>(TempNumCol);
	m_NumOfNonZero = static_cast<Int>(TempNumNnz);

	resizeDeviceVector(m_RowIndices, m_NumOfNonZero);
	resizeDeviceVector(m_ColIndices, m_NumOfNonZero);
	resizeDeviceVector(m_MatrixValue, m_NumOfNonZero);

	CHECK_CUSPARSE(cusparseCooSetPointers(m_CuSparseMatrixDescr, getRowIndGPUPtr(), getColIndGPUPtr(), getMatrixValueGPUPtr()));

	cusparseDenseToSparse_convert
	(
		CCudaContextManager::getInstance().getCusparseHandle(),
		vDenseMatrix.getCuDenseMatrixDescr(),
		m_CuSparseMatrixDescr,
		CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
		CuConvertBuffer
	);

	CHECK_CUDA(cudaFree(CuConvertBuffer));
}

CCuSparseMatrix::~CCuSparseMatrix()
{
	CHECK_CUSPARSE(cusparseDestroySpMat(m_CuSparseMatrixDescr));
}

cusparseSpMatDescr_t CCuSparseMatrix::getCuSparseMatrixDescr() const
{
	return m_CuSparseMatrixDescr;
}

const Real* CCuSparseMatrix::getConstMatrixValueGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_MatrixValue);
}

const Int* CCuSparseMatrix::getConstRowIndGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_RowIndices);
}

const Int* CCuSparseMatrix::getConstColIndGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_ColIndices);
}

Real* CCuSparseMatrix::getMatrixValueGPUPtr()
{
	return getRawDevicePointerReal(m_MatrixValue);
}

Int* CCuSparseMatrix::getRowIndGPUPtr()
{
	return getRawDevicePointerInt(m_RowIndices);
}

Int* CCuSparseMatrix::getColIndGPUPtr()
{
	return getRawDevicePointerInt(m_ColIndices);
}
#include "pch.h"
#include "CuSparseMatrix.h"
#include "CudaContextManager.h"
#include "ThrustWapper.cuh"

TEST(CuSparseMatrix, CuSparseMatrix_Constructor_Des) 
{
	CCudaContextManager::getInstance().initCudaContext();

	vector<Real> MatrixValue(9);
	vector<Int> MatrixRowIndices(9);
	vector<Int> MatrixColIndices(9);

	MatrixValue[0] = 1;
	MatrixValue[1] = 4;
	MatrixValue[2] = 2;
	MatrixValue[3] = 3;
	MatrixValue[4] = 5;
	MatrixValue[5] = 7;
	MatrixValue[6] = 8;
	MatrixValue[7] = 9;
	MatrixValue[8] = 6;

	MatrixRowIndices[0] = 0;
	MatrixRowIndices[1] = 0;
	MatrixRowIndices[2] = 1;
	MatrixRowIndices[3] = 1;
	MatrixRowIndices[4] = 2;
	MatrixRowIndices[5] = 2;
	MatrixRowIndices[6] = 2;
	MatrixRowIndices[7] = 3;
	MatrixRowIndices[8] = 3;

	MatrixColIndices[0] = 0;
	MatrixColIndices[1] = 1;
	MatrixColIndices[2] = 1;
	MatrixColIndices[3] = 2;
	MatrixColIndices[4] = 0;
	MatrixColIndices[5] = 3;
	MatrixColIndices[6] = 4;
	MatrixColIndices[7] = 2;
	MatrixColIndices[8] = 4;

	CCuSparseMatrix Matrix = CCuSparseMatrix(MatrixValue.data(), MatrixRowIndices.data(), MatrixColIndices.data(), 9, 4, 5);

	cusparseSpMatDescr_t MatrixDes = Matrix.getCuSparseMatrixDescr();

	int64_t Row;
	int64_t Col;
	int64_t Nnz;
	cusparseIndexType_t CooIndType;
	cusparseIndexBase_t IdxBase;
	cudaDataType ValueType;
	Int* CooRowIndGPUPtr;
	Int* CooColIndGPUPtr;
	Real* CooValuesGPUPtr;
	Int  *MatrixRowIndResultCPUPtr = nullptr;
	Int  *MatrixColIndResultCPUPtr = nullptr;
	Real *MatrixValueResultCPUPtr = nullptr;
	MatrixRowIndResultCPUPtr = (Int*)malloc(9 * sizeof(Int));
	MatrixColIndResultCPUPtr = (Int*)malloc(9 * sizeof(Int));
	MatrixValueResultCPUPtr = (Real*)malloc(9 * sizeof(Real));

	CHECK_CUSPARSE(cusparseCooGet(MatrixDes, &Row, &Col, &Nnz, (void**)&CooRowIndGPUPtr, (void**)&CooColIndGPUPtr, (void**)&CooValuesGPUPtr, &CooIndType, &IdxBase, &ValueType));

	cudaMemcpy(MatrixRowIndResultCPUPtr, CooRowIndGPUPtr, 9 * sizeof(Int), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixColIndResultCPUPtr, CooColIndGPUPtr, 9 * sizeof(Int), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixValueResultCPUPtr, CooValuesGPUPtr, 9 * sizeof(Real), cudaMemcpyDeviceToHost);
	 
	ASSERT_EQ(Row, 4);
	ASSERT_EQ(Col, 5);
	ASSERT_EQ(Nnz, 9);
	ASSERT_EQ(CooIndType, CUSPARSE_INDEX_32I);
	ASSERT_EQ(IdxBase, CUSPARSE_INDEX_BASE_ZERO);
	ASSERT_EQ(ValueType, CUDA_REAL_TYPE);

	ASSERT_EQ(MatrixRowIndResultCPUPtr[0], 0);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[1], 0);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[2], 1);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[3], 1);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[4], 2);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[5], 2);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[6], 2);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[7], 3);
	ASSERT_EQ(MatrixRowIndResultCPUPtr[8], 3);

	ASSERT_EQ(MatrixColIndResultCPUPtr[0], 0);
	ASSERT_EQ(MatrixColIndResultCPUPtr[1], 1);
	ASSERT_EQ(MatrixColIndResultCPUPtr[2], 1);
	ASSERT_EQ(MatrixColIndResultCPUPtr[3], 2);
	ASSERT_EQ(MatrixColIndResultCPUPtr[4], 0);
	ASSERT_EQ(MatrixColIndResultCPUPtr[5], 3);
	ASSERT_EQ(MatrixColIndResultCPUPtr[6], 4);
	ASSERT_EQ(MatrixColIndResultCPUPtr[7], 2);
	ASSERT_EQ(MatrixColIndResultCPUPtr[8], 4);

	ASSERT_EQ(MatrixValueResultCPUPtr[0], 1);
	ASSERT_EQ(MatrixValueResultCPUPtr[1], 4);
	ASSERT_EQ(MatrixValueResultCPUPtr[2], 2);
	ASSERT_EQ(MatrixValueResultCPUPtr[3], 3);
	ASSERT_EQ(MatrixValueResultCPUPtr[4], 5);
	ASSERT_EQ(MatrixValueResultCPUPtr[5], 7);
	ASSERT_EQ(MatrixValueResultCPUPtr[6], 8);
	ASSERT_EQ(MatrixValueResultCPUPtr[7], 9);
	ASSERT_EQ(MatrixValueResultCPUPtr[8], 6);

	CCudaContextManager::getInstance().freeCudaContext();
}
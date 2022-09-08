#include "pch.h"
#include "CuDenseMatrix.h"
#include "CuSparseMatrix.h"
#include "CudaContextManager.h"
#include "ThrustWapper.cuh"

TEST(CuMatrix, CuSparseMatrix_Constructor)
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

	EXPECT_EQ(Row, 4);
	EXPECT_EQ(Col, 5);
	EXPECT_EQ(Nnz, 9);
	EXPECT_EQ(CooIndType, CUSPARSE_INDEX_32I);
	EXPECT_EQ(IdxBase, CUSPARSE_INDEX_BASE_ZERO);
	EXPECT_EQ(ValueType, CUDA_REAL_TYPE);

	EXPECT_EQ(MatrixRowIndResultCPUPtr[0], 0);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[1], 0);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[2], 1);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[3], 1);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[4], 2);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[5], 2);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[6], 2);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[7], 3);
	EXPECT_EQ(MatrixRowIndResultCPUPtr[8], 3);

	EXPECT_EQ(MatrixColIndResultCPUPtr[0], 0);
	EXPECT_EQ(MatrixColIndResultCPUPtr[1], 1);
	EXPECT_EQ(MatrixColIndResultCPUPtr[2], 1);
	EXPECT_EQ(MatrixColIndResultCPUPtr[3], 2);
	EXPECT_EQ(MatrixColIndResultCPUPtr[4], 0);
	EXPECT_EQ(MatrixColIndResultCPUPtr[5], 3);
	EXPECT_EQ(MatrixColIndResultCPUPtr[6], 4);
	EXPECT_EQ(MatrixColIndResultCPUPtr[7], 2);
	EXPECT_EQ(MatrixColIndResultCPUPtr[8], 4);

	EXPECT_EQ(MatrixValueResultCPUPtr[0], 1);
	EXPECT_EQ(MatrixValueResultCPUPtr[1], 4);
	EXPECT_EQ(MatrixValueResultCPUPtr[2], 2);
	EXPECT_EQ(MatrixValueResultCPUPtr[3], 3);
	EXPECT_EQ(MatrixValueResultCPUPtr[4], 5);
	EXPECT_EQ(MatrixValueResultCPUPtr[5], 7);
	EXPECT_EQ(MatrixValueResultCPUPtr[6], 8);
	EXPECT_EQ(MatrixValueResultCPUPtr[7], 9);
	EXPECT_EQ(MatrixValueResultCPUPtr[8], 6);

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuMatrix, CuDenseMatrix_Constructor)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(100, 10, 1);

	vector<Real> MatrixValue(Res.y * Res.x);

	for (int i = 0; i < Res.x; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			MatrixValue[i * Res.y + j] = j;
		}
	}

	CCuDenseMatrix Matrix(MatrixValue.data(), Res.x, Res.y);

	cusparseDnMatDescr_t MatrixDes = Matrix.getCuDenseMatrixDescr();

	int64_t Row;
	int64_t Col;
	int64_t Idx;
	cusparseOrder_t OrderType;
	cudaDataType ValueType;
	Real* MatValuesGPUPtr;
	Real *MatrixValueResultCPUPtr = nullptr;
	MatrixValueResultCPUPtr = (Real*)malloc(Res.y * Res.x * sizeof(Real));

	CHECK_CUSPARSE(cusparseDnMatGet(MatrixDes, &Row, &Col, &Idx, (void**)&MatValuesGPUPtr, &ValueType, &OrderType));

	cudaMemcpy(MatrixValueResultCPUPtr, MatValuesGPUPtr, Res.y * Res.x * sizeof(Real), cudaMemcpyDeviceToHost);

	EXPECT_EQ(Row, Res.x);
	EXPECT_EQ(Col, Res.y);
	EXPECT_EQ(Idx, Res.y);
	EXPECT_EQ(OrderType, CUSPARSE_ORDER_ROW);
	EXPECT_EQ(ValueType, CUDA_REAL_TYPE);

	for (int i = 0; i < Res.x; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			EXPECT_EQ(MatrixValueResultCPUPtr[i * Res.y + j], j);
		}
	}

	EXPECT_EQ(MatrixValueResultCPUPtr[10], 0);
	EXPECT_EQ(MatrixValueResultCPUPtr[11], 1);

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuMatrix, CuDense2Sparse_Convert1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(9, 80, 1);

	vector<Real> MatrixValue(Res.x * Res.y, 0);

	for (int i = 0; i < Res.y; i++)
	{
		for (int j = 0; j < Res.x; j++)
		{
			if (j == 5)
			{
				MatrixValue[i * Res.x + j] = j;
			}
		}
	}

	CCuDenseMatrix Matrix(MatrixValue.data(), Res.y, Res.x);

	CCuSparseMatrix SpMatrix(Matrix);

	cusparseSpMatDescr_t SpMatrixDes = SpMatrix.getCuSparseMatrixDescr();

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
	MatrixRowIndResultCPUPtr = (Int*)malloc(Res.y * sizeof(Int));
	MatrixColIndResultCPUPtr = (Int*)malloc(Res.y * sizeof(Int));
	MatrixValueResultCPUPtr = (Real*)malloc(Res.y * sizeof(Real));

	CHECK_CUSPARSE(cusparseCooGet(SpMatrixDes, &Row, &Col, &Nnz, (void**)&CooRowIndGPUPtr, (void**)&CooColIndGPUPtr, (void**)&CooValuesGPUPtr, &CooIndType, &IdxBase, &ValueType));

	cudaMemcpy(MatrixRowIndResultCPUPtr, CooRowIndGPUPtr, Res.y * sizeof(Int), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixColIndResultCPUPtr, CooColIndGPUPtr, Res.y * sizeof(Int), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixValueResultCPUPtr, CooValuesGPUPtr, Res.y * sizeof(Real), cudaMemcpyDeviceToHost);

	EXPECT_EQ(Row, Res.y);
	EXPECT_EQ(Col, Res.x);
	EXPECT_EQ(Nnz, Res.y);
	EXPECT_EQ(CooIndType, CUSPARSE_INDEX_32I);
	EXPECT_EQ(IdxBase, CUSPARSE_INDEX_BASE_ZERO);
	EXPECT_EQ(ValueType, CUDA_REAL_TYPE);

	for (int i = 0; i < Nnz; i++)
	{
		EXPECT_EQ(MatrixRowIndResultCPUPtr[i], i);
		EXPECT_EQ(MatrixColIndResultCPUPtr[i], 5);
		EXPECT_EQ(MatrixValueResultCPUPtr[i], 5);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuMatrix, CuDense2Sparse_Convert2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(9, 80, 1);

	vector<Real> MatrixValue(Res.x * Res.y, 0);

	int InputNnz = 0;

	for (int i = 0; i < Res.y; i++)
	{
		for (int j = 0; j < Res.x; j++)
		{
			if (j == 5)
			{
				MatrixValue[i * Res.x + j] = i + 1;
				InputNnz++;
			}
			if (j == 0)
			{
				MatrixValue[i * Res.x + j] = 1;
				InputNnz++;
			}
		}
	}

	CCuDenseMatrix Matrix(MatrixValue.data(), Res.y, Res.x);

	CCuSparseMatrix SpMatrix(Matrix);

	cusparseSpMatDescr_t SpMatrixDes = SpMatrix.getCuSparseMatrixDescr();

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
	MatrixRowIndResultCPUPtr = (Int*)malloc(InputNnz * sizeof(Int));
	MatrixColIndResultCPUPtr = (Int*)malloc(InputNnz * sizeof(Int));
	MatrixValueResultCPUPtr = (Real*)malloc(InputNnz * sizeof(Real));

	CHECK_CUSPARSE(cusparseCooGet(SpMatrixDes, &Row, &Col, &Nnz, (void**)&CooRowIndGPUPtr, (void**)&CooColIndGPUPtr, (void**)&CooValuesGPUPtr, &CooIndType, &IdxBase, &ValueType));

	cudaMemcpy(MatrixRowIndResultCPUPtr, CooRowIndGPUPtr, InputNnz * sizeof(Int), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixColIndResultCPUPtr, CooColIndGPUPtr, InputNnz * sizeof(Int), cudaMemcpyDeviceToHost);
	cudaMemcpy(MatrixValueResultCPUPtr, CooValuesGPUPtr, InputNnz * sizeof(Real), cudaMemcpyDeviceToHost);

	EXPECT_EQ(Row, Res.y);
	EXPECT_EQ(Col, Res.x);
	EXPECT_EQ(Nnz, InputNnz);
	EXPECT_EQ(CooIndType, CUSPARSE_INDEX_32I);
	EXPECT_EQ(IdxBase, CUSPARSE_INDEX_BASE_ZERO);
	EXPECT_EQ(ValueType, CUDA_REAL_TYPE);

	for (int i = 0; i < Nnz; i++)
	{
		if (i % 2 == 0)
		{
			EXPECT_EQ(MatrixRowIndResultCPUPtr[i], static_cast<Int>(i / 2));
			EXPECT_EQ(MatrixColIndResultCPUPtr[i], 0);
			EXPECT_EQ(MatrixValueResultCPUPtr[i], 1);
		}
		else
		{
			EXPECT_EQ(MatrixRowIndResultCPUPtr[i], static_cast<Int>(i / 2));
			EXPECT_EQ(MatrixColIndResultCPUPtr[i], 5);
			EXPECT_EQ(MatrixValueResultCPUPtr[i], static_cast<Int>(i / 2) + 1);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
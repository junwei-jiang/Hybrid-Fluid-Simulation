#include "pch.h"
#include "CuMatVecMultiplier.h"
#include "CudaContextManager.h"
#include "ThrustWapper.cuh"

TEST(CuMatVecMultiplier, MV1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Int  *MatrixRowIndCPUPtr = nullptr;
	Int  *MatrixColIndCPUPtr = nullptr;
	Real *MatrixValueCPUPtr = nullptr;
	MatrixRowIndCPUPtr = (Int*)malloc(9 * sizeof(Int));
	MatrixColIndCPUPtr = (Int*)malloc(9 * sizeof(Int));
	MatrixValueCPUPtr = (Real*)malloc(9 * sizeof(Real));

	MatrixValueCPUPtr[0] = 1;
	MatrixValueCPUPtr[1] = 4;
	MatrixValueCPUPtr[2] = 2;
	MatrixValueCPUPtr[3] = 3;
	MatrixValueCPUPtr[4] = 5;
	MatrixValueCPUPtr[5] = 7;
	MatrixValueCPUPtr[6] = 8;
	MatrixValueCPUPtr[7] = 9;
	MatrixValueCPUPtr[8] = 6;

	MatrixRowIndCPUPtr[0] = 0;
	MatrixRowIndCPUPtr[1] = 0;
	MatrixRowIndCPUPtr[2] = 1;
	MatrixRowIndCPUPtr[3] = 1;
	MatrixRowIndCPUPtr[4] = 2;
	MatrixRowIndCPUPtr[5] = 2;
	MatrixRowIndCPUPtr[6] = 2;
	MatrixRowIndCPUPtr[7] = 3;
	MatrixRowIndCPUPtr[8] = 3;

	MatrixColIndCPUPtr[0] = 0;
	MatrixColIndCPUPtr[1] = 1;
	MatrixColIndCPUPtr[2] = 1;
	MatrixColIndCPUPtr[3] = 2;
	MatrixColIndCPUPtr[4] = 0;
	MatrixColIndCPUPtr[5] = 3;
	MatrixColIndCPUPtr[6] = 4;
	MatrixColIndCPUPtr[7] = 2;
	MatrixColIndCPUPtr[8] = 4;

	CCuSparseMatrix MMatrix = CCuSparseMatrix(MatrixValueCPUPtr, MatrixRowIndCPUPtr, MatrixColIndCPUPtr, 9, 4, 5);

	Real *VVectorValueCPUPtr = nullptr;
	VVectorValueCPUPtr = (Real*)malloc(5 * sizeof(Real));

	for (int i = 0; i < 5; i++)
	{
		VVectorValueCPUPtr[i] = i;
	}

	CCuDenseVector VVector = CCuDenseVector(VVectorValueCPUPtr, 5);
	CCuDenseVector MVVector = CCuDenseVector(4);

	Int  *MatrixRowIndResultCPUPtr = nullptr;
	Int  *MatrixColIndResultCPUPtr = nullptr;
	Real *MatrixValueResultCPUPtr  = nullptr;
	MatrixRowIndResultCPUPtr = (Int*)malloc(5 * sizeof(Int));
	MatrixColIndResultCPUPtr = (Int*)malloc(9 * sizeof(Int));
	MatrixValueResultCPUPtr  = (Real*)malloc(9 * sizeof(Real));
	Real *VVectorValueResultCPUPtr  = nullptr;
	Real *MVVectorValueResultCPUPtr = nullptr;
	VVectorValueResultCPUPtr  = (Real*)malloc(5 * sizeof(Real));
	MVVectorValueResultCPUPtr = (Real*)malloc(4 * sizeof(Real));

	CCuMatVecMultiplier Multiplier;
	Multiplier.mulMatVec(MMatrix, VVector, MVVector);

	cudaMemcpy(MVVectorValueResultCPUPtr, getReadOnlyRawDevicePointer(MVVector.getConstVectorValue()), 4 * sizeof(Real), cudaMemcpyDeviceToHost);

	EXPECT_EQ(MVVectorValueResultCPUPtr[0], 4);
	EXPECT_EQ(MVVectorValueResultCPUPtr[1], 8);
	EXPECT_EQ(MVVectorValueResultCPUPtr[2], 53);
	EXPECT_EQ(MVVectorValueResultCPUPtr[3], 42);

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuMatVecMultiplier, MV2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Real *VVectorValueCPUPtr  = nullptr;
	Real *MMatrixValueCPUPtr  = nullptr;
	Int  *MMatrixRowIndCPUPtr = nullptr;
	Int  *MMatrixColIndCPUPtr = nullptr;
	VVectorValueCPUPtr = (Real*)malloc(3 * sizeof(Real));
	MMatrixValueCPUPtr = (Real*)malloc(9 * sizeof(Real));
	MMatrixRowIndCPUPtr = (Int*)malloc(9 * sizeof(Int));
	MMatrixColIndCPUPtr = (Int*)malloc(9 * sizeof(Int));

	for (int i = 0; i < 3; i++)
	{
		VVectorValueCPUPtr[i] = i;
	}
	for (int i = 0; i < 9; i++)
	{
		MMatrixValueCPUPtr[i] = i + 1;
	}
	for (int i = 0; i < 3; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			MMatrixColIndCPUPtr[i * 3 + k] = k;
			MMatrixRowIndCPUPtr[i * 3 + k] = i;
		}
	}

	CCuDenseVector VVector = CCuDenseVector(VVectorValueCPUPtr, 3);
	CCuDenseVector MVVector = CCuDenseVector(3);
	CCuSparseMatrix MMatrix = CCuSparseMatrix(MMatrixValueCPUPtr, MMatrixRowIndCPUPtr, MMatrixColIndCPUPtr, 9, 3, 3);

	Real *VVectorValueResultCPUPtr  = nullptr;
	Real *MVVectorValueResultCPUPtr = nullptr;
	Real *MMatrixValueResultCPUPtr  = nullptr;
	Int  *MMatrixRowIndResultCPUPtr = nullptr;
	Int  *MMatrixColIndResultCPUPtr = nullptr;
	VVectorValueResultCPUPtr  = (Real*)malloc(3 * sizeof(Real));
	MVVectorValueResultCPUPtr = (Real*)malloc(3 * sizeof(Real));
	MMatrixValueResultCPUPtr  = (Real*)malloc(9 * sizeof(Real));
	MMatrixRowIndResultCPUPtr = (Int*)malloc(9 * sizeof(Int));
	MMatrixColIndResultCPUPtr = (Int*)malloc(9 * sizeof(Int));

	cudaMemcpy(VVectorValueResultCPUPtr, getReadOnlyRawDevicePointer(VVector.getConstVectorValue()), 3 * sizeof(Real), cudaMemcpyDeviceToHost);
	cudaMemcpy(MVVectorValueResultCPUPtr, getReadOnlyRawDevicePointer(MVVector.getConstVectorValue()), 3 * sizeof(Real), cudaMemcpyDeviceToHost);
	cudaMemcpy(MMatrixValueResultCPUPtr, MMatrix.getMatrixValueGPUPtr(), 9 * sizeof(Real), cudaMemcpyDeviceToHost);
	cudaMemcpy(MMatrixRowIndResultCPUPtr, MMatrix.getRowIndGPUPtr(), 9 * sizeof(Int), cudaMemcpyDeviceToHost);
	cudaMemcpy(MMatrixColIndResultCPUPtr, MMatrix.getColIndGPUPtr(), 9 * sizeof(Int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 3; i++)
	{
		EXPECT_EQ(VVectorValueResultCPUPtr[i], i);
		EXPECT_EQ(MVVectorValueResultCPUPtr[i], 0);
	}
	for (int i = 0; i < 9; i++)
	{
		EXPECT_EQ(MMatrixValueResultCPUPtr[i], i + 1);
	}

	EXPECT_EQ(MMatrixRowIndResultCPUPtr[0], 0);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[1], 0);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[2], 0);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[3], 1);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[4], 1);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[5], 1);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[6], 2);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[7], 2);
	EXPECT_EQ(MMatrixRowIndResultCPUPtr[8], 2);

	EXPECT_EQ(MMatrixColIndResultCPUPtr[0], 0);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[1], 1);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[2], 2);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[3], 0);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[4], 1);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[5], 2);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[6], 0);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[7], 1);
	EXPECT_EQ(MMatrixColIndResultCPUPtr[8], 2);

	CCuMatVecMultiplier Multiplier;
	Multiplier.mulMatVec(MMatrix, VVector, MVVector);

	cudaMemcpy(MVVectorValueResultCPUPtr, getReadOnlyRawDevicePointer(MVVector.getConstVectorValue()), 3 * sizeof(Real), cudaMemcpyDeviceToHost);
	EXPECT_EQ(MVVectorValueResultCPUPtr[0], 8);
	EXPECT_EQ(MVVectorValueResultCPUPtr[1], 17);
	EXPECT_EQ(MVVectorValueResultCPUPtr[2], 26);

	CCudaContextManager::getInstance().freeCudaContext();
}
#include "pch.h"
#include "CuDenseVector.h"
#include "CudaContextManager.h"
#include "ThrustWapper.cuh"

TEST(CuDenseVector, CuDenseVector_Constructor)
{
	CCudaContextManager::getInstance().initCudaContext();

	Real *VectorDataCPUPtr = nullptr;
	Real *VectorDataResultCPUPtr = nullptr;

	VectorDataCPUPtr = (Real*)malloc(100 * sizeof(Real));
	VectorDataResultCPUPtr = (Real*)malloc(100 * sizeof(Real));

	for (int i = 0; i < 100; i++)
	{
		VectorDataCPUPtr[i] = i;
	}

	CCuDenseVector V1(10);
	CCuDenseVector V2 = CCuDenseVector(10);
	CCuDenseVector V3 = CCuDenseVector(VectorDataCPUPtr, 100);
	CCuDenseVector V4(V3);
	CCuDenseVector* Vector5 = new CCuDenseVector(10);

	cudaMemcpy(VectorDataResultCPUPtr, getReadOnlyRawDevicePointer(V3.getConstVectorValue()), 100 * sizeof(Real), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++)
	{
		ASSERT_EQ(VectorDataResultCPUPtr[i], i);
	}

	cudaMemcpy(VectorDataResultCPUPtr, getReadOnlyRawDevicePointer(V4.getConstVectorValue()), 100 * sizeof(Real), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++)
	{
		ASSERT_EQ(VectorDataResultCPUPtr[i], i);
	}

	free(Vector5);

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuDenseVector, CuDenseVector_Operator)
{
	CCudaContextManager::getInstance().initCudaContext();

	Real *VectorDataCPUPtr1 = nullptr;
	Real *VectorDataCPUPtr2 = nullptr;
	Real *VectorDataResultCPUPtr = nullptr;

	VectorDataCPUPtr1 = (Real*)malloc(100 * sizeof(Real));
	VectorDataCPUPtr2 = (Real*)malloc(100 * sizeof(Real));
	VectorDataResultCPUPtr = (Real*)malloc(100 * sizeof(Real));

	for (int i = 0; i < 100; i++)
	{
		VectorDataCPUPtr1[i] = i;
		VectorDataCPUPtr2[i] = 99 - i;
	}

	CCuDenseVector Vector1 = CCuDenseVector(VectorDataCPUPtr1, 100);
	CCuDenseVector Vector2 = CCuDenseVector(VectorDataCPUPtr2, 100);

	Vector1 += Vector2;

	cudaMemcpy(VectorDataResultCPUPtr, getReadOnlyRawDevicePointer(Vector1.getConstVectorValue()), 100 * sizeof(Real), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++)
	{
		ASSERT_EQ(VectorDataResultCPUPtr[i], 99);
	}

	cudaMemcpy(VectorDataResultCPUPtr, getReadOnlyRawDevicePointer(Vector2.getConstVectorValue()), 100 * sizeof(Real), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++)
	{
		ASSERT_EQ(VectorDataResultCPUPtr[i], 99 - i);
	}

	Vector1 *= 0.1;
	cudaMemcpy(VectorDataResultCPUPtr, getReadOnlyRawDevicePointer(Vector1.getConstVectorValue()), 100 * sizeof(Real), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++)
	{
		ASSERT_LT(abs(VectorDataResultCPUPtr[i] - 9.9), 1e-6);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuDenseVector, CuDenseVector_DotNorm)
{
	CCudaContextManager::getInstance().initCudaContext();

	Real *VectorDataCPUPtr1 = nullptr;
	Real *VectorDataCPUPtr2 = nullptr;
	Real *VectorDataResultCPUPtr = nullptr;
	Real *VectorDataResultGPUPtr = nullptr;

	VectorDataCPUPtr1 = (Real*)malloc(5 * sizeof(Real));
	VectorDataCPUPtr2 = (Real*)malloc(5 * sizeof(Real));
	VectorDataResultCPUPtr = (Real*)malloc(5 * sizeof(Real));

	for (int i = 0; i < 5; i++)
	{
		VectorDataCPUPtr1[i] = i;
		VectorDataCPUPtr2[i] = 4 - i;
	}

	CCuDenseVector Vector1 = CCuDenseVector(VectorDataCPUPtr1, 5);
	CCuDenseVector Vector2 = CCuDenseVector(VectorDataCPUPtr2, 5);
	Real Result;

	Result = Vector1.norm2();

	EXPECT_LT(abs(Result - sqrt(30)), 1e6);

	Result = Vector1.dot(Vector2);

	EXPECT_LT(abs(Result - 10), 1e6);

	Result = Vector1 * Vector2;

	EXPECT_LT(abs(Result - 10), 1e6);

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CuDenseVector, CuDenseVector_Des)
{
	CCudaContextManager::getInstance().initCudaContext();

	Real *VectorDataCPUPtr = nullptr;
	Real *VectorDataResultCPUPtr = nullptr;
	Real *VectorDataResultGPUPtr = nullptr;
	int64_t Size;
	cudaDataType DataType;

	VectorDataCPUPtr = (Real*)malloc(100 * sizeof(Real));
	VectorDataResultCPUPtr = (Real*)malloc(100 * sizeof(Real));

	for (int i = 0; i < 100; i++)
	{
		VectorDataCPUPtr[i] = i;
	}

	CCuDenseVector Vector = CCuDenseVector(VectorDataCPUPtr, 100);

	CHECK_CUSPARSE(cusparseDnVecGet(Vector.getCuDenseVectorDescr(), &Size, (void**)&VectorDataResultGPUPtr, &DataType));

	cudaMemcpy(VectorDataResultCPUPtr, VectorDataResultGPUPtr, 100 * sizeof(Real), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 100; i++)
	{
		ASSERT_EQ(VectorDataResultCPUPtr[i], i);
	}
	ASSERT_EQ(Size, 100);
	ASSERT_EQ(DataType, CUDA_REAL_TYPE);

	CCudaContextManager::getInstance().freeCudaContext();
}
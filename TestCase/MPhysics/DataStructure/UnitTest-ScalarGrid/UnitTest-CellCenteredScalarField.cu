#include "pch.h"
#include "FieldMathTool.cuh"
#include "CellCenteredScalarField.h"
#include "CellCenteredVectorField.h"
#include "FaceCenteredVectorField.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>

void EXPECT_EQ_VECTOR3(const Vector3& vVecA, const Vector3& vVecB)
{
	EXPECT_EQ(vVecA.x, vVecB.x);
	EXPECT_EQ(vVecA.y, vVecB.y);
	EXPECT_EQ(vVecA.z, vVecB.z);
}

void EXPECT_EQ_VECTOR3I(const Vector3i& vVecA, const Vector3i& vVecB)
{
	EXPECT_EQ(vVecA.x, vVecB.x);
	EXPECT_EQ(vVecA.y, vVecB.y);
	EXPECT_EQ(vVecA.z, vVecB.z);
}

//TEST(CellCenteredScalarField, CUDAThreadInfomation)
//{
//	//定义一个cuda的设备属性结构体
//	cudaDeviceProp prop;
//	//获取第1个gpu设备的属性信息
//	cudaGetDeviceProperties(&prop, 0);
//	//每个block的最大线程数
//	std::cout << "maxThreadsPerBlock: " << prop.maxThreadsPerBlock << std::endl;
//	//block的维度
//	for (int i = 0; i < 3; ++i) std::cout << "maxThreadsDim[" << i << "]: " << prop.maxThreadsDim[i] << std::endl;
//	//输出最大的gridSize
//	std::cout << std::endl;
//	for (int i = 0; i < 3; ++i) std::cout << "maxGridSize[" << i << "]: " << prop.maxGridSize[i] << std::endl;
//}

TEST(CellCenteredScalarField, CellCenteredScalarField_Constructor)
{
	CCudaContextManager::getInstance().initCudaContext();

	CCellCenteredScalarField CCSField1(2, 2, 2, -10, 0, 0, 2, 4, 6);
	CCellCenteredScalarField CCSField2(Vector3i(16, 16, 16), Vector3(2, 2, 2), Vector3(8, 8, 8));
	CCellCenteredScalarField CCSField3(CCSField2);

	EXPECT_EQ_VECTOR3I(CCSField1.getResolution(), Vector3i(2, 2, 2));
	EXPECT_EQ_VECTOR3(CCSField1.getOrigin(), Vector3(-10, 0, 0));
	EXPECT_EQ_VECTOR3(CCSField1.getSpacing(), Vector3(2, 4, 6));
	EXPECT_EQ_VECTOR3I(CCSField3.getResolution(), Vector3i(16, 16, 16));
	EXPECT_EQ_VECTOR3(CCSField3.getOrigin(), Vector3(2, 2, 2));
	EXPECT_EQ_VECTOR3(CCSField3.getSpacing(), Vector3(8, 8, 8));

	Vector3i Res = Vector3i(32, 31, 50);
	vector<Real> FieldData(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		FieldData[i] = i;
	}

	CCellCenteredScalarField CCSField4(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FieldData.data());
	CCellCenteredScalarField CCSField5(CCSField4);
	CCellCenteredScalarField CCSField6(Res);
	CCSField6.resize(CCSField5);

	thrust::host_vector<Real> FieldResultData = CCSField5.getConstGridData();

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		EXPECT_LT(abs(FieldResultData[i] - i), GRID_SOLVER_EPSILON);
	}

	FieldResultData = CCSField6.getConstGridData();

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		EXPECT_LT(abs(FieldResultData[i] - i), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredScalarField, CellCenteredScalarField_Assignment)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(16, 16, 16);
	vector<Real> FieldData(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		FieldData[i] = i;
	}

	CCellCenteredScalarField CCSField1(Res, Vector3(2, 2, 2), Vector3(8, 8, 8), FieldData.data());
	CCellCenteredScalarField CCSField2;

	CCSField2 = CCSField1;

	EXPECT_EQ_VECTOR3I(CCSField2.getResolution(), Res);
	EXPECT_EQ_VECTOR3(CCSField2.getOrigin(), Vector3(2, 2, 2));
	EXPECT_EQ_VECTOR3(CCSField2.getSpacing(), Vector3(8, 8, 8));

	thrust::host_vector<Real> FieldResultData = CCSField2.getConstGridData();

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		EXPECT_LT(abs(FieldResultData[i] - i), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredScalarField, CellCenteredScalarField_PlusAndScale)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(32, 10, 16);
	vector<Real> FieldData1(Res.x * Res.y * Res.z);
	vector<Real> FieldData2(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FieldData1[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
				FieldData2[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 2) * 10;
			}
		}
	}

	CCellCenteredScalarField CCSField1(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), FieldData1.data());
	CCellCenteredScalarField CCSField2(Res, Vector3(-20, 40, -10), Vector3(20, 1, 4), FieldData2.data());

	CCSField1 *= 0.1;
	CCSField1.scale(5);
	CCSField1 += CCSField2;

	thrust::host_vector<Real> FieldResultData = CCSField1.getConstGridData();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultData[i * Res.x * Res.y + j * Res.x + k] - (Int)(k % 2) * 5 - (Int)(j % 2) * 10), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCSField1.plusAlphaX(CCSField1, -1);

	FieldResultData = CCSField1.getConstGridData();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultData[i * Res.x * Res.y + j * Res.x + k] - 0), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//Trilinear插值采样
TEST(CellCenteredScalarField, CellCenteredScalarField_CuSamplingTrilinear)
{
	CCudaContextManager::getInstance().initCudaContext();

	//简单测试采样
	{
		Vector3i Res = Vector3i(2, 2, 1);
		Real Size = Res.x * Res.y * Res.z * sizeof(Real);

		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2);
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.4;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.4;
				}
			}
		}

		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());
		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCSSrcField.sampleField(CCVPosField, CCSDstField);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1] - 0.9), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3] - 0.9), GRID_SOLVER_EPSILON);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.6;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.6;
				}
			}
		}

		CCVPosField.resize(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());
		CCSDstField.resize(Res);
		CCSSrcField.sampleField(CCVPosField, CCSDstField);
		DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0] - 0.1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2] - 0.1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3] - 1), GRID_SOLVER_EPSILON);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.5;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.5;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.5;
				}
			}
		}

		CCVPosField.resize(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());
		CCSDstField.resize(Res);
		CCSSrcField.sampleField(CCVPosField, CCSDstField);
		DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3] - 1), GRID_SOLVER_EPSILON);
	}

	//测试网格Origin和Spacing不是默认情况下的采样
	{
		Vector3i Res = Vector3i(16, 16, 16);
		Real Size = Res.x * Res.y * Res.z * sizeof(Real);

		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 5;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 10;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());
		CCellCenteredScalarField CCSSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);

		CCSSrcField.sampleField(CCVPosField, CCSDstField);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res.x * Res.y + j * Res.x + k] - 10 * (Int)(k % 2)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//测试源网格和目的网格分辨率不同的情况下的采样
	{
		Vector3i Res = Vector3i(16, 16, 16);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
				}
			}
		}

		Vector3i Res2 = Vector3i(32, 2, 16);
		vector<Real> PosVectorFieldDataX(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataY(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataZ(Res2.x * Res2.y * Res2.z);

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					PosVectorFieldDataX[i * Res2.x * Res2.y + j * Res2.x + k] = ((Int)(k / 2)) * 10 - 10 + 5;
					PosVectorFieldDataY[i * Res2.x * Res2.y + j * Res2.x + k] = j * 20 - 40 + 10;
					PosVectorFieldDataZ[i * Res2.x * Res2.y + j * Res2.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res2);
		CCellCenteredVectorField CCVPosField(Res2, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(((Int)(k / 2)) % 2)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//测试采样的边界情况
	{
		Vector3i Res = Vector3i(2, 2, 1);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 2);
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.5;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.4;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 8.5;
				}
			}
		}

		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());
		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);

		CCSSrcField.sampleField(CCVPosField, CCSDstField);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2] - 0.9), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3] - 0.9), GRID_SOLVER_EPSILON);
	}

	//测试只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(16, 16, 16);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
				}
			}
		}

		Vector3i Res2 = Vector3i(32, 2, 16);
		vector<Real> PosVectorFieldDataX(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataY(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataZ(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldData(3 * Res2.x * Res2.y * Res2.z);

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					PosVectorFieldDataX[i * Res2.x * Res2.y + j * Res2.x + k] = ((Int)(k / 2)) * 10 - 10 + 5;
					PosVectorFieldDataY[i * Res2.x * Res2.y + j * Res2.x + k] = j * 20 - 40 + 10;
					PosVectorFieldDataZ[i * Res2.x * Res2.y + j * Res2.x + k] = i * 30 - 50 + 15;
					PosVectorFieldData[3 * (i * Res2.x * Res2.y + j * Res2.x + k)] = PosVectorFieldDataX[i * Res2.x * Res2.y + j * Res2.x + k];
					PosVectorFieldData[3 * (i * Res2.x * Res2.y + j * Res2.x + k) + 1] = PosVectorFieldDataY[i * Res2.x * Res2.y + j * Res2.x + k];
					PosVectorFieldData[3 * (i * Res2.x * Res2.y + j * Res2.x + k) + 2] = PosVectorFieldDataZ[i * Res2.x * Res2.y + j * Res2.x + k];
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(Res2.x * Res2.y * Res2.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res2.x * Res2.y * Res2.z);

		CCellCenteredScalarField CCSSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res2);
		CCellCenteredVectorField CCVPosField(Res2, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField);
		CCSSrcField.sampleField(SamplePos, SampleDstValue);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();
		thrust::host_vector<Real> DstFieldResultData2 = SampleDstValue;

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(((Int)(k / 2)) % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(DstFieldResultData2[i * Res2.x * Res2.y + j * Res2.x + k] - DstFieldResultData[i * Res2.x * Res2.y + j * Res2.x + k]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}
	
	CCudaContextManager::getInstance().freeCudaContext();
}

//Catmull-Rom插值采样
TEST(CellCenteredScalarField, CellCenteredScalarField_CuSamplingCatmullRom)
{
	CCudaContextManager::getInstance().initCudaContext();

	//CatmullRom采样点位于网格中心测试
	{
		Vector3i Res = Vector3i(16, 16, 16);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = k * 10 + j * 20 + i * 30;
				}
			}
		}

		Vector3i Res2 = Vector3i(32, 2, 16);
		vector<Real> PosVectorFieldDataX(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataY(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataZ(Res2.x * Res2.y * Res2.z);

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					PosVectorFieldDataX[i * Res2.x * Res2.y + j * Res2.x + k] = ((Int)(k / 2)) * 10 - 10 + 5;
					PosVectorFieldDataY[i * Res2.x * Res2.y + j * Res2.x + k] = j * 20 - 40 + 10;
					PosVectorFieldDataZ[i * Res2.x * Res2.y + j * Res2.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res2);
		CCellCenteredVectorField CCVPosField(Res2, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res2.x * Res2.y + j * Res2.x + k] - (((Int)(k / 2)) * 10 + j * 20 + i * 30)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//CatmullRom简单手算测试，只考虑了一维其实
	{
		Vector3i Res = Vector3i(4, 4, 1);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = i * Res.x * Res.y + j * Res.x + k;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.5;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.5;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0]  +  0.0405), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1]  -  0.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2]  -  1.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3]  -  2.9405), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[4]  -  3.9595), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[5]  -  4.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[6]  -  5.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[7]  -  6.9405), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[8]  -  7.9595), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[9]  -  8.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[10] -  9.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[11] - 10.9405), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[12] - 11.9595), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[13] - 12.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[14] - 13.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[15] - 14.9405), GRID_SOLVER_EPSILON);
	}

	//CatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[22] - 2.4415), GRID_SOLVER_EPSILON);
	}

	//测试CatmullRom只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldData(3 * Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CATMULLROM);
		CCSSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();
		thrust::host_vector<Real> DstFieldResultData2 = SampleDstValue;

		EXPECT_LT(abs(DstFieldResultData[22] - 2.4415), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData2[22] - 2.4415), GRID_SOLVER_EPSILON);
	}

	//MonoCatmullRom简单手算测试，只考虑了一维其实
	{
		Vector3i Res = Vector3i(4, 4, 1);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = i * Res.x * Res.y + j * Res.x + k;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.5;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.5;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0]  -  0.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1]  -  0.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2]  -  1.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3]  -  2.9405), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[4]  -  4.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[5]  -  4.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[6]  -  5.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[7]  -  6.9405), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[8]  -  8.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[9]  -  8.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[10] -  9.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[11] - 10.9405), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[12] - 12.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[13] - 12.8955), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[14] - 13.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[15] - 14.9405), GRID_SOLVER_EPSILON);
	}

	//MonoCatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[22] - 2.437), GRID_SOLVER_EPSILON);
	}

	//测试MonoCatmullRom只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldData(3 * Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::MONOCATMULLROM);
		CCSSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();
		thrust::host_vector<Real> DstFieldResultData2 = SampleDstValue;

		EXPECT_LT(abs(DstFieldResultData[22] - 2.437), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData2[22] - 2.437), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//CubicBridson插值采样
TEST(CellCenteredScalarField, CellCenteredScalarField_CuSamplingCubicBridson)
{
	CCudaContextManager::getInstance().initCudaContext();

	//CubicBridson采样点位于网格中心测试
	{
		Vector3i Res = Vector3i(16, 16, 16);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = k * 10 + j * 20 + i * 30;
				}
			}
		}

		Vector3i Res2 = Vector3i(32, 2, 16);
		vector<Real> PosVectorFieldDataX(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataY(Res2.x * Res2.y * Res2.z);
		vector<Real> PosVectorFieldDataZ(Res2.x * Res2.y * Res2.z);

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					PosVectorFieldDataX[i * Res2.x * Res2.y + j * Res2.x + k] = ((Int)(k / 2)) * 10 - 10 + 5;
					PosVectorFieldDataY[i * Res2.x * Res2.y + j * Res2.x + k] = j * 20 - 40 + 10;
					PosVectorFieldDataZ[i * Res2.x * Res2.y + j * Res2.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res2);
		CCellCenteredVectorField CCVPosField(Res2, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res2.x * Res2.y + j * Res2.x + k] - (((Int)(k / 2)) * 10 + j * 20 + i * 30)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//CubicBridson简单手算测试，只考虑了一维其实
	{
		Vector3i Res = Vector3i(4, 4, 1);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = i * Res.x * Res.y + j * Res.x + k;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.5;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.5;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0] + 0.0285), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1] - 0.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2] - 1.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3] - 2.9285), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[4] - 3.9715), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[5] - 4.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[6] - 5.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[7] - 6.9285), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[8] - 7.9715), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[9] - 8.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[10] - 9.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[11] - 10.9285), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[12] - 11.9715), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[13] - 12.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[14] - 13.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[15] - 14.9285), GRID_SOLVER_EPSILON);
	}
	 
	//CubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//测试CubicBridson只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldData(3 * Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);


		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CUBICBRIDSON);
		CCSSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();
		thrust::host_vector<Real> DstFieldResultData2 = SampleDstValue;

		EXPECT_LT(abs(DstFieldResultData[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData2[22] - 2.5255), GRID_SOLVER_EPSILON);
	}
	
	//ClampCubicBridson简单手算测试，只考虑了一维其实
	{
		Vector3i Res = Vector3i(4, 4, 1);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = i * Res.x * Res.y + j * Res.x + k;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.5;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.5;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[0] - 0.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[1] - 0.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[2] - 1.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[3] - 2.9285), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[4] - 4.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[5] - 4.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[6] - 5.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[7] - 6.9285), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[8] - 8.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[9] - 8.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[10] - 9.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[11] - 10.9285), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[12] - 12.0000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[13] - 12.8835), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[14] - 13.9000), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData[15] - 14.9285), GRID_SOLVER_EPSILON);
	}
	
	//ClampCubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
				}
			}
		}

		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(DstFieldResultData[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//测试ClampCubicBridson只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
				}
			}
		}

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldData(3 * Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.6;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.8;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);


		CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcScalarFieldData.data());
		CCellCenteredScalarField CCSDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCSSrcField.sampleField(CCVPosField, CCSDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);
		CCSSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();
		thrust::host_vector<Real> DstFieldResultData2 = SampleDstValue;

		EXPECT_LT(abs(DstFieldResultData[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(DstFieldResultData2[22] - 2.5255), GRID_SOLVER_EPSILON);
	}
	
	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredScalarField, CellCenteredScalarField_FieldGradient)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(160, 160, 16);
	vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10 + (Int)(j % 2) * 20 + (Int)(i % 2) * 30;
			}
		}
	}

	CCellCenteredScalarField CCSSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcScalarFieldData.data());
	CCellCenteredVectorField CCVDstField(Res);

	CCSSrcField.gradient(CCVDstField);

	thrust::host_vector<Real> DstFieldResultDataX = CCVDstField.getConstGridDataX();
	thrust::host_vector<Real> DstFieldResultDataY = CCVDstField.getConstGridDataY();
	thrust::host_vector<Real> DstFieldResultDataZ = CCVDstField.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(DstFieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - ((Int)((k + 1 < Res.x ? k + 1 : k) % 2) - (Int)((k > 0 ? k - 1 : k) % 2)) * 0.5), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(DstFieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - ((Int)((j + 1 < Res.y ? j + 1 : j) % 2) - (Int)((j > 0 ? j - 1 : j) % 2)) * 0.5), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(DstFieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - ((Int)((i + 1 < Res.z ? i + 1 : i) % 2) - (Int)((i > 0 ? i - 1 : i) % 2)) * 0.5), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredScalarField, CellCenteredScalarField_FieldLaplacian)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(160, 160, 16);
	vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10 + (Int)(j % 2) * 20 + (Int)(i % 2) * 30;
			}
		}
	}

	CCellCenteredScalarField CCSSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcScalarFieldData.data());
	CCellCenteredScalarField CCSDstField(Res);

	CCSSrcField.laplacian(CCSDstField);

	thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				Real TempX = (((Int)(k + 1 < Res.x ? k + 1 : k) % 2) - 2 * (Int)(k % 2) + (Int)((k > 0 ? k - 1 : k) % 2)) / 10.0;
				Real TempY = (((Int)(j + 1 < Res.y ? j + 1 : j) % 2) - 2 * (Int)(j % 2) + (Int)((j > 0 ? j - 1 : j) % 2)) / 20.0;
				Real TempZ = (((Int)(i + 1 < Res.z ? i + 1 : i) % 2) - 2 * (Int)(i % 2) + (Int)((i > 0 ? i - 1 : i) % 2)) / 30.0;
				EXPECT_LT(abs(DstFieldResultData[i * Res.x * Res.y + j * Res.x + k] - TempX - TempY - TempZ), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
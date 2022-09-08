#include "pch.h"
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

TEST(CellCenteredVectorField, CellCenteredVectorField_Constructor)
{
	CCudaContextManager::getInstance().initCudaContext();

	CCellCenteredVectorField CCVField1(2, 2, 2, -10, 0, 0, 2, 4, 6);
	CCellCenteredVectorField CCVField2(Vector3i(16, 16, 16), Vector3(2, 2, 2), Vector3(8, 8, 8));
	CCellCenteredVectorField CCVField3(CCVField2);

	EXPECT_EQ_VECTOR3I(CCVField1.getResolution(), Vector3i(2, 2, 2));
	EXPECT_EQ_VECTOR3(CCVField1.getOrigin(), Vector3(-10, 0, 0));
	EXPECT_EQ_VECTOR3(CCVField1.getSpacing(), Vector3(2, 4, 6));
	EXPECT_EQ_VECTOR3I(CCVField3.getResolution(), Vector3i(16, 16, 16));
	EXPECT_EQ_VECTOR3(CCVField3.getOrigin(), Vector3(2, 2, 2));
	EXPECT_EQ_VECTOR3(CCVField3.getSpacing(), Vector3(8, 8, 8));

	Vector3i Res = Vector3i(30, 31, 32);
	vector<Real> FieldDataX(Res.x * Res.y * Res.z);
	vector<Real> FieldDataY(Res.x * Res.y * Res.z);
	vector<Real> FieldDataZ(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FieldDataX[i * Res.x * Res.y + j * Res.x + k] = k;
				FieldDataY[i * Res.x * Res.y + j * Res.x + k] = j;
				FieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i;
			}
		}
	}

	CCellCenteredVectorField CCVField4(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FieldDataX.data(), FieldDataY.data(), FieldDataZ.data());
	CCellCenteredVectorField CCVField5(CCVField4);
	CCellCenteredVectorField CCVField6(Res);
	CCVField6.resize(CCVField5);

	thrust::host_vector<Real> FieldResultDataX = CCVField5.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVField5.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVField5.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - k), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - j), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - i), GRID_SOLVER_EPSILON);
			}
		}
	}

	FieldResultDataX = CCVField6.getConstGridDataX();
	FieldResultDataY = CCVField6.getConstGridDataY();
	FieldResultDataZ = CCVField6.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - k), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - j), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - i), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredVectorField, CellCenteredVectorField_Assignment)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(16, 16, 16);
	vector<Real> FieldDataX(Res.x * Res.y * Res.z);
	vector<Real> FieldDataY(Res.x * Res.y * Res.z);
	vector<Real> FieldDataZ(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FieldDataX[i * Res.x * Res.y + j * Res.x + k] = k;
				FieldDataY[i * Res.x * Res.y + j * Res.x + k] = j;
				FieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i;
			}
		}
	}

	CCellCenteredVectorField CCVField1(Res, Vector3(2, 2, 2), Vector3(8, 8, 8), FieldDataX.data(), FieldDataY.data(), FieldDataZ.data());
	CCellCenteredVectorField CCVField2(2, 2, 2);

	CCVField2 = CCVField1;

	EXPECT_EQ_VECTOR3I(CCVField2.getResolution(), Res);
	EXPECT_EQ_VECTOR3(CCVField2.getOrigin(), Vector3(2, 2, 2));
	EXPECT_EQ_VECTOR3(CCVField2.getSpacing(), Vector3(8, 8, 8));

	thrust::host_vector<Real> FieldResultDataX = CCVField2.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVField2.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVField2.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldDataX[i * Res.x * Res.y + j * Res.x + k] - k), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldDataY[i * Res.x * Res.y + j * Res.x + k] - j), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldDataZ[i * Res.x * Res.y + j * Res.x + k] - i), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredVectorField, CellCenteredVectorField_PlusAndScale)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(31, 16, 18);
	vector<Real> FieldData1X(Res.x * Res.y * Res.z);
	vector<Real> FieldData1Y(Res.x * Res.y * Res.z);
	vector<Real> FieldData1Z(Res.x * Res.y * Res.z);
	vector<Real> FieldData2X(Res.x * Res.y * Res.z);
	vector<Real> FieldData2Y(Res.x * Res.y * Res.z);
	vector<Real> FieldData2Z(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FieldData1X[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
				FieldData1Y[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 2) * 20;
				FieldData1Z[i * Res.x * Res.y + j * Res.x + k] = (Int)(i % 2) * 30;

				FieldData2X[i * Res.x * Res.y + j * Res.x + k] = k * 10;
				FieldData2Y[i * Res.x * Res.y + j * Res.x + k] = j * 20;
				FieldData2Z[i * Res.x * Res.y + j * Res.x + k] = i * 30;
			}
		}
	}

	CCellCenteredVectorField CCVField1(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), FieldData1X.data(), FieldData1Y.data(), FieldData1Z.data());
	CCellCenteredVectorField CCVField2(Res, Vector3(-9, 2, -6), Vector3(1, 2, 3), FieldData2X.data(), FieldData2Y.data(), FieldData2Z.data());

	CCVField1 *= 0.1;
	CCVField1.scale(5);
	CCVField1 += CCVField2;

	thrust::host_vector<Real> FieldResultData1X = CCVField1.getConstGridDataX();
	thrust::host_vector<Real> FieldResultData1Y = CCVField1.getConstGridDataY();
	thrust::host_vector<Real> FieldResultData1Z = CCVField1.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1X[i * Res.x * Res.y + j * Res.x + k] - (Int)(k % 2) * 5 - k * 10), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData1Y[i * Res.x * Res.y + j * Res.x + k] - (Int)(j % 2) * 10 - j * 20), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData1Z[i * Res.x * Res.y + j * Res.x + k] - (Int)(i % 2) * 15 - i * 30), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCVField1.plusAlphaX(CCVField1, -1);

	FieldResultData1X = CCVField1.getConstGridDataX();
	FieldResultData1Y = CCVField1.getConstGridDataY();
	FieldResultData1Z = CCVField1.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1X[i * Res.x * Res.y + j * Res.x + k] - 0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData1Y[i * Res.x * Res.y + j * Res.x + k] - 0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData1Z[i * Res.x * Res.y + j * Res.x + k] - 0), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCVField2.scale(Vector3(0.1, 0.5, 1));
	CCVField2.plusAlphaX(CCVField2, Vector3(9, 19, 29));

	thrust::host_vector<Real> FieldResultData2X = CCVField2.getConstGridDataX();
	thrust::host_vector<Real> FieldResultData2Y = CCVField2.getConstGridDataY();
	thrust::host_vector<Real> FieldResultData2Z = CCVField2.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultData2X[i * Res.x * Res.y + j * Res.x + k] - k * 10), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData2Y[i * Res.x * Res.y + j * Res.x + k] - j * 200), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData2Z[i * Res.x * Res.y + j * Res.x + k] - i * 900), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//Trilinear插值采样
TEST(CellCenteredVectorField, CellCenteredVectorField_CuSamplingTrilinear)
{
	CCudaContextManager::getInstance().initCudaContext();

	//简单测试采样
	{
		Vector3i Res = Vector3i(2, 2, 1);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2);
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 2);
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (Int)(i % 2);
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.5;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.5;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.5;
				}
			}
		}

		CCellCenteredVectorField CCVSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[1] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[3] - 1), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataY[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[2] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[3] - 1), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataZ[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[3] - 0), GRID_SOLVER_EPSILON);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k + 0.4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j + 0.4;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + 0.4;
				}
			}
		}

		CCVPosField.resize(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());
		CCVDstField.resize(Res);

		CCVSrcField.sampleField(CCVPosField, CCVDstField);

		FieldResultDataX = CCVDstField.getConstGridDataX();
		FieldResultDataY = CCVDstField.getConstGridDataY();
		FieldResultDataZ = CCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[1] - 0.9), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[3] - 0.9), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataY[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[2] - 0.9), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[3] - 0.9), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataZ[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[3] - 0), GRID_SOLVER_EPSILON);

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
		CCVDstField.resize(Res);

		CCVSrcField.sampleField(CCVPosField, CCVDstField);

		FieldResultDataX = CCVDstField.getConstGridDataX();
		FieldResultDataY = CCVDstField.getConstGridDataY();
		FieldResultDataZ = CCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[0] - 0.1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[1] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[2] - 0.1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[3] - 1), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataY[0] - 0.1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[1] - 0.1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[2] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[3] - 1), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataZ[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[3] - 0), GRID_SOLVER_EPSILON);
	}

	{}
	//测试网格Origin和Spacing不是默认情况下的采样
	{
		Vector3i Res = Vector3i(16, 16, 16);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 2) * 10;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (Int)(i % 2) * 10;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 5;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 10;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - 10 * (Int)(k % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - 10 * (Int)(j % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - 10 * (Int)(i % 2)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//测试源网格和目的网格分辨率不同的情况下的采样
	{
		Vector3i Res = Vector3i(16, 16, 16);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 2) * 10;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (Int)(i % 2) * 10;
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


		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res2);
		CCellCenteredVectorField CCVPosField(Res2, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();


		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataX[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(((Int)(k / 2)) % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataY[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(j % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataZ[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(i % 2)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//测试只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(16, 16, 16);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2) * 10;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 2) * 10;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (Int)(i % 2) * 10;
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

		thrust::device_vector<Real> SampleDstValue(3 * Res2.x * Res2.y * Res2.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res2.x * Res2.y * Res2.z);

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res2);
		CCellCenteredVectorField CCVPosField(Res2, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField);
		CCVSrcField.sampleField(SamplePos, SampleDstValue);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;


		for (int i = 0; i < Res2.z; i++)
		{
			for (int j = 0; j < Res2.y; j++)
			{
				for (int k = 0; k < Res2.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataX[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(((Int)(k / 2)) % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataY[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(j % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataZ[i * Res2.x * Res2.y + j * Res2.x + k] - 10 * (Int)(i % 2)), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res2.x * Res2.y + j * Res2.x + k)] - FieldResultDataX[i * Res2.x * Res2.y + j * Res2.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res2.x * Res2.y + j * Res2.x + k) + 1] - FieldResultDataY[i * Res2.x * Res2.y + j * Res2.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res2.x * Res2.y + j * Res2.x + k) + 2] - FieldResultDataZ[i * Res2.x * Res2.y + j * Res2.x + k]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//Catmull-Rom插值采样插值采样
TEST(CellCenteredVectorField, CellCenteredVectorField_CuSamplingCatmullRom)
{
	CCudaContextManager::getInstance().initCudaContext();

	//CatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.4415), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[22] - 2.4415), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[22] - 2.4415), GRID_SOLVER_EPSILON);
	}

	//MonoCatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.437), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[22] - 2.437), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[22] - 2.437), GRID_SOLVER_EPSILON);
	}

	//测试只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
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
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 24;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(3 * Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::MONOCATMULLROM);
		CCVSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;

		EXPECT_LT(abs(FieldResultDataX[22] - 2.437), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[22] - 2.437), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[22] - 2.437), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[3 * 22] - FieldResultDataX[22]), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[3 * 22 + 1] - FieldResultDataY[22]), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[3 * 22 + 2] - FieldResultDataZ[22]), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//CubicBridson插值采样插值采样
TEST(CellCenteredVectorField, CellCenteredVectorField_CuSamplingCubicBridson)
{
	CCudaContextManager::getInstance().initCudaContext();

	//CubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//ClampCubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//测试只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(4, 4, 4);
		vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);
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
					SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 24;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(3 * Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		CCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);
		CCVSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;

		EXPECT_LT(abs(FieldResultDataX[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[22] - 2.5255), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[3 * 22] - FieldResultDataX[22]), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[3 * 22 + 1] - FieldResultDataY[22]), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[3 * 22 + 2] - FieldResultDataZ[22]), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredVectorField, CellCenteredVectorField_FieldDivergence)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(30, 31, 32);
	vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = (Int)(k % 2);
				SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = (Int)(j % 3);
				SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = (Int)(i % 4);
			}
		}
	}

	CCellCenteredVectorField CCVSrcField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
	CCellCenteredScalarField CCSDstField(Res);

	CCVSrcField.divergence(CCSDstField);

	thrust::host_vector<Real> FieldResultData = CCSDstField.getConstGridData();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				Real TempX = (((Int)(k + 1 < Res.x ? k + 1 : k) % 2) - (Int)((k > 0 ? k - 1 : k) % 2)) / 10.0 * 0.5;
				Real TempY = (((Int)(j + 1 < Res.y ? j + 1 : j) % 3) - (Int)((j > 0 ? j - 1 : j) % 3)) / 20.0 * 0.5;
				Real TempZ = (((Int)(i + 1 < Res.z ? i + 1 : i) % 4) - (Int)((i > 0 ? i - 1 : i) % 4)) / 30.0 * 0.5;
				EXPECT_LT(abs(FieldResultData[i * Res.x * Res.y + j * Res.x + k] - TempX - TempY - TempZ), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(CellCenteredVectorField, CellCenteredVectorField_FieldCurl)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(30, 31, 32);
	vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = i + j + k;
				SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = i + j + k;
				SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i + j + k;
			}
		}
	}

	CCellCenteredVectorField CCVSrcField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstField(Res);

	CCVSrcField.curl(CCVDstField);

	thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				Real Fx_ym = i + (j > 0 ? j - 1 : j) + k;
				Real Fx_yp = i + (j + 1 < Res.y ? j + 1 : j) + k;
				Real Fx_zm = (i > 0 ? i - 1 : i) + j + k;
				Real Fx_zp = (i + 1 < Res.z ? i + 1 : i) + j + k;
				Real Fy_xm = i + j + (k > 0 ? k - 1 : k);
				Real Fy_xp = i + j + (k + 1 < Res.x ? k + 1 : k);
				Real Fy_zm = (i > 0 ? i - 1 : i) + j + k;
				Real Fy_zp = (i + 1 < Res.z ? i + 1 : i) + j + k;
				Real Fz_xm = i + j + (k > 0 ? k - 1 : k);
				Real Fz_xp = i + j + (k + 1 < Res.x ? k + 1 : k);
				Real Fz_ym = i + (j > 0 ? j - 1 : j) + k;
				Real Fz_yp = i + (j + 1 < Res.y ? j + 1 : j) + k;
				Real TempX = 0.5 * (Fz_yp - Fz_ym) / 20.0 - 0.5 * (Fy_zp - Fy_zm) / 30.0;
				Real TempY = 0.5 * (Fx_zp - Fx_zm) / 30.0 - 0.5 * (Fz_xp - Fz_xm) / 10.0;
				Real TempZ = 0.5 * (Fy_xp - Fy_xm) / 10.0 - 0.5 * (Fx_yp - Fx_ym) / 20.0;
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - TempX), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - TempY), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - TempZ), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
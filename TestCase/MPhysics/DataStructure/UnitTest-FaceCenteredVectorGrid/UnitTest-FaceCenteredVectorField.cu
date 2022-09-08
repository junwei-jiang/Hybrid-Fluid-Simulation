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

TEST(FaceCenteredVectorField, FaceCenteredVectorField_Constructor)
{
	CCudaContextManager::getInstance().initCudaContext();

	CFaceCenteredVectorField FCVField1(2, 2, 2, -10, 0, 0, 2, 4, 6);
	CFaceCenteredVectorField FCVField2(Vector3i(16, 16, 16), Vector3(2, 2, 2), Vector3(8, 8, 8));
	CFaceCenteredVectorField FCVField3(FCVField2);

	EXPECT_EQ_VECTOR3I(FCVField1.getResolution(), Vector3i(2, 2, 2));
	EXPECT_EQ_VECTOR3(FCVField1.getOrigin(), Vector3(-10, 0, 0));
	EXPECT_EQ_VECTOR3(FCVField1.getSpacing(), Vector3(2, 4, 6));
	EXPECT_EQ_VECTOR3I(FCVField3.getResolution(), Vector3i(16, 16, 16));
	EXPECT_EQ_VECTOR3(FCVField3.getOrigin(), Vector3(2, 2, 2));
	EXPECT_EQ_VECTOR3(FCVField3.getSpacing(), Vector3(8, 8, 8));

	Vector3i Res = Vector3i(30, 31, 32);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);
	vector<Real> FieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
			}
		}
	}

	CFaceCenteredVectorField FCVField4(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FieldDataX.data(), FieldDataY.data(), FieldDataZ.data());
	CFaceCenteredVectorField FCVField5(FCVField4);
	CFaceCenteredVectorField FCVField6(Res);
	FCVField6.resize(FCVField5);

	thrust::host_vector<Real> FieldResultDataX = FCVField5.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = FCVField5.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = FCVField5.getConstGridDataZ();
	

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * ResX.x * ResX.y + j * ResX.x + k] - k), 1e-4);
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataY[i * ResY.x * ResY.y + j * ResY.x + k] - j), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] - i), GRID_SOLVER_EPSILON);
			}
		}
	}

	FieldResultDataX = FCVField6.getConstGridDataX();
	FieldResultDataY = FCVField6.getConstGridDataY();
	FieldResultDataZ = FCVField6.getConstGridDataZ();

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * ResX.x * ResX.y + j * ResX.x + k] - k), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataY[i * ResY.x * ResY.y + j * ResY.x + k] - j), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] - i), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(FaceCenteredVectorField, FaceCenteredVectorField_Assignment)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(8, 16, 32);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);
	vector<Real> FieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
			}
		}
	}

	CFaceCenteredVectorField CCVField1(Res, Vector3(2, 2, 2), Vector3(8, 8, 8), FieldDataX.data(), FieldDataY.data(), FieldDataZ.data());
	CFaceCenteredVectorField CCVField2(2, 2, 2);

	CCVField2 = CCVField1;

	EXPECT_EQ_VECTOR3I(CCVField2.getResolution(), Res);
	EXPECT_EQ_VECTOR3(CCVField2.getOrigin(), Vector3(2, 2, 2));
	EXPECT_EQ_VECTOR3(CCVField2.getSpacing(), Vector3(8, 8, 8));

	thrust::host_vector<Real> FieldResultDataX = CCVField2.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVField2.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVField2.getConstGridDataZ();

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * ResX.x * ResX.y + j * ResX.x + k] - k), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataY[i * ResY.x * ResY.y + j * ResY.x + k] - j), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] - i), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(FaceCenteredVectorField, FaceCenteredVectorField_PlusAndScale)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(2, 10, 20);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);
	vector<Real> FieldData1X(ResX.x * ResX.y * ResX.z);
	vector<Real> FieldData1Y(ResY.x * ResY.y * ResY.z);
	vector<Real> FieldData1Z(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> FieldData2X(ResX.x * ResX.y * ResX.z);
	vector<Real> FieldData2Y(ResY.x * ResY.y * ResY.z);
	vector<Real> FieldData2Z(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FieldData1X[i * ResX.x * ResX.y + j * ResX.x + k] = (Int)(k % 3) * 10;
				FieldData2X[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FieldData1Y[i * ResY.x * ResY.y + j * ResY.x + k] = (Int)(j % 4) * 10;
				FieldData2Y[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FieldData1Z[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (Int)(i % 5) * 10;
				FieldData2Z[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
			}
		}
	}

	CFaceCenteredVectorField FCVField1(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), FieldData1X.data(), FieldData1Y.data(), FieldData1Z.data());
	CFaceCenteredVectorField FCVField2(Res, Vector3(5, 4, 7), Vector3(2, 20, -50), FieldData2X.data(), FieldData2Y.data(), FieldData2Z.data());

	FCVField1 *= 0.1;
	FCVField1.scale(5);
	FCVField1 += FCVField2;

	thrust::host_vector<Real> FieldResultData1X = FCVField1.getConstGridDataX();
	thrust::host_vector<Real> FieldResultData1Y = FCVField1.getConstGridDataY();
	thrust::host_vector<Real> FieldResultData1Z = FCVField1.getConstGridDataZ();

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1X[i * ResX.x * ResX.y + j * ResX.x + k] - (Int)(k % 3) * 5 - k * 10), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1Y[i * ResY.x * ResY.y + j * ResY.x + k] - (Int)(j % 4) * 5 - j * 10), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1Z[i * ResZ.x * ResZ.y + j * ResZ.x + k] - (Int)(i % 5) * 5 - i * 10), GRID_SOLVER_EPSILON);
			}
		}
	}

	FCVField1.plusAlphaX(FCVField1, -1);

	FieldResultData1X = FCVField1.getConstGridDataX();
	FieldResultData1Y = FCVField1.getConstGridDataY();
	FieldResultData1Z = FCVField1.getConstGridDataZ();

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1X[i * ResX.x * ResX.y + j * ResX.x + k] - 0), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1Y[i * ResY.x * ResY.y + j * ResY.x + k] - 0), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1Z[i * ResZ.x * ResZ.y + j * ResZ.x + k] - 0), GRID_SOLVER_EPSILON);
			}
		}
	}

	FCVField2.scale(Vector3(0.1, 0.5, 1));
	FCVField2.plusAlphaX(FCVField2, Vector3(9, 19, 29));

	thrust::host_vector<Real> FieldResultData2X = FCVField2.getConstGridDataX();
	thrust::host_vector<Real> FieldResultData2Y = FCVField2.getConstGridDataY();
	thrust::host_vector<Real> FieldResultData2Z = FCVField2.getConstGridDataZ();

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				EXPECT_LT(abs(FieldResultData2X[i * ResX.x * ResX.y + j * ResX.x + k] - k * 10), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				EXPECT_LT(abs(FieldResultData2Y[i * ResY.x * ResY.y + j * ResY.x + k] - j * 100), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				EXPECT_LT(abs(FieldResultData2Z[i * ResZ.x * ResZ.y + j * ResZ.x + k] - i * 300), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//Trilinear插值采样
TEST(FaceCenteredVectorField, FaceCenteredVectorField_CuSamplingTrilinear)
{
	CCudaContextManager::getInstance().initCudaContext();
	
	//简单测试采样
	{
		Vector3i Res = Vector3i(2, 2, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (Int)(k % 2);
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j + 0.5;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i + 0.5;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (Int)(j % 2);
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k + 0.5;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i + 0.5;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (Int)(i % 2);
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k + 0.5;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j + 0.5;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[1] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[3] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[4] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[5] - 0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataY[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[2] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[3] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[4] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[5] - 0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataZ[0] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[1] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[2] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[3] - 0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[4] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[5] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[6] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[7] - 1), GRID_SOLVER_EPSILON);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j + 0.4;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i + 0.4;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k + 0.4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i + 0.4;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k + 0.4;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j + 0.4;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				}
			}
		}

		CCVPosFieldX.resize(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCVPosFieldY.resize(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCVPosFieldZ.resize(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());
		FCVDstField.resize(Res);

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField);

		FieldResultDataX = FCVDstField.getConstGridDataX();
FieldResultDataY = FCVDstField.getConstGridDataY();
FieldResultDataZ = FCVDstField.getConstGridDataZ();

EXPECT_LT(abs(FieldResultDataX[0] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[1] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[2] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[3] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[4] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[5] - 0), GRID_SOLVER_EPSILON);

EXPECT_LT(abs(FieldResultDataY[0] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[1] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[2] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[3] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[4] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[5] - 0), GRID_SOLVER_EPSILON);

EXPECT_LT(abs(FieldResultDataZ[0] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[1] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[2] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[3] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[4] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[5] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[6] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[7] - 1), GRID_SOLVER_EPSILON);

for (int i = 0; i < ResX.z; i++)
{
	for (int j = 0; j < ResX.y; j++)
	{
		for (int k = 0; k < ResX.x; k++)
		{
			PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
			PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j + 0.6;
			PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i + 0.6;
		}
	}
}

for (int i = 0; i < ResY.z; i++)
{
	for (int j = 0; j < ResY.y; j++)
	{
		for (int k = 0; k < ResY.x; k++)
		{
			PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k + 0.6;
			PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
			PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i + 0.6;
		}
	}
}

for (int i = 0; i < ResZ.z; i++)
{
	for (int j = 0; j < ResZ.y; j++)
	{
		for (int k = 0; k < ResZ.x; k++)
		{
			PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k + 0.6;
			PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j + 0.6;
			PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
		}
	}
}

CCVPosFieldX.resize(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
CCVPosFieldY.resize(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
CCVPosFieldZ.resize(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());
FCVDstField.resize(Res);

FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField);

FieldResultDataX = FCVDstField.getConstGridDataX();
FieldResultDataY = FCVDstField.getConstGridDataY();
FieldResultDataZ = FCVDstField.getConstGridDataZ();

EXPECT_LT(abs(FieldResultDataX[0] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[1] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[2] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[3] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[4] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataX[5] - 0), GRID_SOLVER_EPSILON);

EXPECT_LT(abs(FieldResultDataY[0] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[1] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[2] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[3] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[4] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataY[5] - 0), GRID_SOLVER_EPSILON);

EXPECT_LT(abs(FieldResultDataZ[0] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[1] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[2] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[3] - 0), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[4] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[5] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[6] - 1), GRID_SOLVER_EPSILON);
EXPECT_LT(abs(FieldResultDataZ[7] - 1), GRID_SOLVER_EPSILON);
	}

	{}

	//测试FCV网格采样FCV网格的复杂情况
	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (Int)(k % 3) * 10 + (Int)(j % 3) * 10;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 + 3;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 18;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (Int)(j % 4) * 10 + (Int)(i % 3) * 10;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 - 4;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 18;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (Int)(i % 5) * 10 + (Int)(k % 3) * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[0] - 4), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[1] - 14), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[2] - 21), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[3] - 14), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[4] - 24), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[5] - 31), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[30] - 4), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[31] - 14), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataX[32] - 21), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataY[0] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[1] - 1), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[2] - 9), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[3] - 9), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[22] - 11), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[23] - 11), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataZ[0] - 6), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[1] - 15), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[2] - 6), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[3] - 15), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[20] - 16), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[21] - 25), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[22] - 16), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataZ[23] - 25), GRID_SOLVER_EPSILON);
	}

	{}

	//测试CCV网格采样FCV网格的复杂情况
	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 3;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 8;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		FCVSrcField.sampleField(CCVPosField, CCVDstField);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - k * 10 - 3), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - j * 10 - 4), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - i * 10 - 5), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//测试只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

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
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 3;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 8;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(3 * Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		FCVSrcField.sampleField(CCVPosField, CCVDstField);
		FCVSrcField.sampleField(SamplePos, SampleDstValue);
		FCVSrcField.sampleField(SamplePos, SampleDstValue);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - k * 10 - 3), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - j * 10 - 4), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - i * 10 - 5), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k)] - FieldResultDataX[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] - FieldResultDataY[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] - FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//Catmull-Rom插值采样
TEST(FaceCenteredVectorField, FaceCenteredVectorField_CuSamplingCatmullRom)
{
	CCudaContextManager::getInstance().initCudaContext();

	//CatmullRom手算测试
	{
		Vector3i Res = Vector3i(3, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 - 4;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 18;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.4415), GRID_SOLVER_EPSILON);
	}

	//CatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 3, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataY[22] - 2.4415), GRID_SOLVER_EPSILON);
	}

	//CatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 4, 3);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 9;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataZ[22] - 2.4415), GRID_SOLVER_EPSILON);
	}

	//测试CatmullRom只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

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
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 3;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 8;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(3 * Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		FCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::CATMULLROM);
		FCVSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::CATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k)] - FieldResultDataX[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] - FieldResultDataY[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] - FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//MonoCatmullRom手算测试
	{
		Vector3i Res = Vector3i(3, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 - 4;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 18;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.437), GRID_SOLVER_EPSILON);
	}

	//MonoCatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 3, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataY[22] - 2.437), GRID_SOLVER_EPSILON);
	}

	//MonoCatmullRom手算测试
	{
		Vector3i Res = Vector3i(4, 4, 3);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 9;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataZ[22] - 2.437), GRID_SOLVER_EPSILON);
	}

	//测试MonoCatmullRom只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

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
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 3;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 8;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(3 * Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		FCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::MONOCATMULLROM);
		FCVSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::MONOCATMULLROM);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k)] - FieldResultDataX[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] - FieldResultDataY[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] - FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//CubicBridson插值采样
TEST(FaceCenteredVectorField, FaceCenteredVectorField_CuSamplingCubicBridson)
{
	CCudaContextManager::getInstance().initCudaContext();

	//CubicBridson手算测试
	{
		Vector3i Res = Vector3i(3, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 - 4;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 18;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//CubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 3, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataY[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//CubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 4, 3);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 9;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataZ[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//测试CubicBridson只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

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
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 3;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 8;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(3 * Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		FCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::CUBICBRIDSON);
		FCVSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::CUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k)] - FieldResultDataX[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] - FieldResultDataY[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] - FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//ClampCubicBridson手算测试
	{
		Vector3i Res = Vector3i(3, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 - 4;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 18;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataX[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//ClampCubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 3, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 6;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 15;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataY[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//ClampCubicBridson手算测试
	{
		Vector3i Res = Vector3i(4, 4, 3);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);

		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		vector<Real> PosVectorFieldDataXX(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXY(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataXZ(ResX.x * ResX.y * ResX.z);
		vector<Real> PosVectorFieldDataYX(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYY(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataYZ(ResY.x * ResY.y * ResY.z);
		vector<Real> PosVectorFieldDataZX(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZY(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> PosVectorFieldDataZZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataXX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10 - 10 - 1;
					PosVectorFieldDataXY[i * ResX.x * ResX.y + j * ResX.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataXZ[i * ResX.x * ResX.y + j * ResX.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataYX[i * ResY.x * ResY.y + j * ResY.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataYY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20 - 40 + 2;
					PosVectorFieldDataYZ[i * ResY.x * ResY.y + j * ResY.x + k] = i * 30 - 50 + 24;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (k % 2) + (j % 3) + i;
					PosVectorFieldDataZX[i * ResZ.x * ResZ.y + j * ResZ.x + k] = k * 10 - 10 + 4;
					PosVectorFieldDataZY[i * ResZ.x * ResZ.y + j * ResZ.x + k] = j * 20 - 40 + 12;
					PosVectorFieldDataZZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30 - 50 + 9;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CFaceCenteredVectorField FCVDstField(Res);
		CCellCenteredVectorField CCVPosFieldX(ResX, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataXX.data(), PosVectorFieldDataXY.data(), PosVectorFieldDataXZ.data());
		CCellCenteredVectorField CCVPosFieldY(ResY, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataYX.data(), PosVectorFieldDataYY.data(), PosVectorFieldDataYZ.data());
		CCellCenteredVectorField CCVPosFieldZ(ResZ, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataZX.data(), PosVectorFieldDataZY.data(), PosVectorFieldDataZZ.data());

		FCVSrcField.sampleField(CCVPosFieldX, CCVPosFieldY, CCVPosFieldZ, FCVDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();

		EXPECT_LT(abs(FieldResultDataZ[22] - 2.5255), GRID_SOLVER_EPSILON);
	}

	//测试ClampCubicBridson只输入位置的情况下的采样
	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

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
					PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 3;
					PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 8;
					PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k)] = PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] = PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k];
					PosVectorFieldData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] = PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k];
				}
			}
		}

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 10;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 10;
				}
			}
		}

		thrust::device_vector<Real> SampleDstValue(3 * Res.x * Res.y * Res.z);
		thrust::device_vector<Real> SamplePos;
		assignDeviceVectorReal(SamplePos, PosVectorFieldData.data(), PosVectorFieldData.data() + 3 * Res.x * Res.y * Res.z);

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredVectorField CCVDstField(Res);
		CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

		FCVSrcField.sampleField(CCVPosField, CCVDstField, ESamplingAlgorithm::CLAMPCUBICBRIDSON);
		FCVSrcField.sampleField(SamplePos, SampleDstValue, ESamplingAlgorithm::CLAMPCUBICBRIDSON);

		thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
		thrust::host_vector<Real> FieldResultData = SampleDstValue;

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k)] - FieldResultDataX[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 1] - FieldResultDataY[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultData[3 * (i * Res.x * Res.y + j * Res.x + k) + 2] - FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(FaceCenteredVectorField, FaceCenteredVectorField_FieldDivergence)
{
	CCudaContextManager::getInstance().initCudaContext();

	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = (Int)(k % 2);
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = (Int)(j % 3);
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = (Int)(i % 4);
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredScalarField CCSDstField(Res);

		FCVSrcField.divergence(CCSDstField);

		thrust::host_vector<Real> FieldResultData = CCSDstField.getConstGridData();

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					Real TempX = (((Int)(k + 1 < Res.x + 1 ? k + 1 : k) % 2) - (Int)(k % 2)) / 10.0;
					Real TempY = (((Int)(j + 1 < Res.y + 1 ? j + 1 : j) % 3) - (Int)(j % 3)) / 20.0;
					Real TempZ = (((Int)(i + 1 < Res.z + 1 ? i + 1 : i) % 4) - (Int)(i % 4)) / 30.0;
					EXPECT_LT(abs(FieldResultData[i * Res.x * Res.y + j * Res.x + k] - TempX - TempY - TempZ), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	{
		Vector3i Res = Vector3i(2, 10, 20);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				}
			}
		}

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredScalarField CCSDstField(Res);

		FCVSrcField.divergence(CCSDstField);

		thrust::host_vector<Real> FieldResultData = CCSDstField.getConstGridData();

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					EXPECT_LT(abs(FieldResultData[i * Res.x * Res.y + j * Res.x + k] - 0.1 - 0.05 - 0.03333333), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	{
		Vector3i Res = Vector3i(2, 2, 2);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

		SrcVectorFieldDataX[0] = 15;
		SrcVectorFieldDataX[1] = 20;
		SrcVectorFieldDataX[2] = 5;
		SrcVectorFieldDataX[3] = 8;
		SrcVectorFieldDataX[4] = 7;
		SrcVectorFieldDataX[5] = 22;
		SrcVectorFieldDataX[6] = 11;
		SrcVectorFieldDataX[7] = 94;
		SrcVectorFieldDataX[8] = 45;
		SrcVectorFieldDataX[9] = 33;
		SrcVectorFieldDataX[10] = 0;
		SrcVectorFieldDataX[11] = 12;

		SrcVectorFieldDataY[0] = 5;
		SrcVectorFieldDataY[1] = 0;
		SrcVectorFieldDataY[2] = 25;
		SrcVectorFieldDataY[3] = 18;
		SrcVectorFieldDataY[4] = 67;
		SrcVectorFieldDataY[5] = 42;
		SrcVectorFieldDataY[6] = 38;
		SrcVectorFieldDataY[7] = 49;
		SrcVectorFieldDataY[8] = 45;
		SrcVectorFieldDataY[9] = 42;
		SrcVectorFieldDataY[10] = 41;
		SrcVectorFieldDataY[11] = 42;

		SrcVectorFieldDataZ[0] = 51;
		SrcVectorFieldDataZ[1] = 2;
		SrcVectorFieldDataZ[2] = 4;
		SrcVectorFieldDataZ[3] = 11;
		SrcVectorFieldDataZ[4] = 34;
		SrcVectorFieldDataZ[5] = 15;
		SrcVectorFieldDataZ[6] = 18;
		SrcVectorFieldDataZ[7] = 0;
		SrcVectorFieldDataZ[8] = 54;
		SrcVectorFieldDataZ[9] = 19;
		SrcVectorFieldDataZ[10] = 0;
		SrcVectorFieldDataZ[11] = 21;

		CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
		CCellCenteredScalarField CCSDstField(Res);

		FCVSrcField.divergence(CCSDstField);

		thrust::host_vector<Real> FieldResultData = CCSDstField.getConstGridData();

		EXPECT_LT(abs(FieldResultData[0] - 5.0 / 10.0 - 20.0 / 20.0 + 17 / 30.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[1] + 15.0 / 10.0 - 18.0 / 20.0 - 13.0 / 30.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[2] + 1.0 / 10.0 - 42.0 / 20.0 - 14.0 / 30.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[3] - 15.0 / 10.0 - 24.0 / 20.0 + 11.0 / 30.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[4] - 83.0 / 10.0 - 7.0 / 20.0 - 20.0 / 30.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[5] + 49.0 / 10.0 + 7.0 / 20.0 - 4.0 / 30.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[6] + 33.0 / 10.0 + 4.0 / 20.0 + 18.0 / 30.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultData[7] - 12.0 / 10.0 - 0.0 / 20.0 - 21.0 / 30.0), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

TEST(FaceCenteredVectorField, FaceCenteredVectorField_FieldCurl)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(2, 10, 20);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	vector<Real> PosVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> PosVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> PosVectorFieldDataZ(Res.x * Res.y * Res.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				PosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 - 10 + 5;
				PosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 - 40 + 10;
				PosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 - 50 + 15;
			}
		}
	}

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k + j;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j + i;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i + k + j;
			}
		}
	}

	CFaceCenteredVectorField FCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
	CCellCenteredVectorField CCVSrcField(Res, Vector3(-10, -40, -50), Vector3(10, 20, 30));
	CCellCenteredVectorField CCVDstField1(Res);
	CCellCenteredVectorField CCVDstField2(Res);
	CCellCenteredVectorField CCVPosField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), PosVectorFieldDataX.data(), PosVectorFieldDataY.data(), PosVectorFieldDataZ.data());

	FCVSrcField.sampleField(CCVPosField, CCVSrcField);

	FCVSrcField.curl(CCVDstField1);
	CCVSrcField.curl(CCVDstField2);

	thrust::host_vector<Real> FieldResultData1X = CCVDstField1.getConstGridDataX();
	thrust::host_vector<Real> FieldResultData1Y = CCVDstField1.getConstGridDataY();
	thrust::host_vector<Real> FieldResultData1Z = CCVDstField1.getConstGridDataZ();
	thrust::host_vector<Real> FieldResultData2X = CCVDstField2.getConstGridDataX();
	thrust::host_vector<Real> FieldResultData2Y = CCVDstField2.getConstGridDataY();
	thrust::host_vector<Real> FieldResultData2Z = CCVDstField2.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultData1X[i * Res.x * Res.y + j * Res.x + k] - FieldResultData2X[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData1Y[i * Res.x * Res.y + j * Res.x + k] - FieldResultData2Y[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultData1Z[i * Res.x * Res.y + j * Res.x + k] - FieldResultData2Z[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
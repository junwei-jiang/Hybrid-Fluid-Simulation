#include "pch.h"
#include "PressureSolver.h"
#include "CudaContextManager.h"
#include "FieldMathTool.cuh"
#include "EulerSolverTool.cuh"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>

void printData2TXT(const char *vFileName, const Vector3i& vRes, const CCellCenteredVectorField& vVectorField)
{
	ofstream fout(vFileName);

	thrust::host_vector<Real> FieldDataX = vVectorField.getConstGridDataX();
	thrust::host_vector<Real> FieldDataY = vVectorField.getConstGridDataY();
	thrust::host_vector<Real> FieldDataZ = vVectorField.getConstGridDataZ();

	if (fout)
	{
		for (int i = 0; i < vRes.z; i++)
		{
			for (int j = 0; j < vRes.y; j++)
			{
				for (int k = 0; k < vRes.x; k++)
				{
					fout << FieldDataX[i * vRes.x * vRes.y + j * vRes.x + k] << " ";
					fout << FieldDataY[i * vRes.x * vRes.y + j * vRes.x + k] << " ";
					fout << FieldDataZ[i * vRes.x * vRes.y + j * vRes.x + k] << "\n";
				}
			}
		}
		fout.close();

		printf("OK \n");
	}
}

CCellCenteredVectorField readDataFromTXT(const char *vFileName, const Vector3i& vRes)
{
	ifstream GridData(vFileName);

	if (!GridData.is_open())
	{
		cout << "can not open this file" << endl;
	}

	vector<Real> SrcVectorFieldDataX(vRes.x * vRes.y * vRes.z);
	vector<Real> SrcVectorFieldDataY(vRes.x * vRes.y * vRes.z);
	vector<Real> SrcVectorFieldDataZ(vRes.x * vRes.y * vRes.z);

	for (int i = 0; i < vRes.z; i++)
	{
		for (int j = 0; j < vRes.y; j++)
		{
			for (int k = 0; k < vRes.x; k++)
			{
				GridData >> SrcVectorFieldDataX[i * vRes.x * vRes.y + j * vRes.x + k];
				GridData >> SrcVectorFieldDataY[i * vRes.x * vRes.y + j * vRes.x + k];
				GridData >> SrcVectorFieldDataZ[i * vRes.x * vRes.y + j * vRes.x + k];
			}
		}
	}

	CCellCenteredVectorField VectorField(vRes, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());

	return VectorField;
}

//__buildDisMarkers手算测试1(二维情况)
TEST(Extrapolation, Extrapolation_buildDisMarkers1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(6, 6, 1);

	vector<Real> ScalarFieldData(Res.x * Res.y * Res.z, UNKNOWN);

	ScalarFieldData[14] = 0;
	ScalarFieldData[15] = 1;
	ScalarFieldData[20] = 2;

	CCellCenteredScalarField CCSSrcScalarField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), ScalarFieldData.data());
	CCellCenteredScalarField CCSDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCSSrcScalarField, CCSDisMarkersField, 3);

	thrust::host_vector<Real> MarkersResultVector = CCSDisMarkersField.getConstGridData();

	EXPECT_LT(abs(MarkersResultVector[0] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[1] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[3] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[4] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[5] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[6] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[7] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[8] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[9] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[10] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[11] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[12] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[13] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[14] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[15] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[16] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[17] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[18] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[19] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[20] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[21] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[22] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[23] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[24] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[25] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[26] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[27] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[28] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[29] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[30] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[31] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[32] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[33] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[34] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[35] - UNKNOWN), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildDisMarkers手算测试2(三维情况)
TEST(Extrapolation, Extrapolation_buildDisMarkers2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);

	vector<Real> ScalarFieldData(Res.x * Res.y * Res.z, UNKNOWN);

	ScalarFieldData[5] = 0;
	ScalarFieldData[21] = 0;
	ScalarFieldData[22] = 1;
	ScalarFieldData[25] = -2;
	ScalarFieldData[26] = 2;

	CCellCenteredScalarField CCSSrcScalarField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), ScalarFieldData.data());
	CCellCenteredScalarField CCSDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCSSrcScalarField, CCSDisMarkersField, 2);

	thrust::host_vector<Real> MarkersResultVector = CCSDisMarkersField.getConstGridData();

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i == 1 || i == 4 || i == 6 || i == 9 || i == 10 || i == 17 || i == 18 || i == 20 || i == 23 || i == 24 || i == 27 || i == 29 || i == 30 || i == 37 || i == 38 || i == 41 || i == 42)
		{
			EXPECT_LT(abs(MarkersResultVector[i] - 1), GRID_SOLVER_EPSILON);
		}
		else if (i == 5 || i == 21 || i == 22 || i == 25 || i == 26)
		{
			EXPECT_LT(abs(MarkersResultVector[i] - 0), GRID_SOLVER_EPSILON);
		}
		else
		{
			EXPECT_LT(abs(MarkersResultVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildDisMarkers手算测试3(给定距离过大且有分离的外推域的情况)
TEST(Extrapolation, Extrapolation_buildDisMarkers3)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(3, 3, 1);

	vector<Real> ScalarFieldData(Res.x * Res.y * Res.z, UNKNOWN);

	ScalarFieldData[2] = 0;
	ScalarFieldData[6] = 1;

	CCellCenteredScalarField CCSSrcScalarField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), ScalarFieldData.data());
	CCellCenteredScalarField CCSDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCSSrcScalarField, CCSDisMarkersField, 200);

	thrust::host_vector<Real> MarkersResultVector = CCSDisMarkersField.getConstGridData();

	EXPECT_LT(abs(MarkersResultVector[0] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[1] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[2] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[3] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[4] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[5] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[6] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[7] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[8] - 2), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildDisMarkers手算测试4(FCVGrid)
TEST(Extrapolation, Extrapolation_buildDisMarkers4)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> VectorFieldDataX(ResX.x * ResX.y * ResX.z, UNKNOWN);
	vector<Real> VectorFieldDataY(ResY.x * ResY.y * ResY.z, UNKNOWN);
	vector<Real> VectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, UNKNOWN);

	VectorFieldDataX[0] = 1;
	VectorFieldDataX[1] = 2;
	VectorFieldDataY[0] = 3;
	VectorFieldDataZ[0] = 4;

	CFaceCenteredVectorField FCVSrcVectorField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VectorFieldDataX.data(), VectorFieldDataY.data(), VectorFieldDataZ.data());
	CFaceCenteredVectorField FCVDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(FCVSrcVectorField, FCVDisMarkersField, 3);

	thrust::host_vector<Real> MarkersResultXVector = FCVDisMarkersField.getConstGridDataX();
	thrust::host_vector<Real> MarkersResultYVector = FCVDisMarkersField.getConstGridDataY();
	thrust::host_vector<Real> MarkersResultZVector = FCVDisMarkersField.getConstGridDataZ();

	for (int i = 0; i < ResX.x * ResX.y * ResX.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 3 && i != 5 && i != 6 && i != 7 && i != 10 && i != 11 && i != 20 && i != 21 && i != 22 && i != 25 && i != 26 && i != 40 && i != 41)
		{
			EXPECT_LT(abs(MarkersResultXVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(MarkersResultXVector[0] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[1] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[2] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[3] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[5] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[6] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[7] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[10] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[11] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[20] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[21] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[22] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[25] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[26] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[40] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[41] - 2), GRID_SOLVER_EPSILON);

	for (int i = 0; i < ResY.x * ResY.y * ResY.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 20 && i != 21 && i != 24 && i != 40)
		{
			EXPECT_LT(abs(MarkersResultYVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(MarkersResultYVector[0] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[1] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[4] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[5] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[8] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[20] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[21] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[24] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[40] - 2), GRID_SOLVER_EPSILON);

	for (int i = 0; i < ResZ.x * ResZ.y * ResZ.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 16 && i != 17 && i != 20 && i != 32)
		{
			EXPECT_LT(abs(MarkersResultZVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(MarkersResultZVector[0] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[1] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[4] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[5] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[8] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[16] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[17] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[20] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[32] - 2), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildDisMarkers手算测试5(CCVGrid)
TEST(Extrapolation, Extrapolation_buildDisMarkers5)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);

	vector<Real> VectorFieldDataX(Res.x * Res.y * Res.z, UNKNOWN);
	vector<Real> VectorFieldDataY(Res.x * Res.y * Res.z, UNKNOWN);
	vector<Real> VectorFieldDataZ(Res.x * Res.y * Res.z, UNKNOWN);

	VectorFieldDataX[0] = 1;
	VectorFieldDataX[1] = 2;
	VectorFieldDataY[0] = 3;
	VectorFieldDataZ[0] = 4;

	CCellCenteredVectorField CCVSrcVectorField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VectorFieldDataX.data(), VectorFieldDataY.data(), VectorFieldDataZ.data());
	CCellCenteredVectorField CCVDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCVSrcVectorField, CCVDisMarkersField, 3);

	thrust::host_vector<Real> MarkersResultXVector = CCVDisMarkersField.getConstGridDataX();
	thrust::host_vector<Real> MarkersResultYVector = CCVDisMarkersField.getConstGridDataY();
	thrust::host_vector<Real> MarkersResultZVector = CCVDisMarkersField.getConstGridDataZ();

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 3 && i != 4 && i != 5 && i != 6 && i != 8 && i != 9 && i != 16 && i != 17 && i != 18 && i != 20 && i != 21 && i != 32 && i != 33)
		{
			EXPECT_LT(abs(MarkersResultXVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(MarkersResultXVector[0] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[1] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[2] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[3] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[4] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[5] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[6] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[8] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[9] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[16] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[17] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[18] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[20] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[21] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[32] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultXVector[33] - 2), GRID_SOLVER_EPSILON);

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 16 && i != 17 && i != 20 && i != 32)
		{
			EXPECT_LT(abs(MarkersResultYVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(MarkersResultYVector[0] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[1] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[4] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[5] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[8] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[16] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[17] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[20] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultYVector[32] - 2), GRID_SOLVER_EPSILON);

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 16 && i != 17 && i != 20 && i != 32)
		{
			EXPECT_LT(abs(MarkersResultZVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(MarkersResultZVector[0] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[1] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[4] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[5] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[8] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[16] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[17] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[20] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultZVector[32] - 2), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__extrapolatingData手算测试1(二维情况)
TEST(Extrapolation, Extrapolation_extrapolationData1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(6, 6, 1);

	vector<Real> ScalarFieldData(Res.x * Res.y * Res.z, UNKNOWN);

	ScalarFieldData[14] = 0;
	ScalarFieldData[15] = 1;
	ScalarFieldData[20] = 2;

	CCellCenteredScalarField CCSSrcScalarField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), ScalarFieldData.data());
	CCellCenteredScalarField CCSDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCSSrcScalarField, CCSDisMarkersField, 3);

	thrust::host_vector<Real> ExtrapolationResultVector = CCSSrcScalarField.getConstGridData();

	EXPECT_LT(abs(ExtrapolationResultVector[0] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[1] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[2] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[3] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[4] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[5] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[6] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[7] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[8] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[9] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[10] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[11] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[12] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[13] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[14] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[15] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[16] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[17] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[18] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[19] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[20] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[21] - 1.5), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[22] - 1.25), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[23] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[24] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[25] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[26] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[27] - 1.75), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[28] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[29] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[30] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[31] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[32] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[33] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[34] - UNKNOWN), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[35] - UNKNOWN), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__extrapolatingData手算测试2(三维情况)
TEST(Extrapolation, Extrapolation_extrapolationData2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);

	vector<Real> ScalarFieldData(Res.x * Res.y * Res.z, UNKNOWN);

	ScalarFieldData[5] = 0;
	ScalarFieldData[21] = 0;
	ScalarFieldData[22] = 1;
	ScalarFieldData[25] = -2;
	ScalarFieldData[26] = 2;

	CCellCenteredScalarField CCSSrcScalarField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), ScalarFieldData.data());
	CCellCenteredScalarField CCSDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCSSrcScalarField, CCSDisMarkersField, 2);

	thrust::host_vector<Real> ExtrapolationResultVector = CCSSrcScalarField.getConstGridData();

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i == 1 || i == 4 || i == 6 || i == 9 || i == 10 || i == 17 || i == 18 || i == 20 || i == 23 || i == 24 || i == 27 || i == 29 || i == 30 || i == 37 || i == 38 || i == 41 || i == 42)
		{

		}
		else if (i == 5 || i == 21 || i == 22 || i == 25 || i == 26)
		{

		}
		else
		{
			EXPECT_LT(abs(ExtrapolationResultVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}

	EXPECT_LT(abs(ExtrapolationResultVector[1] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[4] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[6] - 0.5), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[9] + 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[10] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[17] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[18] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[20] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[23] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[24] + 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[27] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[29] + 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[30] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[37] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[38] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[41] + 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[42] - 2), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(ExtrapolationResultVector[5] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[21] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[22] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[25] + 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[26] - 2), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__extrapolatingData手算测试3(给定距离过大且有分离的外推域的情况)
TEST(Extrapolation, Extrapolation_extrapolationData3)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(3, 3, 1);

	vector<Real> ScalarFieldData(Res.x * Res.y * Res.z, UNKNOWN);

	ScalarFieldData[2] = 0;
	ScalarFieldData[6] = 1;

	CCellCenteredScalarField CCSSrcScalarField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), ScalarFieldData.data());
	CCellCenteredScalarField CCSDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCSSrcScalarField, CCSDisMarkersField, 200);

	thrust::host_vector<Real> ExtrapolationResultVector = CCSSrcScalarField.getConstGridData();

	EXPECT_LT(abs(ExtrapolationResultVector[0] - 0.5), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[1] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[2] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[3] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[4] - 0.5), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[5] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[6] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[7] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultVector[8] - 0.5), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__extrapolatingData手算测试4(FCVGrid)
TEST(Extrapolation, Extrapolation_extrapolationData4)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> VectorFieldDataX(ResX.x * ResX.y * ResX.z, UNKNOWN);
	vector<Real> VectorFieldDataY(ResY.x * ResY.y * ResY.z, UNKNOWN);
	vector<Real> VectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, UNKNOWN);

	VectorFieldDataX[0] = 1;
	VectorFieldDataX[1] = 2;
	VectorFieldDataY[0] = 3;
	VectorFieldDataZ[0] = 4;

	CFaceCenteredVectorField FCVSrcVectorField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VectorFieldDataX.data(), VectorFieldDataY.data(), VectorFieldDataZ.data());
	CFaceCenteredVectorField FCVDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(FCVSrcVectorField, FCVDisMarkersField, 3);

	thrust::host_vector<Real> ExtrapolationResultXVector = FCVSrcVectorField.getConstGridDataX();
	thrust::host_vector<Real> ExtrapolationResultYVector = FCVSrcVectorField.getConstGridDataY();
	thrust::host_vector<Real> ExtrapolationResultZVector = FCVSrcVectorField.getConstGridDataZ();

	for (int i = 0; i < ResX.x * ResX.y * ResX.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 3 && i != 5 && i != 6 && i != 7 && i != 10 && i != 11 && i != 20 && i != 21 && i != 22 && i != 25 && i != 26 && i != 40 && i != 41)
		{
			EXPECT_LT(abs(ExtrapolationResultXVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(ExtrapolationResultXVector[0] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[1] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[3] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[5] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[6] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[7] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[10] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[11] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[20] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[21] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[22] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[25] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[26] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[40] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[41] - 2), GRID_SOLVER_EPSILON);

	for (int i = 0; i < ResY.x * ResY.y * ResY.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 20 && i != 21 && i != 24 && i != 40)
		{
			EXPECT_LT(abs(ExtrapolationResultYVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(ExtrapolationResultYVector[0] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[1] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[2] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[4] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[5] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[8] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[20] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[21] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[24] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[40] - 3), GRID_SOLVER_EPSILON);

	for (int i = 0; i < ResZ.x * ResZ.y * ResZ.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 16 && i != 17 && i != 20 && i != 32)
		{
			EXPECT_LT(abs(ExtrapolationResultZVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(ExtrapolationResultZVector[0] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[1] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[2] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[4] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[5] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[8] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[16] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[17] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[20] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[32] - 4), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__extrapolatingData手算测试4(CCVGrid)
TEST(Extrapolation, Extrapolation_extrapolationData5)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);

	vector<Real> VectorFieldDataX(Res.x * Res.y * Res.z, UNKNOWN);
	vector<Real> VectorFieldDataY(Res.x * Res.y * Res.z, UNKNOWN);
	vector<Real> VectorFieldDataZ(Res.x * Res.y * Res.z, UNKNOWN);

	VectorFieldDataX[0] = 1;
	VectorFieldDataX[1] = 2;
	VectorFieldDataY[0] = 3;
	VectorFieldDataZ[0] = 4;

	CCellCenteredVectorField CCVSrcVectorField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VectorFieldDataX.data(), VectorFieldDataY.data(), VectorFieldDataZ.data());
	CCellCenteredVectorField CCVDisMarkersField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	extrapolatingDataInvoker(CCVSrcVectorField, CCVDisMarkersField, 3);

	thrust::host_vector<Real> ExtrapolationResultXVector = CCVSrcVectorField.getConstGridDataX();
	thrust::host_vector<Real> ExtrapolationResultYVector = CCVSrcVectorField.getConstGridDataY();
	thrust::host_vector<Real> ExtrapolationResultZVector = CCVSrcVectorField.getConstGridDataZ();

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 3 && i != 4 && i != 5 && i != 6 && i != 8 && i != 9 && i != 16 && i != 17 && i != 18 && i != 20 && i != 21 && i != 32 && i != 33)
		{
			EXPECT_LT(abs(ExtrapolationResultXVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(ExtrapolationResultXVector[0] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[1] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[3] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[4] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[5] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[6] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[8] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[9] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[16] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[17] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[18] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[20] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[21] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[32] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultXVector[33] - 2), GRID_SOLVER_EPSILON);

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 16 && i != 17 && i != 20 && i != 32)
		{
			EXPECT_LT(abs(ExtrapolationResultYVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(ExtrapolationResultYVector[0] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[1] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[2] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[4] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[5] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[8] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[16] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[17] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[20] - 3), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultYVector[32] - 3), GRID_SOLVER_EPSILON);

	for (int i = 0; i < Res.x * Res.y * Res.z; i++)
	{
		if (i != 0 && i != 1 && i != 2 && i != 4 && i != 5 && i != 8 && i != 16 && i != 17 && i != 20 && i != 32)
		{
			EXPECT_LT(abs(ExtrapolationResultZVector[i] - UNKNOWN), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(ExtrapolationResultZVector[0] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[1] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[2] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[4] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[5] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[8] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[16] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[17] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[20] - 4), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(ExtrapolationResultZVector[32] - 4), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

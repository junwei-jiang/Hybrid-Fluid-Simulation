#include "pch.h"
#include "PressureSolver.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>

//__buildMarkers手算测试1
TEST(PressureSolver, PressureSolver_buildMarkers1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(3, 3, 3);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 1.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[13] = 0.5;
	SolidSDFScalarFieldData[16] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 1, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	CCellCenteredScalarField MarkersResult = PressureSolver.getMarkers();
	thrust::host_vector<Real> MarkersResultVector = MarkersResult.getConstGridData();

	EXPECT_LT(abs(MarkersResultVector[0] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[1] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[2] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[3] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[4] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[5] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[6] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[7] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[8] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[9] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[10] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[11] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[12] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[13] - 1), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[14] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[15] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[16] - 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[17] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[18] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[19] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[20] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[21] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[22] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[23] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[24] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[25] - 2), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(MarkersResultVector[26] - 2), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildMarkers手算测试2
TEST(PressureSolver, PressureSolver_buildMarkers2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 1, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	CCellCenteredScalarField MarkersResult = PressureSolver.getMarkers();
	thrust::host_vector<Real> MarkersResultVector = MarkersResult.getConstGridData();

	for (int i = 0; i < 64; i++)
	{
		if (i == 29 || i == 30 || i == 45 || i == 46)
		{
			EXPECT_LT(abs(MarkersResultVector[i] - 0), GRID_SOLVER_EPSILON);
		}
		else if (i == 21 || i == 22 || i == 25 || i == 26 || i == 37 || i == 38 || i == 41 || i == 42)
		{
			EXPECT_LT(abs(MarkersResultVector[i] - 1), GRID_SOLVER_EPSILON);
		}
		else 
		{
			EXPECT_LT(abs(MarkersResultVector[i] - 2), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildFdmMatrixA手算测试1
TEST(PressureSolver, PressureSolver_buildFdmMatrixA1)
{
	CCudaContextManager::getInstance().initCudaContext();
	
	Vector3i Res = Vector3i(3, 3, 3);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 1.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[13] = 0.5;
	SolidSDFScalarFieldData[16] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 1, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	thrust::host_vector<Real> FdmMatrixAGPUResult = PressureSolver.getConstFdmMatrixA();

	Real InvY = 0.0025;
	for (int i = 0; i < 27; i++)
	{
		if (i == 13)
		{
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i] - InvY), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 1] - 0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 2] - 0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 3] - 0), GRID_SOLVER_EPSILON);
		}
		else
		{
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i] - 1), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 1] - 0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 2] - 0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 3] - 0), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildFdmMatrixA手算测试2
TEST(PressureSolver, PressureSolver_buildFdmMatrixA2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	thrust::host_vector<Real> FdmMatrixAGPUResult = PressureSolver.getConstFdmMatrixA();

	Real InvX = 2 * 0.01;
	Real InvY = 2 * 0.0025;
	Real InvZ = 2 * 0.00111111;

	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21] - InvX - InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 1] + InvX), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 2] + InvY), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 3] + InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22] - InvX - InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 2] + InvY), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 3] + InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25] - InvX - 2 * InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 1] + InvX), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 3] + InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26] - InvX - 2 * InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 3] + InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37] - InvX - InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 1] + InvX), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 2] + InvY), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38] - InvX - InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 2] + InvY), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41] - InvX - 2 * InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 1] + InvX), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42] - InvX - 2 * InvY - InvZ), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 3] + 0), GRID_SOLVER_EPSILON);

	for (int i = 0; i < 64; i++)
	{
		if (i != 29 && i != 30 && i != 45 && i != 46 && i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i] - 1), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 1] + 0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 2] + 0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 3] + 0), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildVectorb手算测试1
TEST(PressureSolver, PressureSolver_buildVectorb1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(3, 3, 3);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 1.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[13] = 0.5;
	SolidSDFScalarFieldData[16] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k + 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 1, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	CCuDenseVector TempVectorbResult = PressureSolver.getConstVectorb();
	thrust::host_vector<Real> VectorbResult = TempVectorbResult.getConstVectorValue();

	for (int i = 0; i < 27; i++)
	{
		if (i != 13)
		{
			EXPECT_LT(abs(VectorbResult[i] - 0), GRID_SOLVER_EPSILON);
		}
		else
		{
			EXPECT_LT(abs(VectorbResult[i] + 0.2), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//__buildVectorb手算测试2
TEST(PressureSolver, PressureSolver_buildVectorb2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 1, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	CCuDenseVector TempVectorbResult = PressureSolver.getConstVectorb();
	thrust::host_vector<Real> VectorbResult = TempVectorbResult.getConstVectorValue();

	for (int i = 0; i < 64; i++)
	{
		if (i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(VectorbResult[i] - 0), GRID_SOLVER_EPSILON);
		}
	}

	EXPECT_LT(abs(VectorbResult[21] + 0.26666666), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[22] + 0.06666666), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[25] + 0.21666666), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[26] + 0.01666666), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[37] + 0.13333333), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[38] - 0.06666666), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[41] + 0.08333333), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[42] - 0.11666666), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//MatrixFreePCG手算测试1
TEST(PressureSolver, PressureSolver_MatrixFreePCG1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	CCuDenseVector TempVectorxResult = PressureSolver.getConstVectorx();
	thrust::host_vector<Real> VectorxResult = TempVectorxResult.getConstVectorValue();
	vector<Real> XResult(VectorxResult.begin(), VectorxResult.end());

	thrust::host_vector<Real> FdmMatrixAGPUResult = PressureSolver.getConstFdmMatrixA();
	CCuDenseVector TempVectorbResult = PressureSolver.getConstVectorb();
	thrust::host_vector<Real> VectorbResult = TempVectorbResult.getConstVectorValue();


	EXPECT_LT(abs(VectorxResult[21] + 64.10625691), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[22] + 59.1624367), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[25] + 40.88984875), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[26] + 36.39546673), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[37] + 40.83756019), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[38] + 35.8937402), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[41] + 23.60453116), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[42] + 19.11014936), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//PressureSolver手算测试1（固体边界处的流体速度是否正确）
TEST(PressureSolver, PressureSolver_PressureSolver1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k + 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j + 1;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i + 1;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	thrust::host_vector<Real> FieldResultDataX = FCVFluidVelField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = FCVFluidVelField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = FCVFluidVelField.getConstGridDataZ();

	vector<Real> ResultXVector(FieldResultDataX.begin(), FieldResultDataX.end());
	vector<Real> ResultYVector(FieldResultDataY.begin(), FieldResultDataY.end());
	vector<Real> ResultZVector(FieldResultDataZ.begin(), FieldResultDataZ.end());

	EXPECT_LT(abs(FieldResultDataX[26] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[28] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[31] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[33] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[46] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[48] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[51] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[53] - 4.0), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataY[25] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[26] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[45] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[46] - 2.0), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataZ[21] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[22] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[25] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[26] - 2.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[53] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[54] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[57] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[58] - 4.0), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//PressureSolver手算测试2（流体内部的速度受边界影响后是否正确）
TEST(PressureSolver, PressureSolver_PressureSolver2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	thrust::host_vector<Real> FieldResultDataX = FCVFluidVelField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = FCVFluidVelField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = FCVFluidVelField.getConstGridDataZ();
	
	vector<Real> ResultXVector(FieldResultDataX.begin(), FieldResultDataX.end());
	vector<Real> ResultYVector(FieldResultDataY.begin(), FieldResultDataY.end());
	vector<Real> ResultZVector(FieldResultDataZ.begin(), FieldResultDataZ.end());

	EXPECT_LT(abs(FieldResultDataX[27] - 1.011228), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[32] - 1.101123), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[47] - 1.011228), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[52] - 1.101123), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataY[29] + 0.3216407), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[30] + 0.2766933), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[33] + 1.088985), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[34] + 0.6395467), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[49] - 0.276697), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[50] - 0.321641), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[53] - 0.639547), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[54] - 1.088985), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataZ[37] - 0.4487535), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[38] - 0.448756), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[41] - 0.8475453), 10 * GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[42] - 0.8476455), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//PressureSolver手算测试3（整体测试，包括线性系统构建和求解以及应用压力修正速度以及修正后的速度场是否无散）
TEST(PressureSolver, PressureSolver_PressureSolver3)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 4 - k;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(1, 1, 1));

	PressureSolver.solvePressure(FCVFluidVelField, 1, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	thrust::host_vector<Real> FdmMatrixAGPUResult = PressureSolver.getConstFdmMatrixA();

	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21] - 3.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22] - 3.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37] - 3.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38] - 3.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 3] + 0), GRID_SOLVER_EPSILON);

	for (int i = 0; i < 64; i++)
	{
		if (i != 29 && i != 30 && i != 45 && i != 46 && i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i] - 1.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 1] + 0.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 2] + 0.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 3] + 0.0), GRID_SOLVER_EPSILON);
		}
	}

	CCuDenseVector TempVectorbResult = PressureSolver.getConstVectorb();
	thrust::host_vector<Real> VectorbResult = TempVectorbResult.getConstVectorValue();

	for (int i = 0; i < 64; i++)
	{
		if (i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(VectorbResult[i] - 0), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(VectorbResult[21] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[22] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[25] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[26] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[37] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[38] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[41] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[42] - 1.0), GRID_SOLVER_EPSILON);

	CCuDenseVector TempVectorxResult = PressureSolver.getConstVectorx();
	thrust::host_vector<Real> VectorxResult = TempVectorxResult.getConstVectorValue();
	vector<Real> XResult(VectorxResult.begin(), VectorxResult.end());

	for (int i = 0; i < 64; i++)
	{
		if (i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(VectorxResult[i] - 0), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(VectorxResult[21] + 0.45454545), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[22] - 0.45454545), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[25] + 0.36363636), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[26] - 0.36363636), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[37] + 0.45454545), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[38] - 0.45454545), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[41] + 0.36363636), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[42] - 0.36363636), GRID_SOLVER_EPSILON);

	thrust::host_vector<Real> FieldResultDataX = FCVFluidVelField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = FCVFluidVelField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = FCVFluidVelField.getConstGridDataZ();
	vector<Real> ResultXVector(FieldResultDataX.begin(), FieldResultDataX.end());
	vector<Real> ResultYVector(FieldResultDataY.begin(), FieldResultDataY.end());
	vector<Real> ResultZVector(FieldResultDataZ.begin(), FieldResultDataZ.end());

	EXPECT_LT(abs(FieldResultDataX[26] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[28] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[31] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[33] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[46] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[48] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[51] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[53] - 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[27] - 1.090909), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[32] - 1.272727), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[47] - 1.090909), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[52] - 1.272727), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataY[29] + 0.090909), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[30] - 0.090909), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[33] + 0.363636), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[34] - 0.363636), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[49] + 0.090909), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[50] - 0.090909), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[53] + 0.363636), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[54] - 0.363636), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataZ[37] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[38] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[41] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[42] - 0.0), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//PressureSolver手算测试4（整体测试，没有固体的情况下流体速度是否正常）
TEST(PressureSolver, PressureSolver_PressureSolver4)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}

	FluidSDFScalarFieldData[17] = 0.5;
	FluidSDFScalarFieldData[18] = 0.5;
	FluidSDFScalarFieldData[33] = 0.5;
	FluidSDFScalarFieldData[34] = 0.5;

	SolidSDFScalarFieldData[17] = 0.5;
	SolidSDFScalarFieldData[18] = 0.5;
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[33] = 0.5;
	SolidSDFScalarFieldData[34] = 0.5;
	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 0;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = -9.8;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(1, 1, 1));

	PressureSolver.solvePressure(FCVFluidVelField, 1, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	thrust::host_vector<Real> FdmMatrixAGPUResult = PressureSolver.getConstFdmMatrixA();

	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 21 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 22 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 25 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 26 + 3] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 37 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 2] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 38 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 1] + 1.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 41 + 3] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42] - 4.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 1] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 2] + 0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FdmMatrixAGPUResult[4 * 42 + 3] + 0), GRID_SOLVER_EPSILON);

	for (int i = 0; i < 64; i++)
	{
		if (i != 29 && i != 30 && i != 45 && i != 46 && i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i] - 1.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 1] + 0.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 2] + 0.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(FdmMatrixAGPUResult[4 * i + 3] + 0.0), GRID_SOLVER_EPSILON);
		}
	}

	CCuDenseVector TempVectorbResult = PressureSolver.getConstVectorb();
	thrust::host_vector<Real> VectorbResult = TempVectorbResult.getConstVectorValue();

	for (int i = 0; i < 64; i++)
	{
		if (i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(VectorbResult[i] - 0), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(VectorbResult[21] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[22] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[25] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[26] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[37] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[38] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[41] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorbResult[42] - 0.0), GRID_SOLVER_EPSILON);

	CCuDenseVector TempVectorxResult = PressureSolver.getConstVectorx();
	thrust::host_vector<Real> VectorxResult = TempVectorxResult.getConstVectorValue();
	vector<Real> XResult(VectorxResult.begin(), VectorxResult.end());

	for (int i = 0; i < 64; i++)
	{
		if (i != 21 && i != 22 && i != 25 && i != 26 && i != 37 && i != 38 && i != 41 && i != 42)
		{
			EXPECT_LT(abs(VectorxResult[i] - 0), GRID_SOLVER_EPSILON);
		}
	}
	EXPECT_LT(abs(VectorxResult[21] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[22] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[25] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[26] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[37] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[38] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[41] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(VectorxResult[42] - 0.0), GRID_SOLVER_EPSILON);

	thrust::host_vector<Real> FieldResultDataX = FCVFluidVelField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = FCVFluidVelField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = FCVFluidVelField.getConstGridDataZ();
	vector<Real> ResultXVector(FieldResultDataX.begin(), FieldResultDataX.end());
	vector<Real> ResultYVector(FieldResultDataY.begin(), FieldResultDataY.end());
	vector<Real> ResultZVector(FieldResultDataZ.begin(), FieldResultDataZ.end());

	EXPECT_LT(abs(FieldResultDataX[26] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[28] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[31] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[33] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[46] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[48] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[51] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[53] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[27] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[32] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[47] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataX[52] - 0.0), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataY[29] + 9.8), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[30] + 9.8), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[33] + 9.8), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[34] + 9.8), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[49] + 9.8), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[50] + 9.8), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[53] + 9.8), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataY[54] + 9.8), GRID_SOLVER_EPSILON);

	EXPECT_LT(abs(FieldResultDataZ[37] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[38] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[41] - 0.0), GRID_SOLVER_EPSILON);
	EXPECT_LT(abs(FieldResultDataZ[42] - 0.0), GRID_SOLVER_EPSILON);

	CCudaContextManager::getInstance().freeCudaContext();
}

//PressureSolver手算测试2（移动边界测试）
TEST(PressureSolver, PressureSolver_PressureSolver5)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = j - 2.5;
				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -0.5;
			}
		}
	}
	
	SolidSDFScalarFieldData[21] = 0.5;
	SolidSDFScalarFieldData[22] = 0.5;
	SolidSDFScalarFieldData[25] = 0.5;
	SolidSDFScalarFieldData[26] = 0.5;
	SolidSDFScalarFieldData[29] = 0.5;
	SolidSDFScalarFieldData[30] = 0.5;

	SolidSDFScalarFieldData[37] = 0.5;
	SolidSDFScalarFieldData[38] = 0.5;
	SolidSDFScalarFieldData[41] = 0.5;
	SolidSDFScalarFieldData[42] = 0.5;
	SolidSDFScalarFieldData[45] = 0.5;
	SolidSDFScalarFieldData[46] = 0.5;

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				FluidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 0;
				SolidVelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				FluidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
				SolidVelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				FluidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
				SolidVelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidVelVectorFieldDataX.data(), FluidVelVectorFieldDataY.data(), FluidVelVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), FluidSDFScalarFieldData.data());

	CPressureSolver PressureSolver(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);
	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);
	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);
	PressureSolver.solvePressure(FCVFluidVelField, 2, FCVSolidVelField, CCSSolidSDFField, CCSFluidSDFField);

	thrust::host_vector<Real> FieldResultDataX = FCVFluidVelField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = FCVFluidVelField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = FCVFluidVelField.getConstGridDataZ();

	vector<Real> ResultXVector(FieldResultDataX.begin(), FieldResultDataX.end());
	vector<Real> ResultYVector(FieldResultDataY.begin(), FieldResultDataY.end());
	vector<Real> ResultZVector(FieldResultDataZ.begin(), FieldResultDataZ.end());

	CCudaContextManager::getInstance().freeCudaContext();
}
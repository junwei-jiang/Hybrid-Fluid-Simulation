#include "pch.h"
#include "EulerParticles.h"
#include "EulerParticlesTool.cuh"
#include "CudaContextManager.h"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>

//GenerateParticles≤‚ ‘
TEST(ParticlesAdvection, GenerateParticles)
{
	CCudaContextManager::getInstance().initCudaContext();

	//4°¡4°¡4Õ¯∏Ò£¨÷–º‰∞À∏ˆÕ¯∏Ò «ÀÆ£¨√ø∏ˆÕ¯∏Ò∞À∏ˆ¡£◊”
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfPerGrid = 8;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[21] = -1.0; SolidSDFScalarFieldData[21] = 1.0;
		FluidSDFScalarFieldData[22] = -1.0; SolidSDFScalarFieldData[22] = 1.0;
		FluidSDFScalarFieldData[25] = -1.0; SolidSDFScalarFieldData[25] = 1.0;
		FluidSDFScalarFieldData[26] = -1.0; SolidSDFScalarFieldData[26] = 1.0;
		FluidSDFScalarFieldData[37] = -1.0; SolidSDFScalarFieldData[37] = 1.0;
		FluidSDFScalarFieldData[38] = -1.0; SolidSDFScalarFieldData[38] = 1.0;
		FluidSDFScalarFieldData[41] = -1.0; SolidSDFScalarFieldData[41] = 1.0;
		FluidSDFScalarFieldData[42] = -1.0; SolidSDFScalarFieldData[42] = 1.0;

		vector<Vector3> FluidGridIndex
		{
			Vector3(1.0, 1.0, 1.0),
			Vector3(2.0, 1.0, 1.0),
			Vector3(1.0, 2.0, 1.0),
			Vector3(2.0, 2.0, 1.0),
			Vector3(1.0, 1.0, 2.0),
			Vector3(2.0, 1.0, 2.0),
			Vector3(1.0, 2.0, 2.0),
			Vector3(2.0, 2.0, 2.0)
		};

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> ParticlesPos = EulerParticles.getParticlesPos();
		vector<Real> ParticlesResult(ParticlesPos.begin(), ParticlesPos.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 8 * NumOfPerGrid);

		for (int i = 0; i < 8; i++)
		{
			Vector3 CurFluidGridMin = Origin + FluidGridIndex[i] * Spacing;
			Vector3 CurFluidGridMax = CurFluidGridMin + Spacing;
			for (UInt k = 0; k < NumOfPerGrid; k++)
			{
				UInt CurParticlesIndex = i * NumOfPerGrid + k;
				EXPECT_GE(ParticlesPos[3 * CurParticlesIndex], CurFluidGridMin.x);
				EXPECT_GE(ParticlesPos[3 * CurParticlesIndex + 1], CurFluidGridMin.y);
				EXPECT_GE(ParticlesPos[3 * CurParticlesIndex + 2], CurFluidGridMin.z);
				EXPECT_LE(ParticlesPos[3 * CurParticlesIndex], CurFluidGridMax.x);
				EXPECT_LE(ParticlesPos[3 * CurParticlesIndex + 1], CurFluidGridMax.y);
				EXPECT_LE(ParticlesPos[3 * CurParticlesIndex + 2], CurFluidGridMax.z);
			}
		}
	}

	//2°¡2°¡1Õ¯∏Ò£¨µ⁄“ª∏ˆÕ¯∏Ò «ÀÆ£¨√ø∏ˆÕ¯∏ÒŒÂ∏ˆ¡£◊”
	{
		Vector3i Res = Vector3i(2, 2, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);
		UInt NumOfPerGrid = 5;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[0] = -1.0; SolidSDFScalarFieldData[0] = 1.0;

		Vector3 FluidGridIndex = Vector3(0, 0, 0);

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> ParticlesPos = EulerParticles.getParticlesPos();
		vector<Real> ParticlesResult(ParticlesPos.begin(), ParticlesPos.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 1 * NumOfPerGrid);

		for (UInt k = 0; k < NumOfPerGrid; k++)
		{
			EXPECT_GE(ParticlesPos[3 * k], 0.0);
			EXPECT_GE(ParticlesPos[3 * k + 1], 20.0);
			EXPECT_GE(ParticlesPos[3 * k + 2], -30.0);
			EXPECT_LE(ParticlesPos[3 * k], 30.0);
			EXPECT_LE(ParticlesPos[3 * k + 1], 40.0);
			EXPECT_LE(ParticlesPos[3 * k + 2], -20.0);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//transferParticles2CCSGrid≤‚ ‘
TEST(ParticlesAdvection, transferParticles2CCSGrid)
{
	CCudaContextManager::getInstance().initCudaContext();

	//SUM KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfPerGrid = 8;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[21] = -1.0; SolidSDFScalarFieldData[21] = 1.0;
		FluidSDFScalarFieldData[22] = -1.0; SolidSDFScalarFieldData[22] = 1.0;
		FluidSDFScalarFieldData[25] = -1.0; SolidSDFScalarFieldData[25] = 1.0;
		FluidSDFScalarFieldData[26] = -1.0; SolidSDFScalarFieldData[26] = 1.0;
		FluidSDFScalarFieldData[37] = -1.0; SolidSDFScalarFieldData[37] = 1.0;
		FluidSDFScalarFieldData[38] = -1.0; SolidSDFScalarFieldData[38] = 1.0;
		FluidSDFScalarFieldData[41] = -1.0; SolidSDFScalarFieldData[41] = 1.0;
		FluidSDFScalarFieldData[42] = -1.0; SolidSDFScalarFieldData[42] = 1.0;

		vector<Vector3> FluidGridIndex
		{
			Vector3(1.0, 1.0, 1.0),
			Vector3(2.0, 1.0, 1.0),
			Vector3(1.0, 2.0, 1.0),
			Vector3(2.0, 2.0, 1.0),
			Vector3(1.0, 1.0, 2.0),
			Vector3(2.0, 1.0, 2.0),
			Vector3(1.0, 2.0, 2.0),
			Vector3(2.0, 2.0, 2.0)
		};

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 8 * NumOfPerGrid);

		vector<Real> ParticlesColor(8 * NumOfPerGrid, 10);
		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 8 * NumOfPerGrid);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec, 
			CCSParticlesColorField, 
			CCSGridWeightField, 
			EPGTransferAlgorithm::P2GSUM
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					UInt CurIndex = z * Res.x * Res.y + y * Res.x + x;
					if (
						CurIndex != 21 && CurIndex != 22 && CurIndex != 25 && CurIndex != 26 &&
						CurIndex != 37 && CurIndex != 38 && CurIndex != 41 && CurIndex != 42)
					{
						EXPECT_LT(abs(TempParticlesColorResult[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesColorResult[CurIndex] - 80.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	//SUM KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(2, 2, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);
		UInt NumOfPerGrid = 5;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[0] = -1.0; SolidSDFScalarFieldData[0] = 1.0;

		Vector3 FluidGridIndex = Vector3(0, 0, 0);

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> ParticlesPos = EulerParticles.getParticlesPos();
		vector<Real> ParticlesResult(ParticlesPos.begin(), ParticlesPos.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 1 * NumOfPerGrid);

		vector<Real> ParticlesColor(1 * NumOfPerGrid);
		for (UInt i = 0; i < NumOfPerGrid; i++)
		{
			ParticlesColor[i] = i + 1;
		}
		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 8 * NumOfPerGrid);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField, 
			EPGTransferAlgorithm::P2GSUM
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		Vector3 FluidCellCenter = Origin + 0.5 * Spacing;

		EXPECT_LT(abs(TempParticlesColorResult[0] - 15.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[1] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[2] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[3] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Linear KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -25.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -15.0;
		vector<Real> ParticlesColor(2);
		ParticlesColor[0] = 64;
		ParticlesColor[1] = 128;
		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 2);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField, 
			EPGTransferAlgorithm::LINEAR
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		EXPECT_LT(abs(TempParticlesColorResult[0] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[1] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[2] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[4] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[5] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[6] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[7] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[8] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[9] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[10] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[11] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[13] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[14] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[15] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Linear KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesColor(2);
		ParticlesColor[0] = 64;
		ParticlesColor[1] = 128;
		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 2);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField, 
			EPGTransferAlgorithm::LINEAR
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		for (int i = 0; i < 16; i++)
		{
			if (i != 5 && i != 6 && i != 9 && i != 10)
			{
				EXPECT_LT(abs(TempParticlesColorResult[i] - 0.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesColorResult[i] - 96.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Quadratic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -25.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -15.0;
		vector<Real> ParticlesColor(2);
		ParticlesColor[0] = 64;
		ParticlesColor[1] = 128;
		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 2);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField, 
			EPGTransferAlgorithm::QUADRATIC
		);
		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField,
			EPGTransferAlgorithm::QUADRATIC
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		EXPECT_LT(abs(TempParticlesColorResult[0] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[1] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[2] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[4] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[5] - 38.0 * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[6] - 96.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[7] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[8] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[9] - 96.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[10] - 73.0 * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[11] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[13] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[14] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[15] - 128.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesColor(2);
		ParticlesColor[0] = 64;
		ParticlesColor[1] = 128;
		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 2);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField, 
			EPGTransferAlgorithm::QUADRATIC
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		for (int i = 0; i < 16; i++)
		{
			if (i != 5 && i != 6 && i != 9 && i != 10)
			{
				EXPECT_LT(abs(TempParticlesColorResult[i] - 0.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesColorResult[i] - 96.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Cubic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -25.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -15.0;
		vector<Real> ParticlesColor(2);
		ParticlesColor[0] = 64;
		ParticlesColor[1] = 128;
		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 2);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField, 
			EPGTransferAlgorithm::CUBIC
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		EXPECT_LT(abs(TempParticlesColorResult[0] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[1] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[2] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[4] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[5] - (64.0 * 16.0 + 128.0) / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[6] - 96.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[7] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[8] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[9] - 96.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[10] - (64.0 + 128.0 * 16.0) / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[11] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[13] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[14] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResult[15] - 128.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesColor(2);
		ParticlesColor[0] = 64;
		ParticlesColor[1] = 128;
		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::device_vector<Real> ParticlesColorVec;
		assignDeviceVectorReal(ParticlesColorVec, ParticlesColor.data(), ParticlesColor.data() + 2);

		CCellCenteredScalarField CCSParticlesColorField(Res, Origin, Spacing);
		CCellCenteredScalarField CCSGridWeightField(Res, Origin, Spacing);

		tranferParticles2CCSFieldInvoker
		(
			EulerParticles.getParticlesPos(),
			ParticlesColorVec,
			CCSParticlesColorField,
			CCSGridWeightField, 
			EPGTransferAlgorithm::CUBIC
		);

		thrust::host_vector<Real> TempParticlesColorResult = CCSParticlesColorField.getConstGridData();
		vector<Real> ParticlesColorResult(TempParticlesColorResult.begin(), TempParticlesColorResult.end());

		for (int i = 0; i < 16; i++)
		{
			EXPECT_LT(abs(TempParticlesColorResult[i] - 96.0), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//transferParticles2CCVGrid≤‚ ‘
TEST(ParticlesAdvection, transferParticles2CCVGrid)
{
	CCudaContextManager::getInstance().initCudaContext();

	//SUM KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfPerGrid = 8;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[21] = -1.0; SolidSDFScalarFieldData[21] = 1.0;
		FluidSDFScalarFieldData[22] = -1.0; SolidSDFScalarFieldData[22] = 1.0;
		FluidSDFScalarFieldData[25] = -1.0; SolidSDFScalarFieldData[25] = 1.0;
		FluidSDFScalarFieldData[26] = -1.0; SolidSDFScalarFieldData[26] = 1.0;
		FluidSDFScalarFieldData[37] = -1.0; SolidSDFScalarFieldData[37] = 1.0;
		FluidSDFScalarFieldData[38] = -1.0; SolidSDFScalarFieldData[38] = 1.0;
		FluidSDFScalarFieldData[41] = -1.0; SolidSDFScalarFieldData[41] = 1.0;
		FluidSDFScalarFieldData[42] = -1.0; SolidSDFScalarFieldData[42] = 1.0;

		vector<Vector3> FluidGridIndex
		{
			Vector3(1.0, 1.0, 1.0),
			Vector3(2.0, 1.0, 1.0),
			Vector3(1.0, 2.0, 1.0),
			Vector3(2.0, 2.0, 1.0),
			Vector3(1.0, 1.0, 2.0),
			Vector3(2.0, 1.0, 2.0),
			Vector3(1.0, 2.0, 2.0),
			Vector3(2.0, 2.0, 2.0)
		};

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 8 * NumOfPerGrid);

		vector<Real> ParticlesColor(3 * 8 * NumOfPerGrid);
		for (int i = 0; i < 8 * NumOfPerGrid; i++)
		{
			ParticlesColor[3 * i] = 1;
			ParticlesColor[3 * i + 1] = 2;
			ParticlesColor[3 * i + 2] = 3;
		}
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::P2GSUM);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					UInt CurIndex = z * Res.x * Res.y + y * Res.x + x;
					if (
						CurIndex != 21 && CurIndex != 22 && CurIndex != 25 && CurIndex != 26 &&
						CurIndex != 37 && CurIndex != 38 && CurIndex != 41 && CurIndex != 42)
					{
						EXPECT_LT(abs(TempParticlesColorResultX[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(TempParticlesColorResultY[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(TempParticlesColorResultZ[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesColorResultX[CurIndex] - 8.0), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(TempParticlesColorResultY[CurIndex] - 16.0), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(TempParticlesColorResultZ[CurIndex] - 24.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	//SUM KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(2, 2, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);
		UInt NumOfPerGrid = 5;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[0] = -1.0; SolidSDFScalarFieldData[0] = 1.0;

		Vector3 FluidGridIndex = Vector3(0, 0, 0);

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> ParticlesPos = EulerParticles.getParticlesPos();
		vector<Real> ParticlesResult(ParticlesPos.begin(), ParticlesPos.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 1 * NumOfPerGrid);

		vector<Real> ParticlesColor(3 * 1 * NumOfPerGrid);
		for (UInt i = 0; i < 3 * NumOfPerGrid; i++)
		{
			ParticlesColor[i] = i + 1;
		}
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCvGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCvGridWeightField, EPGTransferAlgorithm::P2GSUM);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesColorResultX[0] - 35.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[1] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[2] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[3] - 0.0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(TempParticlesColorResultY[0] - 40.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[1] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[2] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[3] - 0.0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(TempParticlesColorResultZ[0] - 45.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[1] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[2] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[3] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Linear KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -25.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -15.0;
		vector<Real> ParticlesColor(6);
		ParticlesColor[0] = 2;
		ParticlesColor[1] = 4;
		ParticlesColor[2] = 6;
		ParticlesColor[3] = 8;
		ParticlesColor[4] = 10;
		ParticlesColor[5] = 12;
		CEulerParticles EulerParticles(2, ParticlesPos.data());
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		for (int i = 0; i < 16; i++)
		{
			if (i == 5)
			{
				EXPECT_LT(abs(TempParticlesColorResultX[i] - 2.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultY[i] - 4.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultZ[i] - 6.0), GRID_SOLVER_EPSILON);
			}
			else if (i == 10)
			{
				EXPECT_LT(abs(TempParticlesColorResultX[i] - 8.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultY[i] - 10.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultZ[i] - 12.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesColorResultX[i] - 0.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultY[i] - 0.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultZ[i] - 0.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Linear KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesColor(6);
		ParticlesColor[0] = 2;
		ParticlesColor[1] = 4;
		ParticlesColor[2] = 6;
		ParticlesColor[3] = 8;
		ParticlesColor[4] = 10;
		ParticlesColor[5] = 12;
		CEulerParticles EulerParticles(2, ParticlesPos.data());
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		for (int i = 0; i < 16; i++)
		{
			if (i == 5 || i == 6 || i == 9 || i == 10)
			{
				EXPECT_LT(abs(TempParticlesColorResultX[i] - 5.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultY[i] - 7.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultZ[i] - 9.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesColorResultX[i] - 0.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultY[i] - 0.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultZ[i] - 0.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Quadratic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -25.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -15.0;
		vector<Real> ParticlesColor(6);
		ParticlesColor[0] = 2;
		ParticlesColor[1] = 4;
		ParticlesColor[2] = 6;
		ParticlesColor[3] = 8;
		ParticlesColor[4] = 10;
		ParticlesColor[5] = 12;
		CEulerParticles EulerParticles(2, ParticlesPos.data());
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::QUADRATIC);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesColorResultX[0] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[1] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[2] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[4] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[5] - (9.0 / 16.0 * 2.0 + 1.0 / 64.0 * 8.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[6] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[7] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[8] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[9] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[10] - (1.0 / 64.0 * 2.0 + 9.0 / 16.0 * 8.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[11] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[13] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[14] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[15] - 8.0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(TempParticlesColorResultY[0] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[1] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[2] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[4] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[5] - (9.0 / 16.0 * 4.0 + 1.0 / 64.0 * 10.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[6] - 7.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[7] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[8] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[9] - 7.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[10] - (1.0 / 64.0 * 4.0 + 9.0 / 16.0 * 10.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[11] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[13] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[14] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[15] - 10.0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(TempParticlesColorResultZ[0] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[1] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[2] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[4] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[5] - (9.0 / 16.0 * 6.0 + 1.0 / 64.0 * 12.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[6] - 9.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[7] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[8] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[9] - 9.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[10] - (1.0 / 64.0 * 6.0 + 9.0 / 16.0 * 12.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[11] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[13] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[14] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[15] - 12.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesColor(6);
		ParticlesColor[0] = 2;
		ParticlesColor[1] = 4;
		ParticlesColor[2] = 6;
		ParticlesColor[3] = 8;
		ParticlesColor[4] = 10;
		ParticlesColor[5] = 12;
		CEulerParticles EulerParticles(2, ParticlesPos.data());
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		for (int i = 0; i < 16; i++)
		{
			if (i != 5 && i != 6 && i != 9 && i != 10)
			{
				EXPECT_LT(abs(TempParticlesColorResultX[i] - 0.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultY[i] - 0.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultZ[i] - 0.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesColorResultX[i] - 5.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultY[i] - 7.0), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(TempParticlesColorResultZ[i] - 9.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Cubic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -25.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -15.0;
		vector<Real> ParticlesColor(6);
		ParticlesColor[0] = 2;
		ParticlesColor[1] = 4;
		ParticlesColor[2] = 6;
		ParticlesColor[3] = 8;
		ParticlesColor[4] = 10;
		ParticlesColor[5] = 12;
		CEulerParticles EulerParticles(2, ParticlesPos.data());
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesColorResultX[0] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[1] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[2] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[4] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[5] - (4.0 / 9.0 * 2.0 + 1.0 / 36.0 * 8.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[6] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[7] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[8] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[9] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[10] - (1.0 / 36.0 * 2.0 + 4.0 / 9.0 * 8.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[11] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[13] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[14] - 8.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultX[15] - 8.0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(TempParticlesColorResultY[0] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[1] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[2] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[4] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[5] - (4.0 / 9.0 * 4.0 + 1.0 / 36.0 * 10.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[6] - 7.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[7] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[8] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[9] - 7.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[10] - (1.0 / 36.0 * 4.0 + 4.0 / 9.0 * 10.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[11] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[13] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[14] - 10.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultY[15] - 10.0), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(TempParticlesColorResultZ[0] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[1] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[2] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[4] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[5] - (4.0 / 9.0 * 6.0 + 1.0 / 36.0 * 12.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[6] - 9.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[7] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[8] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[9] - 9.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[10] - (1.0 / 36.0 * 6.0 + 4.0 / 9.0 * 12.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[11] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[13] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[14] - 12.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesColorResultZ[15] - 12.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesColor(6);
		ParticlesColor[0] = 2;
		ParticlesColor[1] = 4;
		ParticlesColor[2] = 6;
		ParticlesColor[3] = 8;
		ParticlesColor[4] = 10;
		ParticlesColor[5] = 12;
		CEulerParticles EulerParticles(2, ParticlesPos.data());
		EulerParticles.setParticlesColor(ParticlesColor.data());

		CCellCenteredVectorField CCVParticlesColorField(Res, Origin, Spacing);
		CCellCenteredVectorField CCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesColor2Field(CCVParticlesColorField, CCVGridWeightField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesColorResultX = CCVParticlesColorField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesColorResultY = CCVParticlesColorField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesColorResultZ = CCVParticlesColorField.getConstGridDataZ();

		for (int i = 0; i < 16; i++)
		{
			EXPECT_LT(abs(TempParticlesColorResultX[i] - 5.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(TempParticlesColorResultY[i] - 7.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(TempParticlesColorResultZ[i] - 9.0), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//transferParticles2FCVGrid≤‚ ‘
TEST(ParticlesAdvection, transferParticles2FCVGrid)
{
	CCudaContextManager::getInstance().initCudaContext();

	//SUM KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfparticles = 16;
		vector<Real> ParticlesVelCPU(3 * NumOfparticles);

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -16.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < 1; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 4; x++)
				{
					ParticlesPos[3 * (z * 4 * 4 + y * 4 + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * 4 * 4 + y * 4 + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * 4 * 4 + y * 4 + x) + 2] = BasePos.z;
				}
			}
		}

		for (int i = 0; i < NumOfparticles; i++)
		{
			ParticlesVelCPU[3 * i] = 1;
			ParticlesVelCPU[3 * i + 1] = 2;
			ParticlesVelCPU[3 * i + 2] = 3;
		}

		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data(), ParticlesVelCPU.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::P2GSUM);

		thrust::host_vector<Real> TempParticlesVelResultX = FCVParticlesVelField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		vector<Real> tt(TempParticlesVelResultX.begin(), TempParticlesVelResultX.end());

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					UInt CurIndex = z * ResX.x * ResX.y + y * ResX.x + x;
					if(CurIndex == 6 || CurIndex == 8 || CurIndex == 11 || CurIndex == 13)
					{
						EXPECT_LT(abs(TempParticlesVelResultX[CurIndex] - 2.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 7 || CurIndex == 12)
					{
						EXPECT_LT(abs(TempParticlesVelResultX[CurIndex] - 4.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResultX[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}

		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					UInt CurIndex = z * ResY.x * ResY.y + y * ResY.x + x;
					if (CurIndex == 5 || CurIndex == 6 || CurIndex == 13 || CurIndex == 14)
					{
						EXPECT_LT(abs(TempParticlesVelResultY[CurIndex] - 4.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 9 || CurIndex == 10)
					{
						EXPECT_LT(abs(TempParticlesVelResultY[CurIndex] - 8.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResultY[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}

		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					UInt CurIndex = z * ResZ.x * ResZ.y + y * ResZ.x + x;
					if (CurIndex == 5 || CurIndex == 6 || CurIndex == 9 || CurIndex == 10)
					{
						EXPECT_LT(abs(TempParticlesVelResultZ[CurIndex] - 12.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResultZ[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	{}

	//SUM KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfparticles = 16;
		vector<Real> ParticlesVelCPU(3 * NumOfparticles);

		Vector3 BasePos = Vector3(-6.0, (Origin.y + 1.25 * Spacing.y), (Origin.z + 1.25 * Spacing.z));
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < 4; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 1; x++)
				{
					ParticlesPos[3 * (z * 1 * 4 + y * 1 + x)] = BasePos.x;
					ParticlesPos[3 * (z * 1 * 4 + y * 1 + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * 1 * 4 + y * 1 + x) + 2] = BasePos.z + z * 0.5 * Spacing.z;
				}
			}
		}

		for (int i = 0; i < NumOfparticles; i++)
		{
			ParticlesVelCPU[3 * i] = 1;
			ParticlesVelCPU[3 * i + 1] = 2;
			ParticlesVelCPU[3 * i + 2] = 3;
		}

		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data(), ParticlesVelCPU.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::P2GSUM);

		thrust::host_vector<Real> TempParticlesVelResultX = FCVParticlesVelField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		vector<Real> tt(TempParticlesVelResultX.begin(), TempParticlesVelResultX.end());

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					UInt CurIndex = z * ResX.x * ResX.y + y * ResX.x + x;
					if (CurIndex == 10 || CurIndex == 12 || CurIndex == 18 || CurIndex == 20)
					{
						EXPECT_LT(abs(TempParticlesVelResultX[CurIndex] - 4.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResultX[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
		
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					UInt CurIndex = z * ResY.x * ResY.y + y * ResY.x + x;
					if (CurIndex == 6 || CurIndex == 8 || CurIndex == 11 || CurIndex == 13)
					{
						EXPECT_LT(abs(TempParticlesVelResultY[CurIndex] - 4.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 7 || CurIndex == 12)
					{
						EXPECT_LT(abs(TempParticlesVelResultY[CurIndex] - 8.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResultY[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}

		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					UInt CurIndex = z * ResZ.x * ResZ.y + y * ResZ.x + x;
					if (CurIndex == 5 || CurIndex == 6 || CurIndex == 13 || CurIndex == 14)
					{
						EXPECT_LT(abs(TempParticlesVelResultZ[CurIndex] - 6.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 9 || CurIndex == 10)
					{
						EXPECT_LT(abs(TempParticlesVelResultZ[CurIndex] - 12.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResultZ[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	{}

	//Linear KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 30.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 60.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesVelResultX = FCVParticlesVelField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		for (int i = 0; i < 20; i++)
		{
			if (i == 6)
			{
				EXPECT_LT(abs(TempParticlesVelResultX[i] - 1.0), GRID_SOLVER_EPSILON);
			}
			else if (i == 12)
			{
				EXPECT_LT(abs(TempParticlesVelResultX[i] - 2.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesVelResultX[i] - 0.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Linear KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 40.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -26.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();

		for (int i = 0; i < 20; i++)
		{
			if (i == 5)
			{
				EXPECT_LT(abs(TempParticlesVelResultY[i] - 3.0), GRID_SOLVER_EPSILON);
			}
			else if (i == 10)
			{
				EXPECT_LT(abs(TempParticlesVelResultY[i] - 4.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesVelResultY[i] - 0.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Linear KernelFunction≤‚ ‘3
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -20.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		for (int i = 0; i < 20; i++)
		{
			if (i == 5)
			{
				EXPECT_LT(abs(TempParticlesVelResultZ[i] - 5.0), GRID_SOLVER_EPSILON);
			}
			else if (i == 10)
			{
				EXPECT_LT(abs(TempParticlesVelResultZ[i] - 6.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(TempParticlesVelResultZ[i] - 0.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Quadratic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 30.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 60.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::QUADRATIC);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesVelResultX = FCVParticlesVelField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesVelResultX[0] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[1] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[2] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[4] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[5] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[6] - (9.0 / 16.0 * 1.0 + 1.0 / 64.0 * 2.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[7] - 1.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[8] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[9] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[10] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[11] - 1.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[12] - (1.0 / 64.0 * 1.0 + 9.0 / 16.0 * 2.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[13] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[14] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[15] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[16] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[17] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[18] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[19] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 40.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -26.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesVelResultX = FCVParticlesVelField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesVelResultY[0] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[1] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[2] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[4] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[5] - (9.0 / 16.0 * 3.0 + 1.0 / 64.0 * 4.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[6] - 3.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[7] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[8] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[9] - 3.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[10] - (1.0 / 64.0 * 3.0 + 9.0 / 16.0 * 4.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[11] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[13] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[14] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[15] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[16] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[17] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[18] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[19] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘3
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -20.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesVelResultZ[0] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[1] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[2] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[4] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[5] - (9.0 / 16.0 * 5.0 + 1.0 / 64.0 * 6.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[6] - 5.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[7] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[8] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[9] - 5.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[10] - (1.0 / 64.0 * 5.0 + 9.0 / 16.0 * 6.0) * 64.0 / 37.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[11] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[13] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[14] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[15] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[16] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[17] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[18] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[19] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 30.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 60.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesVelResultX = FCVParticlesVelField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesVelResultX[0] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[1] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[2] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[4] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[5] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[6] - (4.0 / 9.0 * 1.0 + 1.0 / 36.0 * 2.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[7] - 1.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[8] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[9] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[10] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[11] - 1.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[12] - (1.0 / 36.0 * 1.0 + 4.0 / 9.0 * 2.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[13] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[14] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[15] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[16] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[17] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[18] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultX[19] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 40.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -26.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesVelResultX = FCVParticlesVelField.getConstGridDataX();
		thrust::host_vector<Real> TempParticlesVelResultY = FCVParticlesVelField.getConstGridDataY();
		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesVelResultY[0] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[1] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[2] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[4] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[5] - (4.0 / 9.0 * 3.0 + 1.0 / 36.0 * 4.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[6] - 3.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[7] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[8] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[9] - 3.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[10] - (1.0 / 36.0 * 3.0 + 4.0 / 9.0 * 4.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[11] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[13] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[14] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[15] - 4.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[16] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[17] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[18] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultY[19] - 0.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘3
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -20.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -10.0;
		vector<Real> ParticlesVelCPU(6);
		ParticlesVelCPU[0] = 1;
		ParticlesVelCPU[1] = 3;
		ParticlesVelCPU[2] = 5;
		ParticlesVelCPU[3] = 2;
		ParticlesVelCPU[4] = 4;
		ParticlesVelCPU[5] = 6;

		CEulerParticles EulerParticles(2, ParticlesPos.data(), ParticlesVelCPU.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing);
		CFaceCenteredVectorField FCVGridWeightField(Res, Origin, Spacing);
		EulerParticles.transferParticlesVel2Field(FCVParticlesVelField, FCVGridWeightField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesVelResultZ = FCVParticlesVelField.getConstGridDataZ();

		EXPECT_LT(abs(TempParticlesVelResultZ[0] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[1] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[2] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[3] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[4] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[5] - (4.0 / 9.0 * 5.0 + 1.0 / 36.0 * 6.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[6] - 5.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[7] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[8] - 5.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[9] - 5.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[10] - (1.0 / 36.0 * 5.0 + 4.0 / 9.0 * 6.0) * 36.0 / 17.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[11] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[12] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[13] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[14] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[15] - 6.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[16] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[17] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[18] - 0.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResultZ[19] - 0.0), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//transferCCSGrid2Particles≤‚ ‘
TEST(ParticlesAdvection, transferCCSGrid2Particles)
{
	CCudaContextManager::getInstance().initCudaContext();

	//Nearest KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfPerGrid = 8;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[21] = -1.0; SolidSDFScalarFieldData[21] = 1.0;
		FluidSDFScalarFieldData[22] = -1.0; SolidSDFScalarFieldData[22] = 1.0;
		FluidSDFScalarFieldData[25] = -1.0; SolidSDFScalarFieldData[25] = 1.0;
		FluidSDFScalarFieldData[26] = -1.0; SolidSDFScalarFieldData[26] = 1.0;
		FluidSDFScalarFieldData[37] = -1.0; SolidSDFScalarFieldData[37] = 1.0;
		FluidSDFScalarFieldData[38] = -1.0; SolidSDFScalarFieldData[38] = 1.0;
		FluidSDFScalarFieldData[41] = -1.0; SolidSDFScalarFieldData[41] = 1.0;
		FluidSDFScalarFieldData[42] = -1.0; SolidSDFScalarFieldData[42] = 1.0;

		vector<Vector3> FluidGridIndex
		{
			Vector3(1.0, 1.0, 1.0),
			Vector3(2.0, 1.0, 1.0),
			Vector3(1.0, 2.0, 1.0),
			Vector3(2.0, 2.0, 1.0),
			Vector3(1.0, 1.0, 2.0),
			Vector3(2.0, 1.0, 2.0),
			Vector3(1.0, 2.0, 2.0),
			Vector3(2.0, 2.0, 2.0)
		};

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 8 * NumOfPerGrid);

		vector<Real> ColorFieldData(Res.x * Res.y * Res.z);
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldData[i] = i;
		}
		CCellCenteredScalarField CCSColorField(Res, Origin, Spacing, ColorFieldData.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 8 * NumOfPerGrid);

		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField,
			EPGTransferAlgorithm::G2PNEAREST
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		for (int i = 0; i < 8; i++)
		{
			for (int k = 0; k < NumOfPerGrid; k++)
			{
				Real Value = FluidGridIndex[i].z * Res.x * Res.y + FluidGridIndex[i].y * Res.x + FluidGridIndex[i].x;
				EXPECT_LT(abs(ParticlesColorResult[i * NumOfPerGrid + k] - Value), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Linear KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;

		vector<Real> ColorFieldData(Res.x * Res.y * Res.z);
		ColorFieldData[0] = 64.0;
		ColorFieldData[1] = 64.0;
		ColorFieldData[2] = 64.0;
		ColorFieldData[3] = 0.0;
		ColorFieldData[4] = 64.0;
		ColorFieldData[5] = 64.0;
		ColorFieldData[6] = 96.0;
		ColorFieldData[7] = 128.0;
		ColorFieldData[8] = 64.0;
		ColorFieldData[9] = 96.0;
		ColorFieldData[10] = 128.0;
		ColorFieldData[11] = 128.0;
		ColorFieldData[12] = 0.0;
		ColorFieldData[13] = 128.0;
		ColorFieldData[14] = 128.0;
		ColorFieldData[15] = 128.0;
		CCellCenteredScalarField CCSColorField(Res, Origin, Spacing, ColorFieldData.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 128.0), GRID_SOLVER_EPSILON);
	}

	//Linear KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -5.0;

		vector<Real> ColorFieldData(Res.x * Res.y * Res.z);
		ColorFieldData[0] = 64.0;
		ColorFieldData[1] = 64.0;
		ColorFieldData[2] = 64.0;
		ColorFieldData[3] = 0.0;
		ColorFieldData[4] = 64.0;
		ColorFieldData[5] = 64.0;
		ColorFieldData[6] = 96.0;
		ColorFieldData[7] = 128.0;
		ColorFieldData[8] = 64.0;
		ColorFieldData[9] = 96.0;
		ColorFieldData[10] = 128.0;
		ColorFieldData[11] = 128.0;
		ColorFieldData[12] = 0.0;
		ColorFieldData[13] = 128.0;
		ColorFieldData[14] = 128.0;
		ColorFieldData[15] = 128.0;
		CCellCenteredScalarField CCSColorField(Res, Origin, Spacing, ColorFieldData.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 96.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 128.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘1
	{	
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;

		vector<Real> ColorFieldData(Res.x * Res.y * Res.z);
		ColorFieldData[0] = 64.0;
		ColorFieldData[1] = 64.0;
		ColorFieldData[2] = 64.0;
		ColorFieldData[3] = 0.0;
		ColorFieldData[4] = 64.0;
		ColorFieldData[5] = 64.0;
		ColorFieldData[6] = 96.0;
		ColorFieldData[7] = 128.0;
		ColorFieldData[8] = 64.0;
		ColorFieldData[9] = 96.0;
		ColorFieldData[10] = 128.0;
		ColorFieldData[11] = 128.0;
		ColorFieldData[12] = 0.0;
		ColorFieldData[13] = 128.0;
		ColorFieldData[14] = 128.0;
		ColorFieldData[15] = 128.0;
		CCellCenteredScalarField CCSColorField(Res, Origin, Spacing, ColorFieldData.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField,
			EPGTransferAlgorithm::QUADRATIC
		);
		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField,
			EPGTransferAlgorithm::QUADRATIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 71.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 121.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -15.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -5.0;

		vector<Real> ColorFieldData(Res.x * Res.y * Res.z);
		ColorFieldData[0] = 64.0;
		ColorFieldData[1] = 64.0;
		ColorFieldData[2] = 64.0;
		ColorFieldData[3] = 0.0;
		ColorFieldData[4] = 64.0;
		ColorFieldData[5] = 64.0;
		ColorFieldData[6] = 96.0;
		ColorFieldData[7] = 128.0;
		ColorFieldData[8] = 64.0;
		ColorFieldData[9] = 96.0;
		ColorFieldData[10] = 128.0;
		ColorFieldData[11] = 128.0;
		ColorFieldData[12] = 0.0;
		ColorFieldData[13] = 128.0;
		ColorFieldData[14] = 128.0;
		ColorFieldData[15] = 128.0;
		CCellCenteredScalarField CCSColorField(Res, Origin, Spacing, ColorFieldData.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField,
			EPGTransferAlgorithm::QUADRATIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 71.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 121.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;

		vector<Real> ColorFieldData(Res.x * Res.y * Res.z);
		ColorFieldData[0] = 36.0;
		ColorFieldData[1] = 36.0;
		ColorFieldData[2] = 36.0;
		ColorFieldData[3] = 0.0;
		ColorFieldData[4] = 36.0;
		ColorFieldData[5] = 36.0;
		ColorFieldData[6] = 54.0;
		ColorFieldData[7] = 72.0;
		ColorFieldData[8] = 36.0;
		ColorFieldData[9] = 54.0;
		ColorFieldData[10] = 72.0;
		ColorFieldData[11] = 72.0;
		ColorFieldData[12] = 0.0;
		ColorFieldData[13] = 72.0;
		ColorFieldData[14] = 72.0;
		ColorFieldData[15] = 72.0;
		CCellCenteredScalarField CCSColorField(Res, Origin, Spacing, ColorFieldData.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField,
			EPGTransferAlgorithm::CUBIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 41.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 67.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -15.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -5.0;

		vector<Real> ColorFieldData(Res.x * Res.y * Res.z);
		ColorFieldData[0] = 36.0;
		ColorFieldData[1] = 36.0;
		ColorFieldData[2] = 36.0;
		ColorFieldData[3] = 0.0;
		ColorFieldData[4] = 36.0;
		ColorFieldData[5] = 36.0;
		ColorFieldData[6] = 54.0;
		ColorFieldData[7] = 72.0;
		ColorFieldData[8] = 36.0;
		ColorFieldData[9] = 54.0;
		ColorFieldData[10] = 72.0;
		ColorFieldData[11] = 72.0;
		ColorFieldData[12] = 0.0;
		ColorFieldData[13] = 72.0;
		ColorFieldData[14] = 72.0;
		ColorFieldData[15] = 72.0;
		CCellCenteredScalarField CCSColorField(Res, Origin, Spacing, ColorFieldData.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCSField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCSColorField,
			EPGTransferAlgorithm::CUBIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 41.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 67.0), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//transferCCVGrid2Particles≤‚ ‘
TEST(ParticlesAdvection, transferCCVGrid2Particles)
{
	CCudaContextManager::getInstance().initCudaContext();

	//Nearest KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfPerGrid = 8;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[21] = -1.0; SolidSDFScalarFieldData[21] = 1.0;
		FluidSDFScalarFieldData[22] = -1.0; SolidSDFScalarFieldData[22] = 1.0;
		FluidSDFScalarFieldData[25] = -1.0; SolidSDFScalarFieldData[25] = 1.0;
		FluidSDFScalarFieldData[26] = -1.0; SolidSDFScalarFieldData[26] = 1.0;
		FluidSDFScalarFieldData[37] = -1.0; SolidSDFScalarFieldData[37] = 1.0;
		FluidSDFScalarFieldData[38] = -1.0; SolidSDFScalarFieldData[38] = 1.0;
		FluidSDFScalarFieldData[41] = -1.0; SolidSDFScalarFieldData[41] = 1.0;
		FluidSDFScalarFieldData[42] = -1.0; SolidSDFScalarFieldData[42] = 1.0;

		vector<Vector3> FluidGridIndex
		{
			Vector3(1.0, 1.0, 1.0),
			Vector3(2.0, 1.0, 1.0),
			Vector3(1.0, 2.0, 1.0),
			Vector3(2.0, 2.0, 1.0),
			Vector3(1.0, 1.0, 2.0),
			Vector3(2.0, 1.0, 2.0),
			Vector3(1.0, 2.0, 2.0),
			Vector3(2.0, 2.0, 2.0)
		};

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		EXPECT_EQ(EulerParticles.getNumOfParticles(), 8 * NumOfPerGrid);

		vector<Real> ColorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataZ(Res.x * Res.y * Res.z);
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldDataX[i] = i;
			ColorFieldDataY[i] = i + 1;
			ColorFieldDataZ[i] = i + 2;
		}
		CCellCenteredVectorField CCVColorField(Res, Origin, Spacing, ColorFieldDataX.data(), ColorFieldDataY.data(), ColorFieldDataZ.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 3 * 8 * NumOfPerGrid);

		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField,
			EPGTransferAlgorithm::G2PNEAREST
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		for (int i = 0; i < 8; i++)
		{
			for (int k = 0; k < NumOfPerGrid; k++)
			{
				Real Value = FluidGridIndex[i].z * Res.x * Res.y + FluidGridIndex[i].y * Res.x + FluidGridIndex[i].x;
				EXPECT_LT(abs(ParticlesColorResult[3 * (i * NumOfPerGrid + k)] - Value), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(ParticlesColorResult[3 * (i * NumOfPerGrid + k) + 1] - Value - 1), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(ParticlesColorResult[3 * (i * NumOfPerGrid + k) + 2] - Value - 2), GRID_SOLVER_EPSILON);
			}
		}
	}

	//Linear KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;

		vector<Real> ColorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataZ(Res.x * Res.y * Res.z);
		ColorFieldDataX[0] = 64.0;
		ColorFieldDataX[1] = 64.0;
		ColorFieldDataX[2] = 64.0;
		ColorFieldDataX[3] = 0.0;
		ColorFieldDataX[4] = 64.0;
		ColorFieldDataX[5] = 64.0;
		ColorFieldDataX[6] = 96.0;
		ColorFieldDataX[7] = 128.0;
		ColorFieldDataX[8] = 64.0;
		ColorFieldDataX[9] = 96.0;
		ColorFieldDataX[10] = 128.0;
		ColorFieldDataX[11] = 128.0;
		ColorFieldDataX[12] = 0.0;
		ColorFieldDataX[13] = 128.0;
		ColorFieldDataX[14] = 128.0;
		ColorFieldDataX[15] = 128.0;
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldDataY[i] = 2 * ColorFieldDataX[i];
			ColorFieldDataZ[i] = 3 * ColorFieldDataX[i];
		}
		CCellCenteredVectorField CCVColorField(Res, Origin, Spacing, ColorFieldDataX.data(), ColorFieldDataY.data(), ColorFieldDataZ.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 3 * 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 64.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[2] - 192.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[3] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[4] - 256.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[5] - 384.0), GRID_SOLVER_EPSILON);
	}

	//Linear KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 60.0;
		ParticlesPos[2] = -10.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -5.0;

		vector<Real> ColorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataZ(Res.x * Res.y * Res.z);
		ColorFieldDataX[0] = 64.0;
		ColorFieldDataX[1] = 64.0;
		ColorFieldDataX[2] = 64.0;
		ColorFieldDataX[3] = 0.0;
		ColorFieldDataX[4] = 64.0;
		ColorFieldDataX[5] = 64.0;
		ColorFieldDataX[6] = 96.0;
		ColorFieldDataX[7] = 128.0;
		ColorFieldDataX[8] = 64.0;
		ColorFieldDataX[9] = 96.0;
		ColorFieldDataX[10] = 128.0;
		ColorFieldDataX[11] = 128.0;
		ColorFieldDataX[12] = 0.0;
		ColorFieldDataX[13] = 128.0;
		ColorFieldDataX[14] = 128.0;
		ColorFieldDataX[15] = 128.0;
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldDataY[i] = 2 * ColorFieldDataX[i];
			ColorFieldDataZ[i] = 3 * ColorFieldDataX[i];
		}
		CCellCenteredVectorField CCVColorField(Res, Origin, Spacing, ColorFieldDataX.data(), ColorFieldDataY.data(), ColorFieldDataZ.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 3 * 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 96.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 192.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[2] - 288.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[3] - 128.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[4] - 256.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[5] - 384.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;

		vector<Real> ColorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataZ(Res.x * Res.y * Res.z);
		ColorFieldDataX[0] = 64.0;
		ColorFieldDataX[1] = 64.0;
		ColorFieldDataX[2] = 64.0;
		ColorFieldDataX[3] = 0.0;
		ColorFieldDataX[4] = 64.0;
		ColorFieldDataX[5] = 64.0;
		ColorFieldDataX[6] = 96.0;
		ColorFieldDataX[7] = 128.0;
		ColorFieldDataX[8] = 64.0;
		ColorFieldDataX[9] = 96.0;
		ColorFieldDataX[10] = 128.0;
		ColorFieldDataX[11] = 128.0;
		ColorFieldDataX[12] = 0.0;
		ColorFieldDataX[13] = 128.0;
		ColorFieldDataX[14] = 128.0;
		ColorFieldDataX[15] = 128.0;
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldDataY[i] = 2 * ColorFieldDataX[i];
			ColorFieldDataZ[i] = 3 * ColorFieldDataX[i];
		}
		CCellCenteredVectorField CCVColorField(Res, Origin, Spacing, ColorFieldDataX.data(), ColorFieldDataY.data(), ColorFieldDataZ.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 3 * 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField,
			EPGTransferAlgorithm::QUADRATIC
		);
		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField,
			EPGTransferAlgorithm::QUADRATIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 71.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 142.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[2] - 213.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[3] - 121.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[4] - 242.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[5] - 363.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -15.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -5.0;

		vector<Real> ColorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataZ(Res.x * Res.y * Res.z);
		ColorFieldDataX[0] = 64.0;
		ColorFieldDataX[1] = 64.0;
		ColorFieldDataX[2] = 64.0;
		ColorFieldDataX[3] = 0.0;
		ColorFieldDataX[4] = 64.0;
		ColorFieldDataX[5] = 64.0;
		ColorFieldDataX[6] = 96.0;
		ColorFieldDataX[7] = 128.0;
		ColorFieldDataX[8] = 64.0;
		ColorFieldDataX[9] = 96.0;
		ColorFieldDataX[10] = 128.0;
		ColorFieldDataX[11] = 128.0;
		ColorFieldDataX[12] = 0.0;
		ColorFieldDataX[13] = 128.0;
		ColorFieldDataX[14] = 128.0;
		ColorFieldDataX[15] = 128.0;
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldDataY[i] = 2 * ColorFieldDataX[i];
			ColorFieldDataZ[i] = 3 * ColorFieldDataX[i];
		}
		CCellCenteredVectorField CCVColorField(Res, Origin, Spacing, ColorFieldDataX.data(), ColorFieldDataY.data(), ColorFieldDataZ.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 3 * 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField,
			EPGTransferAlgorithm::QUADRATIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 71.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 142.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[2] - 213.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[3] - 121.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[4] - 242.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[5] - 363.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;

		vector<Real> ColorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataZ(Res.x * Res.y * Res.z);
		ColorFieldDataX[0] = 36.0;
		ColorFieldDataX[1] = 36.0;
		ColorFieldDataX[2] = 36.0;
		ColorFieldDataX[3] = 0.0;
		ColorFieldDataX[4] = 36.0;
		ColorFieldDataX[5] = 36.0;
		ColorFieldDataX[6] = 54.0;
		ColorFieldDataX[7] = 72.0;
		ColorFieldDataX[8] = 36.0;
		ColorFieldDataX[9] = 54.0;
		ColorFieldDataX[10] = 72.0;
		ColorFieldDataX[11] = 72.0;
		ColorFieldDataX[12] = 0.0;
		ColorFieldDataX[13] = 72.0;
		ColorFieldDataX[14] = 72.0;
		ColorFieldDataX[15] = 72.0;
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldDataY[i] = 2 * ColorFieldDataX[i];
			ColorFieldDataZ[i] = 3 * ColorFieldDataX[i];
		}
		CCellCenteredVectorField CCVColorField(Res, Origin, Spacing, ColorFieldDataX.data(), ColorFieldDataY.data(), ColorFieldDataZ.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 3 * 2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField,
			EPGTransferAlgorithm::CUBIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 41.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 82.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[2] - 123.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[3] - 67.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[4] - 134.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[5] - 201.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -15.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -5.0;

		vector<Real> ColorFieldDataX(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataY(Res.x * Res.y * Res.z);
		vector<Real> ColorFieldDataZ(Res.x * Res.y * Res.z);
		ColorFieldDataX[0] = 36.0;
		ColorFieldDataX[1] = 36.0;
		ColorFieldDataX[2] = 36.0;
		ColorFieldDataX[3] = 0.0;
		ColorFieldDataX[4] = 36.0;
		ColorFieldDataX[5] = 36.0;
		ColorFieldDataX[6] = 54.0;
		ColorFieldDataX[7] = 72.0;
		ColorFieldDataX[8] = 36.0;
		ColorFieldDataX[9] = 54.0;
		ColorFieldDataX[10] = 72.0;
		ColorFieldDataX[11] = 72.0;
		ColorFieldDataX[12] = 0.0;
		ColorFieldDataX[13] = 72.0;
		ColorFieldDataX[14] = 72.0;
		ColorFieldDataX[15] = 72.0;
		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			ColorFieldDataY[i] = 2 * ColorFieldDataX[i];
			ColorFieldDataZ[i] = 3 * ColorFieldDataX[i];
		}
		CCellCenteredVectorField CCVColorField(Res, Origin, Spacing, ColorFieldDataX.data(), ColorFieldDataY.data(), ColorFieldDataZ.data());

		thrust::device_vector<Real> TempParticlesColorResult;
		resizeDeviceVector(TempParticlesColorResult, 3 *2);

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		thrust::host_vector<Real> TempParticlesPosResult = EulerParticles.getParticlesPos();
		vector<Real> ParticlesPosResult(TempParticlesPosResult.begin(), TempParticlesPosResult.end());

		tranferCCVField2ParticlesInvoker
		(
			EulerParticles.getParticlesPos(),
			TempParticlesColorResult,
			CCVColorField,
			EPGTransferAlgorithm::CUBIC
		);

		thrust::host_vector<Real> ParticlesColorResult = TempParticlesColorResult;

		EXPECT_LT(abs(ParticlesColorResult[0] - 41.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[1] - 82.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[2] - 123.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[3] - 67.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[4] - 134.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(ParticlesColorResult[5] - 201.0), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//transferFCVGrid2Particles≤‚ ‘
TEST(ParticlesAdvection, transferFCVGrid2Particles)
{
	CCudaContextManager::getInstance().initCudaContext();

	//Nearest KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3i ParticlesRes = Res;
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfparticles = 16;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -16.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < 1; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 4; x++)
				{
					ParticlesPos[3 * (z * 4 * 4 + y * 4 + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * 4 * 4 + y * 4 + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * 4 * 4 + y * 4 + x) + 2] = BasePos.z;
				}
			}
		}

		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z;
				}
			}
		}
		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::G2PNEAREST);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					UInt CurIndex = z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x;
					if (CurIndex == 0 || CurIndex == 4 || CurIndex == 8 || CurIndex == 12)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 3 || CurIndex == 7 || CurIndex == 11 || CurIndex == 15)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex] - 3.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex] - 2.0), GRID_SOLVER_EPSILON);
					}

					if (CurIndex == 0 || CurIndex == 1 || CurIndex == 2 || CurIndex == 3)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 1] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 12 || CurIndex == 13 || CurIndex == 14 || CurIndex == 15)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 1] - 3.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 1] - 2.0), GRID_SOLVER_EPSILON);
					}

					EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 2] - 0.0), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	{}

	//Nearest KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3i ParticlesRes = Res;
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		Vector3 BasePos = Vector3(-4.0, (Origin.y + 1.25 * Spacing.y), (Origin.z + 1.25 * Spacing.z));
		vector<Real> ParticlesPos(3 * ParticlesRes.x * ParticlesRes.y * ParticlesRes.z);

		for (int z = 0; z < 4; z++)
		{
			for (int y = 0; y < 4; y++)
			{
				for (int x = 0; x < 1; x++)
				{
					ParticlesPos[3 * (z * 1 * 4 + y * 1 + x)] = BasePos.x;
					ParticlesPos[3 * (z * 1 * 4 + y * 1 + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * 1 * 4 + y * 1 + x) + 2] = BasePos.z + z * 0.5 * Spacing.z;
				}
			}
		}

		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z;
				}
			}
		}
		CEulerParticles EulerParticles(ParticlesRes.x * ParticlesRes.y * ParticlesRes.z, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::G2PNEAREST);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					UInt CurIndex = z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x;

					EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex] - 1.0), GRID_SOLVER_EPSILON);

					if (CurIndex == 0 || CurIndex == 4 || CurIndex == 8 || CurIndex == 12)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 1] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 3 || CurIndex == 7 || CurIndex == 11 || CurIndex == 15)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 1] - 3.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 1] - 2.0), GRID_SOLVER_EPSILON);
					}

					if (CurIndex == 0 || CurIndex == 1 || CurIndex == 2 || CurIndex == 3)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 2] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 12 || CurIndex == 13 || CurIndex == 14 || CurIndex == 15)
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 2] - 3.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(TempParticlesVelResult[3 * CurIndex + 2] - 2.0), GRID_SOLVER_EPSILON);
					}
				}
			}
}
	}

	{}

	//Linear KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 60.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0] - 1.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1] - 2.0), GRID_SOLVER_EPSILON);
	}

	//Linear KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -26.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0 + 1] - 2.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1 + 1] - 3.0), GRID_SOLVER_EPSILON);
	}

	//Linear KernelFunction≤‚ ‘3
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -15.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -10.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::LINEAR);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0 + 2] - 3.5), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1 + 2] - 4.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 30.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 60.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::QUADRATIC);
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1] - 2.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 40.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -26.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0 + 1] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1 + 1] - 3.0), GRID_SOLVER_EPSILON);
	}

	//Quadratic KernelFunction≤‚ ‘3
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -20.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -10.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::QUADRATIC);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0 + 2] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1 + 2] - 4.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘1
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 30.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 60.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -26.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0] - 1.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1] - 2.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘2
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 45.0;
		ParticlesPos[1] = 40.0;
		ParticlesPos[2] = -24.0;
		ParticlesPos[3] = 75.0;
		ParticlesPos[4] = 60.0;
		ParticlesPos[5] = -26.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0 + 1] - 2.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1 + 1] - 3.0), GRID_SOLVER_EPSILON);
	}

	//Cubic KernelFunction≤‚ ‘3
	{
		Vector3i Res = Vector3i(1, 4, 4);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(0.0, 20.0, -30.0);
		Vector3 Spacing = Vector3(30.0, 20.0, 10.0);

		vector<Real> ParticlesPos(6);
		ParticlesPos[0] = 15.0;
		ParticlesPos[1] = 50.0;
		ParticlesPos[2] = -20.0;
		ParticlesPos[3] = 15.0;
		ParticlesPos[4] = 70.0;
		ParticlesPos[5] = -10.0;
		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);
		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = y + 1;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = z + 2;
				}
			}
		}

		CEulerParticles EulerParticles(2, ParticlesPos.data());

		CFaceCenteredVectorField FCVParticlesVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());
		EulerParticles.transferVelField2Particles(FCVParticlesVelField, EPGTransferAlgorithm::CUBIC);

		thrust::host_vector<Real> TempParticlesVelResult = EulerParticles.getParticlesVel();

		EXPECT_LT(abs(TempParticlesVelResult[3 * 0 + 2] - 3.0), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(TempParticlesVelResult[3 * 1 + 2] - 4.0), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//advectParticlesInVelField≤‚ ‘
TEST(ParticlesAdvection, advectParticlesInVelField)
{
	CCudaContextManager::getInstance().initCudaContext();

	//RK1_NoSubSteps
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		Vector3i ParticlesRes = Res;
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}

		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = 10 * x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = 20 * y;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = 30 * z;
				}
			}
		}

		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());
		CFaceCenteredVectorField FCVVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());

		Real DeltaT = 0.1;
		Real CFLNumber = 1.0;
		EulerParticles.advectParticlesInVelField(FCVVelField, DeltaT, CFLNumber, CCellCenteredScalarField(), EAdvectionAccuracy::RK1, ESamplingAlgorithm::TRILINEAR);

		thrust::host_vector<Real> ParticlesPosResult = EulerParticles.getParticlesPos();
		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					Vector3 RelOriPos = Vector3(
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2]) - Origin;
					RelOriPos /= Spacing;
					
					Real ResultPosX = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] + DeltaT * 10 * RelOriPos.x;
					Real ResultPosY = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] + DeltaT * 20 * RelOriPos.y;
					Real ResultPosZ = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] + DeltaT * 30 * RelOriPos.z;
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] - ResultPosX), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] - ResultPosY), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] - ResultPosZ), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//RK1_3SubSteps
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		Vector3i ParticlesRes = Res;
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}

		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = 10 * x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = 20 * y;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = 30 * z;
				}
			}
		}

		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());
		CFaceCenteredVectorField FCVVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());

		Real DeltaT = 0.4;
		Real CFLNumber = 1.1;
		EulerParticles.advectParticlesInVelField(FCVVelField, DeltaT, CFLNumber, CCellCenteredScalarField(), EAdvectionAccuracy::RK1, ESamplingAlgorithm::TRILINEAR);

		thrust::host_vector<Real> ParticlesPosResult = EulerParticles.getParticlesPos();
		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					Vector3 RelOriPos = Vector3(
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2]) - Origin;
					Vector3 SecondRelPos = RelOriPos;
					Vector3 ThirdRelPos = RelOriPos;
					RelOriPos /= Spacing;
					SecondRelPos += 0.2 * Vector3(10, 20, 30) * RelOriPos;
					SecondRelPos /= Spacing;
					ThirdRelPos += 0.2 * Vector3(10, 20, 30) * RelOriPos + 0.1 * Vector3(10, 20, 30) * SecondRelPos;
					ThirdRelPos /= Spacing;

					Real ResultPosX = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] + 0.2 * 10 * RelOriPos.x + 0.1 * 10 * SecondRelPos.x + 0.1 * 10 * ThirdRelPos.x;
					Real ResultPosY = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] + 0.2 * 20 * RelOriPos.y + 0.1 * 20 * SecondRelPos.y + 0.1 * 20 * ThirdRelPos.y;
					Real ResultPosZ = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] + 0.2 * 30 * RelOriPos.z + 0.1 * 30 * SecondRelPos.z + 0.1 * 30 * ThirdRelPos.z;
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] - ResultPosX), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] - ResultPosY), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] - ResultPosZ), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//RK2_NoSubSteps
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		Vector3i ParticlesRes = Res;
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}

		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = 10 * x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = 20 * y;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = 30 * z;
				}
			}
		}

		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());
		CFaceCenteredVectorField FCVVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());

		Real DeltaT = 0.1;
		Real CFLNumber = 1.0;
		EulerParticles.advectParticlesInVelField(FCVVelField, DeltaT, CFLNumber, CCellCenteredScalarField(), EAdvectionAccuracy::RK2, ESamplingAlgorithm::TRILINEAR);

		thrust::host_vector<Real> ParticlesPosResult = EulerParticles.getParticlesPos();
		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					Vector3 RelOriPos = Vector3(
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2]) - Origin;
					Vector3 RelMidPos = RelOriPos;
					RelOriPos /= Spacing;

					RelMidPos += 0.5 * DeltaT * Vector3(10, 20, 30) * RelOriPos;
					RelMidPos /= Spacing;

					Real ResultPosX = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] + DeltaT * 10 * RelMidPos.x;
					Real ResultPosY = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] + DeltaT * 20 * RelMidPos.y;
					Real ResultPosZ = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] + DeltaT * 30 * RelMidPos.z;
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] - ResultPosX), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] - ResultPosY), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] - ResultPosZ), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//RK3_NoSubSteps
	{
		Vector3i Res = Vector3i(4, 4, 1);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		Vector3i ParticlesRes = Res;
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}

		vector<Real> VelFieldDataCPUX(ResX.x * ResX.y * ResX.z);
		vector<Real> VelFieldDataCPUY(ResY.x * ResY.y * ResY.z);
		vector<Real> VelFieldDataCPUZ(ResZ.x * ResZ.y * ResZ.z);

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					VelFieldDataCPUX[z * ResX.x * ResX.y + y * ResX.x + x] = 10 * x;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					VelFieldDataCPUY[z * ResY.x * ResY.y + y * ResY.x + x] = 20 * y;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					VelFieldDataCPUZ[z * ResZ.x * ResZ.y + y * ResZ.x + x] = 30 * z;
				}
			}
		}

		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());
		CFaceCenteredVectorField FCVVelField(Res, Origin, Spacing, VelFieldDataCPUX.data(), VelFieldDataCPUY.data(), VelFieldDataCPUZ.data());

		Real DeltaT = 0.1;
		Real CFLNumber = 1.0;
		EulerParticles.advectParticlesInVelField(FCVVelField, DeltaT, CFLNumber, CCellCenteredScalarField(), EAdvectionAccuracy::RK3, ESamplingAlgorithm::TRILINEAR);

		thrust::host_vector<Real> ParticlesPosResult = EulerParticles.getParticlesPos();
		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					Vector3 RelOriPos = Vector3(
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1],
						ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2]) - Origin;
					Vector3 RelMidPos = RelOriPos;
					Vector3 RelThreeFourthsPos = RelOriPos;
					RelOriPos /= Spacing;

					RelMidPos += 0.5 * DeltaT * Vector3(10, 20, 30) * RelOriPos;
					RelMidPos /= Spacing;

					RelThreeFourthsPos += 0.75 * DeltaT * Vector3(10, 20, 30) * RelMidPos;
					RelThreeFourthsPos /= Spacing;

					Real ResultPosX = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] + 2.0 / 9.0 * DeltaT * 10 * RelOriPos.x + 3.0 / 9.0 * DeltaT * 10 * RelMidPos.x + 4.0 / 9.0 * DeltaT * 10 * RelThreeFourthsPos.x;
					Real ResultPosY = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] + 2.0 / 9.0 * DeltaT * 20 * RelOriPos.y + 3.0 / 9.0 * DeltaT * 20 * RelMidPos.y + 4.0 / 9.0 * DeltaT * 20 * RelThreeFourthsPos.y;
					Real ResultPosZ = ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] + 2.0 / 9.0 * DeltaT * 30 * RelOriPos.z + 3.0 / 9.0 * DeltaT * 30 * RelMidPos.z + 4.0 / 9.0 * DeltaT * 30 * RelThreeFourthsPos.z;
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] - ResultPosX), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] - ResultPosY), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(ParticlesPosResult[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] - ResultPosZ), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//buildFluidMarkers≤‚ ‘
TEST(ParticlesAdvection, buildFluidMarkers)
{
	CCudaContextManager::getInstance().initCudaContext();

	//SUM
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		SolidSDFScalarFieldData[5] = 1.0;
		SolidSDFScalarFieldData[6] = 1.0;
		SolidSDFScalarFieldData[9] = 1.0;
		SolidSDFScalarFieldData[10] = 1.0;
		SolidSDFScalarFieldData[13] = 1.0;
		SolidSDFScalarFieldData[14] = 1.0;

		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSMarkersField(Res, Origin, Spacing);

		Vector3i ParticlesRes = Vector3i(4, 4, 1);
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}
		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());

		buildFluidMarkersInvoker
		(
			CCSSolidSDFField,
			EulerParticles.getParticlesPos(),
			CCSMarkersField
		);

		thrust::host_vector<Real> MarkersResult = CCSMarkersField.getConstGridData();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurIndex = z * Res.x * Res.y + y * Res.x + x;

					if (CurIndex == 5 || CurIndex == 6 || CurIndex == 9 || CurIndex == 10)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 13 || CurIndex == 14)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 2.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	/*//Linear
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		for (int i = 0; i < 16; i++)
		{
			SolidSDFScalarFieldData[i] = 1.0;
		}

		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSMarkersField(Res, Origin, Spacing);

		Vector3i ParticlesRes = Vector3i(3, 4, 1);
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}
		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());

		buildFluidMarkersInvoker
		(
			CCSSolidSDFField,
			EulerParticles.getParticlesPos(),
			CCSMarkersField,
			EPGTransferAlgorithm::LINEAR
		);

		thrust::host_vector<Real> MarkersResult = CCSMarkersField.getConstGridData();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurIndex = z * Res.x * Res.y + y * Res.x + x;

					if (CurIndex == 0 || CurIndex == 1 || CurIndex == 2 || 
						CurIndex == 4 || CurIndex == 5 || CurIndex == 6 ||
						CurIndex == 8 || CurIndex == 9 || CurIndex == 10 || 
						CurIndex == 12 || CurIndex == 13 || CurIndex == 14)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex == 3 || CurIndex == 7 || CurIndex == 11 || CurIndex == 15)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 2.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	//Quadratic
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		for (int i = 0; i < 16; i++)
		{
			SolidSDFScalarFieldData[i] = 1.0;
		}

		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSMarkersField(Res, Origin, Spacing);

		Vector3i ParticlesRes = Vector3i(2, 1, 1);
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}
		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());

		buildFluidMarkersInvoker
		(
			CCSSolidSDFField,
			EulerParticles.getParticlesPos(),
			CCSMarkersField,
			EPGTransferAlgorithm::QUADRATIC
		);

		thrust::host_vector<Real> MarkersResult = CCSMarkersField.getConstGridData();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurIndex = z * Res.x * Res.y + y * Res.x + x;

					if (CurIndex < 12)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex < 16)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 2.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	//Cubic
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);

		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		for (int i = 0; i < 16; i++)
		{
			SolidSDFScalarFieldData[i] = 1.0;
		}

		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSMarkersField(Res, Origin, Spacing);

		Vector3i ParticlesRes = Vector3i(2, 1, 1);
		UInt NumOfparticles = ParticlesRes.x * ParticlesRes.y * ParticlesRes.z;

		Vector3 BasePos = Vector3((Origin.x + 1.25 * Spacing.x), (Origin.y + 1.25 * Spacing.y), -15.0);
		vector<Real> ParticlesPos(3 * NumOfparticles);

		for (int z = 0; z < ParticlesRes.z; z++)
		{
			for (int y = 0; y < ParticlesRes.y; y++)
			{
				for (int x = 0; x < ParticlesRes.x; x++)
				{
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x)] = BasePos.x + x * 0.5 * Spacing.x;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 1] = BasePos.y + y * 0.5 * Spacing.y;
					ParticlesPos[3 * (z * ParticlesRes.x * ParticlesRes.y + y * ParticlesRes.x + x) + 2] = BasePos.z;
				}
			}
		}
		CEulerParticles EulerParticles(NumOfparticles, ParticlesPos.data());

		buildFluidMarkersInvoker
		(
			CCSSolidSDFField,
			EulerParticles.getParticlesPos(),
			CCSMarkersField,
			EPGTransferAlgorithm::CUBIC
		);

		thrust::host_vector<Real> MarkersResult = CCSMarkersField.getConstGridData();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurIndex = z * Res.x * Res.y + y * Res.x + x;

					if (CurIndex < 12)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 1.0), GRID_SOLVER_EPSILON);
					}
					else if (CurIndex < 16)
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(MarkersResult[CurIndex] - 2.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}*/

	CCudaContextManager::getInstance().freeCudaContext();
}

//statisticalFluidDensity≤‚ ‘
TEST(ParticlesAdvection, statisticalFluidDensity)
{
	CCudaContextManager::getInstance().initCudaContext();

	//4°¡4°¡4Õ¯∏Ò£¨÷–º‰∞À∏ˆÕ¯∏Ò «ÀÆ£¨√ø∏ˆÕ¯∏Ò∞À∏ˆ¡£◊”
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		UInt NumOfPerGrid = 8;
		vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z, 1.0);
		vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z, -1.0);

		FluidSDFScalarFieldData[21] = -1.0; SolidSDFScalarFieldData[21] = 1.0;
		FluidSDFScalarFieldData[22] = -1.0; SolidSDFScalarFieldData[22] = 1.0;
		FluidSDFScalarFieldData[25] = -1.0; SolidSDFScalarFieldData[25] = 1.0;
		FluidSDFScalarFieldData[26] = -1.0; SolidSDFScalarFieldData[26] = 1.0;
		FluidSDFScalarFieldData[37] = -1.0; SolidSDFScalarFieldData[37] = 1.0;
		FluidSDFScalarFieldData[38] = -1.0; SolidSDFScalarFieldData[38] = 1.0;
		FluidSDFScalarFieldData[41] = -1.0; SolidSDFScalarFieldData[41] = 1.0;
		FluidSDFScalarFieldData[42] = -1.0; SolidSDFScalarFieldData[42] = 1.0;

		vector<Vector3> FluidGridIndex
		{
			Vector3(1.0, 1.0, 1.0),
			Vector3(2.0, 1.0, 1.0),
			Vector3(1.0, 2.0, 1.0),
			Vector3(2.0, 2.0, 1.0),
			Vector3(1.0, 1.0, 2.0),
			Vector3(2.0, 1.0, 2.0),
			Vector3(1.0, 2.0, 2.0),
			Vector3(2.0, 2.0, 2.0)
		};

		CCellCenteredScalarField CCSFluidSDFField(Res, Origin, Spacing, FluidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSSolidSDFField(Res, Origin, Spacing, SolidSDFScalarFieldData.data());

		CEulerParticles EulerParticles;
		EulerParticles.generateParticlesInFluid(CCSFluidSDFField, CCSSolidSDFField, NumOfPerGrid);

		CCellCenteredScalarField CCSFluidDensityField(Res, Origin, Spacing);
		EulerParticles.statisticalFluidDensity(CCSFluidDensityField);
		thrust::host_vector<Real> FluidDensityResult = CCSFluidDensityField.getConstGridData();

		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			if (i == 21 || i == 22 || i == 25 || i == 26 || i == 37 || i == 38 || i == 41 || i == 42)
			{
				EXPECT_LT(abs(FluidDensityResult[i] - 8.0), GRID_SOLVER_EPSILON);
			}
			else
			{
				EXPECT_LT(abs(FluidDensityResult[i] - 0.0), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
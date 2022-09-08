#include "pch.h"
#include "HybridSimulatorKernel.cuh"
#include "GridFluidSolver.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>

//根据密度混合两个场测试
TEST(HybridHelper, MixFieldWithDensity)
{
	CCudaContextManager::getInstance().initCudaContext();

	//混合CCS标量场
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		vector<Real> SrcScalarFieldAData(Res.x * Res.y * Res.z);
		vector<Real> SrcScalarFieldBData(Res.x * Res.y * Res.z);
		vector<Real> WeightFieldAData(Res.x * Res.y * Res.z);
		vector<Real> WeightFieldBData(Res.x * Res.y * Res.z, 0);

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurLinearIndex = z * Res.x * Res.y + y * Res.x + x;

					SrcScalarFieldAData[CurLinearIndex] = CurLinearIndex;
					SrcScalarFieldBData[CurLinearIndex] = CurLinearIndex + 2;
					WeightFieldAData[CurLinearIndex] = y;
					if (y == 1 || y == 2)
						WeightFieldBData[CurLinearIndex] = y + 2;
				}
			}
		}

		CCellCenteredScalarField CCSSrcScalarFieldA(Res, Origin, Spacing, SrcScalarFieldAData.data());
		CCellCenteredScalarField CCSSrcScalarFieldB(Res, Origin, Spacing, SrcScalarFieldBData.data());
		CCellCenteredScalarField CCSWeightFieldA(Res, Origin, Spacing, WeightFieldAData.data());
		CCellCenteredScalarField CCSWeightFieldB(Res, Origin, Spacing, WeightFieldBData.data());

		mixFieldWithDensityInvoker
		(
			CCSSrcScalarFieldA,
			CCSSrcScalarFieldB,
			CCSWeightFieldA,
			CCSWeightFieldB
		);

		thrust::host_vector<Real> MixResult = CCSSrcScalarFieldB.getConstGridData();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurLinearIndex = z * Res.x * Res.y + y * Res.x + x;

					if (y == 0)
					{
						
					}
					else if (y == 3)
					{
						EXPECT_LT(abs(MixResult[CurLinearIndex] - CurLinearIndex), GRID_SOLVER_EPSILON);
					}
					else
					{
						Real TempMixResult = (y * CurLinearIndex + (y + 2.0) * (CurLinearIndex + 2.0)) / (y + y + 2.0);
						EXPECT_LT(abs(MixResult[CurLinearIndex] - TempMixResult), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	//混合CCV向量场
	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		vector<Real> SrcVectorFieldADataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldADataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldADataZ(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldBDataX(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldBDataY(Res.x * Res.y * Res.z);
		vector<Real> SrcVectorFieldBDataZ(Res.x * Res.y * Res.z);
		vector<Real> WeightFieldAData(Res.x * Res.y * Res.z);
		vector<Real> WeightFieldBData(Res.x * Res.y * Res.z, 0);

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurLinearIndex = z * Res.x * Res.y + y * Res.x + x;

					SrcVectorFieldADataX[CurLinearIndex] = CurLinearIndex;
					SrcVectorFieldADataY[CurLinearIndex] = CurLinearIndex + 1;
					SrcVectorFieldADataZ[CurLinearIndex] = CurLinearIndex + 2;
					SrcVectorFieldBDataX[CurLinearIndex] = CurLinearIndex + 3;
					SrcVectorFieldBDataY[CurLinearIndex] = CurLinearIndex + 4;
					SrcVectorFieldBDataZ[CurLinearIndex] = CurLinearIndex + 5;

					WeightFieldAData[CurLinearIndex] = y;
					if (y == 1 || y == 2)
						WeightFieldBData[CurLinearIndex] = y + 2;
				}
			}
		}

		CCellCenteredVectorField CCVSrcVectorFieldA(Res, Origin, Spacing, SrcVectorFieldADataX.data(), SrcVectorFieldADataY.data(), SrcVectorFieldADataZ.data());
		CCellCenteredVectorField CCVSrcVectorFieldB(Res, Origin, Spacing, SrcVectorFieldBDataX.data(), SrcVectorFieldBDataY.data(), SrcVectorFieldBDataZ.data());
		CCellCenteredScalarField CCSWeightFieldA(Res, Origin, Spacing, WeightFieldAData.data());
		CCellCenteredScalarField CCSWeightFieldB(Res, Origin, Spacing, WeightFieldBData.data());

		mixFieldWithDensityInvoker
		(
			CCVSrcVectorFieldA,
			CCVSrcVectorFieldB,
			CCSWeightFieldA,
			CCSWeightFieldB
		);

		thrust::host_vector<Real> MixResultX = CCVSrcVectorFieldB.getConstGridDataX();
		thrust::host_vector<Real> MixResultY = CCVSrcVectorFieldB.getConstGridDataY();
		thrust::host_vector<Real> MixResultZ = CCVSrcVectorFieldB.getConstGridDataZ();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurLinearIndex = z * Res.x * Res.y + y * Res.x + x;

					if (y == 0)
					{

					}
					else if (y == 3)
					{
						EXPECT_LT(abs(MixResultX[CurLinearIndex] - CurLinearIndex), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(MixResultY[CurLinearIndex] - CurLinearIndex - 1.0), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(MixResultZ[CurLinearIndex] - CurLinearIndex - 2.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						Real TempMixResultX = (y * CurLinearIndex + (y + 2.0) * (CurLinearIndex + 3.0)) / (y + y + 2.0);
						Real TempMixResultY = (y * (CurLinearIndex + 1.0) + (y + 2.0) * (CurLinearIndex + 4.0)) / (y + y + 2.0);
						Real TempMixResultZ = (y * (CurLinearIndex + 2.0) + (y + 2.0) * (CurLinearIndex + 5.0)) / (y + y + 2.0);
						EXPECT_LT(abs(MixResultX[CurLinearIndex] - TempMixResultX), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(MixResultY[CurLinearIndex] - TempMixResultY), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(MixResultZ[CurLinearIndex] - TempMixResultZ), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	//混合FCV向量场
	{
		Vector3i Res = Vector3i(2, 2, 2);
		Vector3i ResX = Res + Vector3i(1, 0, 0);
		Vector3i ResY = Res + Vector3i(0, 1, 0);
		Vector3i ResZ = Res + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		vector<Real> SrcVectorFieldADataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldADataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldADataZ(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> SrcVectorFieldBDataX(ResX.x * ResX.y * ResX.z);
		vector<Real> SrcVectorFieldBDataY(ResY.x * ResY.y * ResY.z);
		vector<Real> SrcVectorFieldBDataZ(ResZ.x * ResZ.y * ResZ.z);
		vector<Real> WeightFieldAData(Res.x * Res.y * Res.z, 0);
		vector<Real> WeightFieldBData(Res.x * Res.y * Res.z, 0);

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					Int CurLinearIndex = z * ResX.x * ResX.y + y * ResX.x + x;

					SrcVectorFieldADataX[CurLinearIndex] = x;
					SrcVectorFieldBDataX[CurLinearIndex] = x + 1;
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					Int CurLinearIndex = z * ResY.x * ResY.y + y * ResY.x + x;

					SrcVectorFieldADataY[CurLinearIndex] = y;
					SrcVectorFieldBDataY[CurLinearIndex] = y + 2;
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					Int CurLinearIndex = z * ResZ.x * ResZ.y + y * ResZ.x + x;

					SrcVectorFieldADataZ[CurLinearIndex] = z;
					SrcVectorFieldBDataZ[CurLinearIndex] = z + 3;
				}
			}
		}

		WeightFieldAData[0] = 0.3;
		WeightFieldAData[1] = 0.5;
		WeightFieldBData[0] = 0.9;
		WeightFieldBData[1] = 0.2;

		CFaceCenteredVectorField FCVSrcVectorFieldA(Res, Origin, Spacing, SrcVectorFieldADataX.data(), SrcVectorFieldADataY.data(), SrcVectorFieldADataZ.data());
		CFaceCenteredVectorField FCVSrcVectorFieldB(Res, Origin, Spacing, SrcVectorFieldBDataX.data(), SrcVectorFieldBDataY.data(), SrcVectorFieldBDataZ.data());
		CCellCenteredScalarField CCSWeightFieldA(Res, Origin, Spacing, WeightFieldAData.data());
		CCellCenteredScalarField CCSWeightFieldB(Res, Origin, Spacing, WeightFieldBData.data());

		mixFieldWithDensityInvoker
		(
			FCVSrcVectorFieldA,
			FCVSrcVectorFieldB,
			CCSWeightFieldA,
			CCSWeightFieldB
		);

		thrust::host_vector<Real> MixResultX = FCVSrcVectorFieldB.getConstGridDataX();
		thrust::host_vector<Real> MixResultY = FCVSrcVectorFieldB.getConstGridDataY();
		thrust::host_vector<Real> MixResultZ = FCVSrcVectorFieldB.getConstGridDataZ();

		for (int z = 0; z < ResX.z; z++)
		{
			for (int y = 0; y < ResX.y; y++)
			{
				for (int x = 0; x < ResX.x; x++)
				{
					Int CurLinearIndex = z * ResX.x * ResX.y + y * ResX.x + x;

					if (CurLinearIndex == 0)
						EXPECT_LT(abs(MixResultX[CurLinearIndex] - 0.75), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 1)
						EXPECT_LT(abs(MixResultX[CurLinearIndex] - 1.578947), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 2)
						EXPECT_LT(abs(MixResultX[CurLinearIndex] - 2.285714), GRID_SOLVER_EPSILON);
					else
						EXPECT_LT(abs(MixResultX[CurLinearIndex] - x - 1.0), GRID_SOLVER_EPSILON);
				}
			}
		}
		for (int z = 0; z < ResY.z; z++)
		{
			for (int y = 0; y < ResY.y; y++)
			{
				for (int x = 0; x < ResY.x; x++)
				{
					Int CurLinearIndex = z * ResY.x * ResY.y + y * ResY.x + x;

					if (CurLinearIndex == 0)
						EXPECT_LT(abs(MixResultY[CurLinearIndex] - 1.5), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 1)
						EXPECT_LT(abs(MixResultY[CurLinearIndex] - 0.571428), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 2)
						EXPECT_LT(abs(MixResultY[CurLinearIndex] - 2.5), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 3)
						EXPECT_LT(abs(MixResultY[CurLinearIndex] - 1.571429), GRID_SOLVER_EPSILON);
					else
						EXPECT_LT(abs(MixResultY[CurLinearIndex] - y - 2.0), GRID_SOLVER_EPSILON);
				}
			}
		}
		for (int z = 0; z < ResZ.z; z++)
		{
			for (int y = 0; y < ResZ.y; y++)
			{
				for (int x = 0; x < ResZ.x; x++)
				{
					Int CurLinearIndex = z * ResZ.x * ResZ.y + y * ResZ.x + x;

					if (CurLinearIndex == 0)
						EXPECT_LT(abs(MixResultZ[CurLinearIndex] - 2.25), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 1)
						EXPECT_LT(abs(MixResultZ[CurLinearIndex] - 0.857143), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 4)
						EXPECT_LT(abs(MixResultZ[CurLinearIndex] - 3.25), GRID_SOLVER_EPSILON);
					else if (CurLinearIndex == 5)
						EXPECT_LT(abs(MixResultZ[CurLinearIndex] - 1.857143), GRID_SOLVER_EPSILON);
					else
						EXPECT_LT(abs(MixResultZ[CurLinearIndex] - z - 3.0), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//从BBOX生成初始流体域
TEST(HybridHelper, generateFluidDomainFromBBox)
{
	CCudaContextManager::getInstance().initCudaContext();

	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		Vector3 Min = Origin + 1.6 * Spacing;
		Vector3 Max = Origin + 2.4 * Spacing;

		CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.2, Res, Origin, Spacing);
		GridFluidSolver.generateFluidDomainFromBBox(Min, Max);

		CCellCenteredScalarField CCSDstFluidDomainField = GridFluidSolver.getFluidDomainField();

		thrust::host_vector<Real> FluidDomainResult = CCSDstFluidDomainField.getConstGridData();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurLinearIndex = z * Res.x * Res.y + y * Res.x + x;
					EXPECT_LT(abs(FluidDomainResult[CurLinearIndex] - FluidDomainValue), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	{
		Vector3i Res = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(-10.0, -20.0, -30.0);
		Vector3 Spacing = Vector3(10.0, 20.0, 30.0);
		Vector3 Min = Origin + 1.6 * Spacing;
		Vector3 Max = Origin + 3.6 * Spacing;

		CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.2, Res, Origin, Spacing);
		GridFluidSolver.generateFluidDomainFromBBox(Min, Max);

		CCellCenteredScalarField CCSDstFluidDomainField = GridFluidSolver.getFluidDomainField();

		thrust::host_vector<Real> FluidDomainResult = CCSDstFluidDomainField.getConstGridData();

		for (int z = 0; z < Res.z; z++)
		{
			for (int y = 0; y < Res.y; y++)
			{
				for (int x = 0; x < Res.x; x++)
				{
					Int CurLinearIndex = z * Res.x * Res.y + y * Res.x + x;

					if 
						(CurLinearIndex == 42 || CurLinearIndex == 43 || 
						 CurLinearIndex == 46 || CurLinearIndex == 47 || 
						 CurLinearIndex == 58 || CurLinearIndex == 59 ||
						 CurLinearIndex == 62 || CurLinearIndex == 63
			            )
					{
						EXPECT_LT(abs(FluidDomainResult[CurLinearIndex] + FluidDomainValue), GRID_SOLVER_EPSILON);
					}
					else
					{
						EXPECT_LT(abs(FluidDomainResult[CurLinearIndex] - FluidDomainValue), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//buildFluidSDF
TEST(HybridHelper, buildFluidInsideSDF)
{
	CCudaContextManager::getInstance().initCudaContext();

	//buildFluidInsideSDF
	{
		Vector3i Res = Vector3i(5, 5, 5);
		vector<Real> FluidDomainScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> SolidDomainScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -SolidDomainValue;
					}
					else
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = SolidDomainValue;
					}
					if (j < 3)
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					}
					else
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					}
				}
			}
		}

		CCellCenteredScalarField CCSFluidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSSolidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSDstFluidDensityField(Res);
		CCellCenteredScalarField CCSDstFluidInsideSDFField(Res);

		buildFluidDensityInvoker
		(
			CCSFluidDomainField,
			CCSSolidDomainField,
			CCSDstFluidDensityField
		);

		thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		vector<Real> DebugResult(Result.begin(), Result.end());

		buildFluidInsideSDFInvoker
		(
			CCSDstFluidDensityField,
			CCSSolidDomainField,
			CCSDstFluidInsideSDFField,
			20
		);

		Result = CCSDstFluidInsideSDFField.getConstGridData();
		vector<Real> DebugResult2(Result.begin(), Result.end());

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						if (j == 1)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - (0.5 + FluidSurfaceDensity)), GRID_SOLVER_EPSILON);
						}
						else if (j == 2)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - max((FluidSurfaceDensity - 0.5), 0.0)), GRID_SOLVER_EPSILON);
						}
						else
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
						}
					}
				}
			}
		}
	}

	//buildFluidOutsideSDF
	{
		Vector3i Res = Vector3i(5, 5, 5);
		vector<Real> FluidDomainScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> SolidDomainScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -SolidDomainValue;
					}
					else
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = SolidDomainValue;
					}
					if (j < 3)
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					}
					else
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					}
				}
			}
		}

		CCellCenteredScalarField CCSFluidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSSolidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSDstFluidDensityField(Res);
		CCellCenteredScalarField CCSDstFluidOutsideSDFField(Res);

		buildFluidDensityInvoker
		(
			CCSFluidDomainField,
			CCSSolidDomainField,
			CCSDstFluidDensityField
		);

		thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		vector<Real> DebugResult(Result.begin(), Result.end());

		buildFluidOutsideSDFInvoker
		(
			CCSDstFluidDensityField,
			CCSSolidDomainField,
			CCSDstFluidOutsideSDFField,
			20
		);

		Result = CCSDstFluidOutsideSDFField.getConstGridData();
		vector<Real> DebugResult2(Result.begin(), Result.end());

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						if (j == 1)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
						}
						else if (j == 2)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - max((0.5 - FluidSurfaceDensity), 0.0)), GRID_SOLVER_EPSILON);
						}
						else
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - (1.5 - FluidSurfaceDensity)), GRID_SOLVER_EPSILON);
						}
					}
				}
			}
		}
	}

	//buildFluidOutsideSDF2
	{
		Vector3i Res = Vector3i(5, 5, 5);
		vector<Real> FluidDomainScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> SolidDomainScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> SrcFluidDensityScalarFieldData(Res.x * Res.y * Res.z, 0.0);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -SolidDomainValue;
					}
					else
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = SolidDomainValue;
					}
					if (j < 3)
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					}
					else
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					}
				}
			}
		}

		CCellCenteredScalarField CCSFluidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSSolidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSSrcFluidDensityField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcFluidDensityScalarFieldData.data());
		CCellCenteredScalarField CCSDstFluidDensityField(Res);
		CCellCenteredScalarField CCSDstFluidOutsideSDFField(Res);

		buildFluidDensityInvoker
		(
			CCSFluidDomainField,
			CCSSolidDomainField,
			CCSDstFluidDensityField
		);

		//CCSDstFluidDensityField += CCSSrcFluidDensityField;

		thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		vector<Real> DebugResult(Result.begin(), Result.end());

		buildFluidOutsideSDFInvoker
		(
			CCSDstFluidDensityField,
			CCSSolidDomainField,
			CCSDstFluidOutsideSDFField,
			20
		);

		Result = CCSDstFluidOutsideSDFField.getConstGridData();
		vector<Real> DebugResult2(Result.begin(), Result.end());

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						if (j == 1)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
						}
						else if (j == 2)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - max((0.5 - FluidSurfaceDensity), 0.0)), GRID_SOLVER_EPSILON);
						}
						else
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - (1.5 - FluidSurfaceDensity)), GRID_SOLVER_EPSILON);
						}
					}
				}
			}
		}
	}

	//buildFluidDensity
	{
		Vector3i Res = Vector3i(5, 5, 5);
		vector<Real> FluidDomainScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> SolidDomainScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -SolidDomainValue;
					}
					else
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = SolidDomainValue;
					}
					if (j < 3)
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					}
					else
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					}
				}
			}
		}

		CCellCenteredScalarField CCSFluidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSSolidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSDstFluidDensityField(Res);

		buildFluidDensityInvoker
		(
			CCSFluidDomainField,
			CCSSolidDomainField,
			CCSDstFluidDensityField
		);

		thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		vector<Real> DebugResult(Result.begin(), Result.end());

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (i == 0 || i == 4 || j == 0 || j == 4 || k == 0 || k == 4)
					{
						EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
					}
					else
					{
						if (j == 1)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 1.0), GRID_SOLVER_EPSILON);
						}
						else if (j == 2)
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - FluidSurfaceDensity), GRID_SOLVER_EPSILON);
						}
						else
						{
							EXPECT_LT(abs(Result[i * Res.x * Res.y + j * Res.x + k] - 0.0), GRID_SOLVER_EPSILON);
						}
					}
				}
			}
		}
	}

	//buildFluidDensity2
	{
		Vector3i Res = Vector3i(5, 5, 1);
		vector<Real> FluidDomainScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> SolidDomainScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (j == 0 || j == 4 || k == 0 || k == 4)
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -SolidDomainValue;
					}
					else
					{
						SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = SolidDomainValue;
					}
					if (j < 3 && k < 3)
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					}
					else
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					}
				}
			}
		}

		CCellCenteredScalarField CCSFluidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSSolidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSDstFluidDensityField(Res);

		buildFluidDensityInvoker
		(
			CCSFluidDomainField,
			CCSSolidDomainField,
			CCSDstFluidDensityField
		);

		thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		vector<Real> DebugResult(Result.begin(), Result.end());
	}

	//buildFluidDensity3
	{
		Vector3i Res = Vector3i(321, 321, 1);
		vector<Real> FluidDomainScalarFieldData(Res.x * Res.y * Res.z);
		vector<Real> SolidDomainScalarFieldData(Res.x * Res.y * Res.z);

		for (int i = 0; i < Res.z; i++)
		{
			for (int j = 0; j < Res.y; j++)
			{
				for (int k = 0; k < Res.x; k++)
				{
					if (k > 0 && k < 91 && j > 0 && j < 151)
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					}
					else
					{
						FluidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					}

					SolidDomainScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = std::min(std::min((k - 0.5), (j - 0.5)), (std::min(Res.x - 1.5 - k, Res.x - 1.5 - j)));
				}
			}
		}

		CCellCenteredScalarField CCSFluidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSSolidDomainField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidDomainScalarFieldData.data());
		CCellCenteredScalarField CCSDstFluidDensityField(Res);

		buildFluidDensityInvoker
		(
			CCSFluidDomainField,
			CCSSolidDomainField,
			CCSDstFluidDensityField
		);

		thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		vector<Real> DebugResult(Result.begin(), Result.end());
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
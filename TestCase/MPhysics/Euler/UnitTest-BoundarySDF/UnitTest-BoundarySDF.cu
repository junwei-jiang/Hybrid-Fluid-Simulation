#include "pch.h"
#include "Boundary.h"
#include "BoundaryTool.cuh"
#include "BoundaryHelper.h"
#include "GridFluidSolver.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>

//GenerateBoundarySDF≤‚ ‘
//TEST(GenerateBoundarySDF, GenerateBoundarySDF)
//{
//	CCudaContextManager::getInstance().initCudaContext();
//
//	{
//		Vector3i Res = Vector3i(2, 2, 2);
//
//		CCellCenteredScalarField SDFField(Res, Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0));
//
//		generateSDF("./Cube.obj", SDFField);
//
//		thrust::host_vector<Real> SDF = SDFField.getConstGridData();
//		vector<Real> SDFResult(SDF.begin(), SDF.end());
//
//		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
//		{
//			EXPECT_LT(abs(SDFResult[i] + 0.5), GRID_SOLVER_EPSILON);
//		}
//	}
//	
//	{
//		Vector3i Res = Vector3i(1, 1, 1);
//
//		CCellCenteredScalarField SDFField(Res, Vector3(-1.0, -1.0, -1.0), Vector3(2.0, 2.0, 2.0));
//
//		generateSDF("./Cube.obj", SDFField);
//
//		thrust::host_vector<Real>SDF = SDFField.getConstGridData();
//		vector<Real> SDFResult = vector<Real>(SDF.begin(), SDF.end());
//
//		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
//		{
//			EXPECT_LT(abs(SDFResult[i] + 1.0), GRID_SOLVER_EPSILON);
//		}
//	}
//
//	{
//		Vector3i Res = Vector3i(1, 1, 1);
//
//		CCellCenteredScalarField SDFField(Res, Vector3(-1.0, -1.0, -1.0), Vector3(2.0, 2.0, 2.0));
//
//		generateSDF("./Cube.obj", SDFField, true);
//
//		thrust::host_vector<Real>SDF = SDFField.getConstGridData();
//		vector<Real> SDFResult = vector<Real>(SDF.begin(), SDF.end());
//
//		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
//		{
//			EXPECT_LT(abs(SDFResult[i] - 1.0), GRID_SOLVER_EPSILON);
//		}
//	}
//
//	{
//		Vector3i Res = Vector3i(3, 3, 3);
//
//		CCellCenteredScalarField SDFField(Res, Vector3(-3, -3, -3), Vector3(2, 2, 2));
//
//		generateSDF("./Cube.obj", SDFField, true);
//
//		thrust::host_vector<Real>SDF = SDFField.getConstGridData();
//		vector<Real> SDFResult = vector<Real>(SDF.begin(), SDF.end());
//
//		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
//		{
//			//EXPECT_LT(abs(SDFResult[i] - 1.0), GRID_SOLVER_EPSILON);
//		}
//	}
//
//	CCudaContextManager::getInstance().freeCudaContext();
//}

//ReSamplingBoundarySDF≤‚ ‘
TEST(GenerateBoundarySDF, ReSamplingBoundarySDF)
{
	CCudaContextManager::getInstance().initCudaContext();

	{
		Vector3i Res = Vector3i(2, 2, 2);

		CCellCenteredVectorField PosField(Res, Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0));

		thrust::device_vector<Real> Translation(3);
		thrust::device_vector<Real> Rotation(3);
		thrust::device_vector<Real> Scale(3);
		transformFieldInvoker(PosField, Translation, Rotation, Scale, 0);

		thrust::host_vector<Real> PosX = PosField.getConstGridDataX();
		thrust::host_vector<Real> PosY = PosField.getConstGridDataY();
		thrust::host_vector<Real> PosZ = PosField.getConstGridDataZ();
		vector<Real> PosXResult(PosX.begin(), PosX.end());
		vector<Real> PosYResult(PosY.begin(), PosY.end());
		vector<Real> PosZResult(PosZ.begin(), PosZ.end());

		CCellCenteredScalarField SDFField(Res, Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0));

		generateSDF("./Cube.obj", SDFField);

		thrust::host_vector<Real> SDF = SDFField.getConstGridData();
		vector<Real> SDFResult(SDF.begin(), SDF.end());

		CCellCenteredScalarField ResamplingSDFField(Res, Vector3(-1.0, -1.0, -1.0), Vector3(1.0, 1.0, 1.0));

		SDFField.sampleField(PosField, ResamplingSDFField);

		thrust::host_vector<Real> ResamplingSDF = ResamplingSDFField.getConstGridData();
		vector<Real> ResamplingSDFResult(ResamplingSDF.begin(), ResamplingSDF.end());

		for (int i = 0; i < Res.x * Res.y * Res.z; i++)
		{
			EXPECT_LT(abs(SDFResult[i] + 0.5), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(ResamplingSDFResult[i] + 0.5), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//updateTotalBoundarysSDF&Marker≤‚ ‘
TEST(GenerateBoundarySDF, updateTotalBoundarysSDFAndMarker)
{
	CCudaContextManager::getInstance().initCudaContext();

	{
		Vector3i Resolution = Vector3i(4, 4, 4);
		Vector3 Origin = Vector3(1.0, 3.0, -10.0);
		Vector3 Spacing = Vector3(11.0, 3.40, -140.0);

		vector<Real> ASolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 100);
		vector<Real> BSolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 200);

		for (int i = 0; i < Resolution.z; i++)
		{
			for (int j = 0; j < Resolution.y; j++)
			{
				for (int k = 0; k < Resolution.x; k++)
				{
					if (k == 0 || k == Resolution.x - 1 || j == 0 || j == Resolution.y - 1 || i == 0 || i == Resolution.z - 1)
					{
						ASolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -1;
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
					else if (k == 1 || k == Resolution.x - 2 || j == 1 || j == Resolution.y - 2 || i == 1 || i == Resolution.z - 2)
					{
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
				}
			}
		}

		CCellCenteredScalarField CCSASolidSDFField(Resolution, Origin, Spacing, ASolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSBSolidSDFField(Resolution, Origin, Spacing, BSolidSDFScalarFieldData.data());

		CBoundarys Boundarys(Resolution, Origin, Spacing);

		Boundarys.addBoundary(CCSASolidSDFField);
		Boundarys.addBoundary(CCSBSolidSDFField);

		Boundarys.updateBoundarys(0.0);

		CCellCenteredScalarField TotalBoundarysMarker = Boundarys.getTotalBoundarysMarker();
		CCellCenteredScalarField TotalBoundarysSDF = Boundarys.getTotalBoundarysSDF();

		thrust::host_vector<Real> DstMarker = TotalBoundarysMarker.getConstGridData();
		thrust::host_vector<Real> DstSDF = TotalBoundarysSDF.getConstGridData();

		vector<Real> DstMarkerResult(DstMarker.begin(), DstMarker.end());

		for (int i = 0; i < Resolution.z; i++)
		{
			for (int j = 0; j < Resolution.y; j++)
			{
				for (int k = 0; k < Resolution.x; k++)
				{
					if (k == 0 || k == Resolution.x - 1 || j == 0 || j == Resolution.y - 1 || i == 0 || i == Resolution.z - 1)
					{
						EXPECT_LT(abs(DstMarker[i * Resolution.x * Resolution.y + j * Resolution.x + k] + 1), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(DstSDF[i * Resolution.x * Resolution.y + j * Resolution.x + k] + 1), GRID_SOLVER_EPSILON);
					}
					else if (k == 1 || k == Resolution.x - 2 || j == 1 || j == Resolution.y - 2 || i == 1 || i == Resolution.z - 2)
					{
						EXPECT_LT(abs(DstMarker[i * Resolution.x * Resolution.y + j * Resolution.x + k] + 2), GRID_SOLVER_EPSILON);
						EXPECT_LT(abs(DstSDF[i * Resolution.x * Resolution.y + j * Resolution.x + k] + 2), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//updateTotalBoundarysVel≤‚ ‘
TEST(GenerateBoundarySDF, updateTotalBoundarysVel)
{
	CCudaContextManager::getInstance().initCudaContext();

	{
		Vector3i Resolution = Vector3i(12, 44, 14);
		Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
		Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
		Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(1.0, 3.0, -10.0);
		Vector3 Spacing = Vector3(11.0, 3.40, -140.0);

		vector<Real> ASolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 100);
		vector<Real> BSolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 200);

		for (int i = 0; i < Resolution.z; i++)
		{
			for (int j = 0; j < Resolution.y; j++)
			{
				for (int k = 0; k < Resolution.x; k++)
				{
					if (k == 0 || k == Resolution.x - 1 || j == 0 || j == Resolution.y - 1 || i == 0 || i == Resolution.z - 1)
					{
						ASolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -1;
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
					else if (k == 1 || k == Resolution.x - 2 || j == 1 || j == Resolution.y - 2 || i == 1 || i == Resolution.z - 2)
					{
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
				}
			}
		}

		CCellCenteredScalarField CCSASolidSDFField(Resolution, Origin, Spacing, ASolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSBSolidSDFField(Resolution, Origin, Spacing, BSolidSDFScalarFieldData.data());

		Vector3 VelocityA = Vector3(2, 3, 4);
		Vector3 VelocityB = Vector3(-2, -3, -4);

		CBoundarys Boundarys(Resolution, Origin, Spacing);

		Boundarys.addBoundary(CCSASolidSDFField, VelocityA);
		Boundarys.addBoundary(CCSBSolidSDFField, VelocityB);

		Boundarys.updateBoundarys(0.0);

		CFaceCenteredVectorField TotalBoundarysVel = Boundarys.getTotalBoundarysVel();

		thrust::host_vector<Real> DstVelX = TotalBoundarysVel.getConstGridDataX();
		thrust::host_vector<Real> DstVelY = TotalBoundarysVel.getConstGridDataY();
		thrust::host_vector<Real> DstVelZ = TotalBoundarysVel.getConstGridDataZ();

		vector<Real> DstVelXResult(DstVelX.begin(), DstVelX.end());

		for (int i = 0; i < ResolutionX.z; i++)
		{
			for (int j = 0; j < ResolutionX.y; j++)
			{
				for (int k = 0; k < ResolutionX.x; k++)
				{
					if (k == 0 || k == 1 ||k == ResolutionX.x - 1 || k == ResolutionX.x - 2 || j == 0 || j == ResolutionX.y - 1 || i == 0 || i == ResolutionX.z - 1)
					{
						EXPECT_LT(abs(abs(DstVelX[i * ResolutionX.x * ResolutionX.y + j * ResolutionX.x + k]) - 2.0), GRID_SOLVER_EPSILON);
					}
					else if (k == 2 || k == ResolutionX.x - 3 || j == 1 || j == ResolutionX.y - 2 || i == 1 || i == ResolutionX.z - 2)
					{
						EXPECT_LT(abs(DstVelX[i * ResolutionX.x * ResolutionX.y + j * ResolutionX.x + k] + 2.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}

		for (int i = 0; i < ResolutionY.z; i++)
		{
			for (int j = 0; j < ResolutionY.y; j++)
			{
				for (int k = 0; k < ResolutionY.x; k++)
				{
					if (k == 0 || k == ResolutionY.x - 1 || j == 0 || j == 1 || j == ResolutionY.y - 1 || j == ResolutionY.y - 2 || i == 0 || i == ResolutionY.z - 1)
					{
						EXPECT_LT(abs(abs(DstVelY[i * ResolutionY.x * ResolutionY.y + j * ResolutionY.x + k]) - 3.0), GRID_SOLVER_EPSILON);
					}
					else if (k == 1 || k == ResolutionY.x - 2 || j == 2 || j == ResolutionY.y - 3 || i == 1 || i == ResolutionY.z - 2)
					{
						EXPECT_LT(abs(DstVelY[i * ResolutionY.x * ResolutionY.y + j * ResolutionY.x + k] + 3.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}

		for (int i = 0; i < ResolutionZ.z; i++)
		{
			for (int j = 0; j < ResolutionZ.y; j++)
			{
				for (int k = 0; k < ResolutionZ.x; k++)
				{
					if (k == 0 || k == ResolutionZ.x - 1 || j == 0 || j == ResolutionZ.y - 1 || i == 0 || i == 1 || i == ResolutionZ.z - 1 || i == ResolutionZ.z - 2)
					{
						EXPECT_LT(abs(abs(DstVelZ[i * ResolutionZ.x * ResolutionZ.y + j * ResolutionZ.x + k]) - 4.0), GRID_SOLVER_EPSILON);
					}
					else if (k == 1 || k == ResolutionZ.x - 2 || j == 1 || j == ResolutionZ.y - 2 || i == 2 || i == ResolutionZ.z - 3)
					{
						EXPECT_LT(abs(DstVelZ[i * ResolutionZ.x * ResolutionZ.y + j * ResolutionZ.x + k] + 4.0), GRID_SOLVER_EPSILON);
					}
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//moveBoundarys≤‚ ‘
TEST(GenerateBoundarySDF, moveBoundarys)
{
	CCudaContextManager::getInstance().initCudaContext();

	{
		Vector3i Resolution = Vector3i(12, 44, 14);
		Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
		Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
		Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(1.0, 3.0, -10.0);
		Vector3 Spacing = Vector3(11.0, 3.40, -140.0);

		vector<Real> ASolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 100);
		vector<Real> BSolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 200);

		for (int i = 0; i < Resolution.z; i++)
		{
			for (int j = 0; j < Resolution.y; j++)
			{
				for (int k = 0; k < Resolution.x; k++)
				{
					if (k == 0 || k == Resolution.x - 1 || j == 0 || j == Resolution.y - 1 || i == 0 || i == Resolution.z - 1)
					{
						ASolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -1;
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
					else if (k == 1 || k == Resolution.x - 2 || j == 1 || j == Resolution.y - 2 || i == 1 || i == Resolution.z - 2)
					{
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
				}
			}
		}

		CCellCenteredScalarField CCSASolidSDFField(Resolution, Origin, Spacing, ASolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSBSolidSDFField(Resolution, Origin, Spacing, BSolidSDFScalarFieldData.data());

		vector<Vector3> Velocity = { Vector3(2, 3, 4), Vector3(-2, -3, -4) };

		CBoundarys Boundarys(Resolution);

		Boundarys.addBoundary(CCSASolidSDFField, Velocity[0]);
		Boundarys.addBoundary(CCSBSolidSDFField, Velocity[1]);

		Real DeltaT = 0.0;
		Boundarys.updateBoundarys(DeltaT);

		thrust::host_vector<Real> DstTranslation = Boundarys.getBoundarysTranslation();

		vector<Real> DstTranslationResult(DstTranslation.begin(), DstTranslation.end());

		for (int i = 0; i < Boundarys.getNumOfBoundarys(); i++)
		{
			EXPECT_LT(abs(DstTranslation[3 * i] - 0.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(DstTranslation[3 * i + 1] - 0.0), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(DstTranslation[3 * i + 2] - 0.0), GRID_SOLVER_EPSILON);
		}
	}
	
	{
		Vector3i Resolution = Vector3i(12, 44, 14);
		Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
		Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
		Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(1.0, 3.0, -10.0);
		Vector3 Spacing = Vector3(11.0, 3.40, -140.0);

		vector<Real> ASolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 100);
		vector<Real> BSolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 200);

		for (int i = 0; i < Resolution.z; i++)
		{
			for (int j = 0; j < Resolution.y; j++)
			{
				for (int k = 0; k < Resolution.x; k++)
				{
					if (k == 0 || k == Resolution.x - 1 || j == 0 || j == Resolution.y - 1 || i == 0 || i == Resolution.z - 1)
					{
						ASolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -1;
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
					else if (k == 1 || k == Resolution.x - 2 || j == 1 || j == Resolution.y - 2 || i == 1 || i == Resolution.z - 2)
					{
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
				}
			}
		}

		CCellCenteredScalarField CCSASolidSDFField(Resolution, Origin, Spacing, ASolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSBSolidSDFField(Resolution, Origin, Spacing, BSolidSDFScalarFieldData.data());

		vector<Vector3> Velocity = { Vector3(2, 3, 4), Vector3(-2, -3, -4) };

		CBoundarys Boundarys(Resolution);

		Boundarys.addBoundary(CCSASolidSDFField, Velocity[0]);
		Boundarys.addBoundary(CCSBSolidSDFField, Velocity[1]);

		Real DeltaT = 1.0;
		Boundarys.updateBoundarys(DeltaT);

		thrust::host_vector<Real> DstTranslation = Boundarys.getBoundarysTranslation();

		vector<Real> DstTranslationResult(DstTranslation.begin(), DstTranslation.end());

		for (int i = 0; i < Boundarys.getNumOfBoundarys(); i++)
		{
			EXPECT_LT(abs(DstTranslation[3 * i] + DeltaT * Velocity[i].x), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(DstTranslation[3 * i + 1] + DeltaT * Velocity[i].y), GRID_SOLVER_EPSILON);
			EXPECT_LT(abs(DstTranslation[3 * i + 2] + DeltaT * Velocity[i].z), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//reSamplingCurrentBoundarysSDF≤‚ ‘
TEST(GenerateBoundarySDF, reSamplingCurrentBoundarysSDF)
{
	CCudaContextManager::getInstance().initCudaContext();

	{
		Vector3i Resolution = Vector3i(4, 4, 4);
		Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
		Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
		Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);
		Vector3 Origin = Vector3(1.0, 3.0, -10.0);
		Vector3 Spacing = Vector3(11.0, 3.40, -140.0);

		vector<Real> ASolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 100);
		vector<Real> BSolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 200);

		for (int i = 0; i < Resolution.z; i++)
		{
			for (int j = 0; j < Resolution.y; j++)
			{
				for (int k = 0; k < Resolution.x; k++)
				{
					if (k == 0 || k == Resolution.x - 1 || j == 0 || j == Resolution.y - 1 || i == 0 || i == Resolution.z - 1)
					{
						ASolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -1;
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
					else if (k == 1 || k == Resolution.x - 2 || j == 1 || j == Resolution.y - 2 || i == 1 || i == Resolution.z - 2)
					{
						BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
					}
				}
			}
		}

		CCellCenteredScalarField CCSASolidSDFField(Resolution, Origin, Spacing, ASolidSDFScalarFieldData.data());
		CCellCenteredScalarField CCSBSolidSDFField(Resolution, Origin, Spacing, BSolidSDFScalarFieldData.data());

		vector<Vector3> Velocity = { Vector3(2, 3, 4), Vector3(-2, -3, -4) };

		CBoundarys Boundarys(Resolution, Origin, Spacing);

		Boundarys.addBoundary(CCSASolidSDFField, Velocity[0]);
		Boundarys.addBoundary(CCSBSolidSDFField, Velocity[1]);

		Real DeltaT = 0.0;
		Boundarys.updateBoundarys(DeltaT);

		thrust::host_vector<Real> DstBoundarySDFA = Boundarys.getCurrentBoundarysSDF(0).getConstGridData();
		thrust::host_vector<Real> DstBoundarySDFB = Boundarys.getCurrentBoundarysSDF(1).getConstGridData();

		vector<Real> DstBoundarySDFAResult(DstBoundarySDFA.begin(), DstBoundarySDFA.end());

		for (int i = 0; i < Resolution.z; i++)
		{
			for (int j = 0; j < Resolution.y; j++)
			{
				for (int k = 0; k < Resolution.x; k++)
				{
					Int CurIndex = i * Resolution.x * Resolution.y + j * Resolution.x + k;
					EXPECT_LT(abs(DstBoundarySDFA[CurIndex] - ASolidSDFScalarFieldData[CurIndex]), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(DstBoundarySDFB[CurIndex] - BSolidSDFScalarFieldData[CurIndex]), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	//{
	//	Vector3i Resolution = Vector3i(12, 44, 14);
	//	Vector3i ResolutionX = Resolution + Vector3i(1, 0, 0);
	//	Vector3i ResolutionY = Resolution + Vector3i(0, 1, 0);
	//	Vector3i ResolutionZ = Resolution + Vector3i(0, 0, 1);
	//	Vector3 Origin = Vector3(1.0, 3.0, -10.0);
	//	Vector3 Spacing = Vector3(11.0, 3.40, -140.0);

	//	vector<Real> ASolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 100);
	//	vector<Real> BSolidSDFScalarFieldData(Resolution.x * Resolution.y * Resolution.z, 200);

	//	for (int i = 0; i < Resolution.z; i++)
	//	{
	//		for (int j = 0; j < Resolution.y; j++)
	//		{
	//			for (int k = 0; k < Resolution.x; k++)
	//			{
	//				if (k == 0 || k == Resolution.x - 1 || j == 0 || j == Resolution.y - 1 || i == 0 || i == Resolution.z - 1)
	//				{
	//					ASolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -1;
	//					BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
	//				}
	//				else if (k == 1 || k == Resolution.x - 2 || j == 1 || j == Resolution.y - 2 || i == 1 || i == Resolution.z - 2)
	//				{
	//					BSolidSDFScalarFieldData[i * Resolution.x * Resolution.y + j * Resolution.x + k] = -2;
	//				}
	//			}
	//		}
	//	}

	//	CCellCenteredScalarField CCSASolidSDFField(Resolution, Origin, Spacing, ASolidSDFScalarFieldData.data());
	//	CCellCenteredScalarField CCSBSolidSDFField(Resolution, Origin, Spacing, BSolidSDFScalarFieldData.data());

	//	vector<Vector3> Velocity = { Vector3(2, 3, 4), Vector3(-2, -3, -4) };

	//	CBoundarys Boundarys(Resolution);

	//	Boundarys.addBoundary(CCSASolidSDFField, Velocity[0]);
	//	Boundarys.addBoundary(CCSBSolidSDFField, Velocity[1]);

	//	Real DeltaT = 1.0;
	//	Boundarys.updateBoundarys(DeltaT);

	//	thrust::host_vector<Real> DstTranslation = Boundarys.getBoundarysTranslation();

	//	vector<Real> DstTranslationResult(DstTranslation.begin(), DstTranslation.end());

	//	for (int i = 0; i < Boundarys.getNumOfBoundarys(); i++)
	//	{
	//		EXPECT_LT(abs(DstTranslation[3 * i] + DeltaT * Velocity[i].x), GRID_SOLVER_EPSILON);
	//		EXPECT_LT(abs(DstTranslation[3 * i + 1] + DeltaT * Velocity[i].y), GRID_SOLVER_EPSILON);
	//		EXPECT_LT(abs(DstTranslation[3 * i + 2] + DeltaT * Velocity[i].z), GRID_SOLVER_EPSILON);
	//	}
	//}

	CCudaContextManager::getInstance().freeCudaContext();
}
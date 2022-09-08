#include "pch.h"
#include "ExternalForcesSolver.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>

//applyExternalForces≤‚ ‘
TEST(ExternalForcesSolver, ExternalForcesSolver_ApplyExternalForces)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				VelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				VelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				VelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
			}
		}
	}

	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CExternalForcesSolver ExternalForcesSolver = CExternalForcesSolver(Res);

	ExternalForcesSolver.addExternalForces(Vector3(10, 0, 50));

	for (int n = 1; n < 10; n++)
	{
		ExternalForcesSolver.applyExternalForces(FCVVelField, 2);

		thrust::host_vector<Real> FieldResultDataX = FCVVelField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVVelField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVVelField.getConstGridDataZ();

		for (int i = 0; i < ResX.z; i++)
		{
			for (int j = 0; j < ResX.y; j++)
			{
				for (int k = 0; k < ResX.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataX[i * ResX.x * ResX.y + j * ResX.x + k] - (k + 10 * n * 2)), GRID_SOLVER_EPSILON);
				}
			}
		}

		for (int i = 0; i < ResY.z; i++)
		{
			for (int j = 0; j < ResY.y; j++)
			{
				for (int k = 0; k < ResY.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataY[i * ResY.x * ResY.y + j * ResY.x + k] - (j - 9.8 * n * 2)), GRID_SOLVER_EPSILON);
				}
			}
		}

		for (int i = 0; i < ResZ.z; i++)
		{
			for (int j = 0; j < ResZ.y; j++)
			{
				for (int k = 0; k < ResZ.x; k++)
				{
					EXPECT_LT(abs(FieldResultDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] - (i + 50 * n * 2)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	ExternalForcesSolver.resizeExternalForcesSolver(Res);
	FCVVelField.resize(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	for (int n = 1; n < 10; n++)
	{
		ExternalForcesSolver.applyExternalForces(FCVVelField, 2);

		thrust::host_vector<Real> FieldResultDataX = FCVVelField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVVelField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVVelField.getConstGridDataZ();

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
					EXPECT_LT(abs(FieldResultDataY[i * ResY.x * ResY.y + j * ResY.x + k] - (j - 9.8 * n * 2)), GRID_SOLVER_EPSILON);
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
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
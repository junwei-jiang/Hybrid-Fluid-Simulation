#pragma once
#include "pch.h"
#include "GridFluidSolver.h"
#include "CudaContextManager.h"
#include "GPUTimer.h"

#include <iostream>
#include <fstream>
#include <cmath>

void printData2TXT(const char *vFileName, const Vector3i& vRes, const CFaceCenteredVectorField& vVectorField)
{
	ofstream fout(vFileName);

	thrust::host_vector<Real> FieldData = vVectorField.getConstGridDataY();

	if (fout)
	{
		for (int i = 0; i < vRes.z; i++)
		{
			for (int j = 0; j < vRes.y; j++)
			{
				for (int k = 0; k < vRes.x; k++)
				{
					fout << abs(FieldData[i * vRes.x * vRes.y + j * vRes.x + k]) << " ";
					fout << abs(FieldData[i * vRes.x * vRes.y + j * vRes.x + k]) << " ";
					fout << abs(FieldData[i * vRes.x * vRes.y + j * vRes.x + k]) << "\n";
					//fout << (FieldData[i * vRes.x * vRes.y + j * vRes.x + k] + 2000.0) / 4000.0 * 255.0 << " ";
					//fout << (FieldData[i * vRes.x * vRes.y + j * vRes.x + k] + 2000.0) / 4000.0 * 255.0 << " ";
					//fout << (FieldData[i * vRes.x * vRes.y + j * vRes.x + k] + 2000.0) / 4000.0 * 255.0 << "\n";
				}
			}
		}
		fout.close();

		printf("OK \n");
	}
}

void printData2TXT(const char *vFileName, const Vector3i& vRes, const CCellCenteredScalarField& vScalarField)
{
	ofstream fout(vFileName);

	thrust::host_vector<Real> FieldData = vScalarField.getConstGridData();

	if (fout)
	{
		for (int i = 0; i < vRes.z; i++)
		{
			for (int j = 0; j < vRes.y; j++)
			{
				for (int k = 0; k < vRes.x; k++)
				{
					fout << FieldData[i * vRes.x * vRes.y + j * vRes.x + k] << " ";
					fout << FieldData[i * vRes.x * vRes.y + j * vRes.x + k] << " ";
					fout << FieldData[i * vRes.x * vRes.y + j * vRes.x + k] << "\n";
				}
			}
		}
		fout.close();

		printf("OK \n");
	}
}

void printSDFData2TXT(const char *vFileName, const Vector3i& vRes, const CCellCenteredScalarField& vScalarField)
{
	ofstream fout(vFileName);

	thrust::host_vector<Real> FieldData = vScalarField.getConstGridData();

	if (fout)
	{
		for (int i = 0; i < vRes.z; i++)
		{
			for (int j = 0; j < vRes.y; j++)
			{
				for (int k = 0; k < vRes.x; k++)
				{
					//if (FieldData[i * vRes.x * vRes.y + j * vRes.x + k] >= -1.0 && FieldData[i * vRes.x * vRes.y + j * vRes.x + k] <= 1.0)
					//{
					//	fout << 255 << " ";
					//	fout << 0 << " ";
					//	fout << 0 << "\n";
					//}
					//else
					//{
						fout << std::min(abs(FieldData[i * vRes.x * vRes.y + j * vRes.x + k]), static_cast<Real>(255.0)) << " ";
						fout << std::min(abs(FieldData[i * vRes.x * vRes.y + j * vRes.x + k]), static_cast<Real>(255.0)) << " ";
						fout << std::min(abs(FieldData[i * vRes.x * vRes.y + j * vRes.x + k]), static_cast<Real>(255.0)) << "\n";
					//}
				}
			}
		}
		fout.close();

		printf("OK \n");
	}
}

void printDensityData2TXT(const char *vFileName, const Vector3i& vRes, const CCellCenteredScalarField& vScalarField)
{
	ofstream fout(vFileName);

	thrust::host_vector<Real> FieldData = vScalarField.getConstGridData();

	if (fout)
	{
		for (int i = 0; i < vRes.z; i++)
		{
			for (int j = 0; j < vRes.y; j++)
			{
				for (int k = 0; k < vRes.x; k++)
				{
					if (FieldData[i * vRes.x * vRes.y + j * vRes.x + k] == 0.5)
					{
						fout << 255 << " ";
						fout << 0 << " ";
						fout << 0 << "\n";
					}
					else
					{
						fout << std::min(static_cast<Real>((FieldData[i * vRes.x * vRes.y + j * vRes.x + k] + 1.0) / 2.0 * 255.0), static_cast<Real>(255.0)) << " ";
						fout << std::min(static_cast<Real>((FieldData[i * vRes.x * vRes.y + j * vRes.x + k] + 1.0) / 2.0 * 255.0), static_cast<Real>(255.0)) << " ";
						fout << std::min(static_cast<Real>((FieldData[i * vRes.x * vRes.y + j * vRes.x + k] + 1.0) / 2.0 * 255.0), static_cast<Real>(255.0)) << "\n";
					}
				}
			}
		}
		fout.close();

		printf("OK \n");
	}
}

CCellCenteredScalarField readScalarDataFromTXT(const char *vFileName, const Vector3i& vRes)
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

	CCellCenteredScalarField ScalarField(vRes, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcVectorFieldDataX.data());
	//CCellCenteredVectorField VectorField(vRes, Vector3(0, 0, 0), Vector3(1, 1, 1), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());

	return ScalarField;
}

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

CCellCenteredVectorField readVectorDataFromTXT(const char *vFileName, const Vector3i& vRes)
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

//4×4×4网格，中间八个网格是水，上面中间四个网格是空气
TEST(GridFluidSolver, GridFluidSolver1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);

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

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1));
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	CGridFluidSolver GridFluidSolver = CGridFluidSolver(1, Res);

	GridFluidSolver.setFluidDomainField(CCSFluidSDFField);
	GridFluidSolver.addSolidBoundary(CCSSolidSDFField);
	GridFluidSolver.setSolidVelField(FCVSolidVelField);

	for (int i = 0; i < 3; i++)
	{
		GridFluidSolver.update();

		FCVFluidVelField = GridFluidSolver.getVelocityField();

		thrust::host_vector<Real> FieldResultDataX = FCVFluidVelField.getConstGridDataX();
		thrust::host_vector<Real> FieldResultDataY = FCVFluidVelField.getConstGridDataY();
		thrust::host_vector<Real> FieldResultDataZ = FCVFluidVelField.getConstGridDataZ();

		vector<Real> ResultX(FieldResultDataX.begin(), FieldResultDataX.end());
		vector<Real> ResultY(FieldResultDataY.begin(), FieldResultDataY.end());
		vector<Real> ResultZ(FieldResultDataZ.begin(), FieldResultDataZ.end());

		for (int j = 0; j < ResX.x * ResX.y * ResX.z; j++)
		{
			EXPECT_LT(abs(FieldResultDataX[j] - 0.0), GRID_SOLVER_EPSILON);
		}

		for (int j = 0; j < ResY.x * ResY.y * ResY.z; j++)
		{
			EXPECT_LT(abs(FieldResultDataY[j] - 0.0), GRID_SOLVER_EPSILON);
		}

		for (int j = 0; j < ResZ.x * ResZ.y * ResZ.z; j++)
		{
			EXPECT_LT(abs(FieldResultDataZ[j] - 0.0), GRID_SOLVER_EPSILON);
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//4×4×4网格，中间八个网格是水，上面和下面的中间八个网格是空气
TEST(GridFluidSolver, GridFluidSolver2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(4, 4, 4);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);

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

	CFaceCenteredVectorField FCVFluidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1));
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	Real DeltaT = 0.01;
	CGridFluidSolver GridFluidSolver = CGridFluidSolver(DeltaT, Res);

	GridFluidSolver.setFluidDomainField(CCSFluidSDFField);
	GridFluidSolver.addSolidBoundary(CCSSolidSDFField);
	GridFluidSolver.setSolidVelField(FCVSolidVelField);

	UInt SimulationTimes = 10;

	for (int i = 0; i < SimulationTimes; i++)
	{
		GridFluidSolver.update();

		FCVFluidVelField = GridFluidSolver.getVelocityField();

		thrust::host_vector<Real> FieldResultDataY = FCVFluidVelField.getConstGridDataY();

		EXPECT_LT(abs(FieldResultDataY[25] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[26] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[29] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[30] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[33] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[34] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);

		EXPECT_LT(abs(FieldResultDataY[45] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[46] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[49] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[50] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[53] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
		EXPECT_LT(abs(FieldResultDataY[54] + 9.8 * DeltaT * (i + 1)), GRID_SOLVER_EPSILON);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//641×641×1网格，太极
TEST(GridFluidSolver, GridFluidSolver3)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(641, 641, 1);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);

	Real TaichiRadius = 185.0;
	Vector3 TaichiCenter = Vector3(320.5, 320.5, 0.5);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				Vector3 CurRadius = Vector3(k + 0.5, j + 0.5, i + 0.5) - TaichiCenter;
				Real CurSDF = length(CurRadius) - TaichiRadius;
				FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = CurSDF;

				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = std::min(std::min((k - 0.5), (j - 0.5)), (std::min(Res.x - 1.5 - k, Res.x - 1.5 - j)));
			}
		}
	}
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (j == 0 || j == 640 || k == 0 || k == 640)
				{
					EXPECT_LT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
				else
				{
					EXPECT_GT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
			}
		}
	}

	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	//printData2TXT("D:\\My_Project\\PythonCode\\SolidSDF.txt", Res, CCSSolidSDFField);
	//printData2TXT("D:\\My_Project\\PythonCode\\FluidSDF.txt", Res, CCSFluidSDFField);

	CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.05, Res);

	string InputFileName = "D:\\My_Project\\PythonCode\\Taichi_blue.txt";

	CCellCenteredVectorField CCVSrcColorField = readVectorDataFromTXT(InputFileName.data(), Res);
	CCellCenteredVectorField CCVDstColorField(Res);

	GridFluidSolver.setColorField(CCVSrcColorField);
	GridFluidSolver.setFluidDomainField(CCSFluidSDFField);
	GridFluidSolver.addSolidBoundary(CCSSolidSDFField);
	GridFluidSolver.setSolidVelField(FCVSolidVelField);

	int Frames = 0;

	for (int i = 0; i < Frames; i++)
	{
		string OutputFileName = "D:\\My_Project\\PythonCode\\Taichi" + to_string(i + 1) + ".txt";

		GridFluidSolver.update();

		CCVDstColorField = GridFluidSolver.getColorField();

		printData2TXT(OutputFileName.data(), Res, CCVDstColorField);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//321×321×1网格，溃坝(SemiLagrangian)
TEST(GridFluidSolver, GridFluidSolver4)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3 Color = Vector3(0, 191, 254);
	//Vector3i Res = Vector3i(641, 641, 1);
	Vector3i Res = Vector3i(321, 321, 1);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidColorVectorFieldDataX(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataY(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataZ(Res.x * Res.y * Res.z, 0);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (k > 0 && k < 91 && j > 0 && j < 151)
				{
					FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -500;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = Color.x;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = Color.y;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = Color.z;
				}
				else
				{
					FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = 5;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = 255.0;
				}

				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = std::min(std::min((k - 0.5), (j - 0.5)), (std::min(Res.x - 1.5 - k, Res.x - 1.5 - j)));
			}
		}
	}
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (j == 0 || j == Res.y - 1 || k == 0 || k == Res.x - 1)
				{
					EXPECT_LT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
				else
				{
					EXPECT_GT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
			}
		}
	}

	CCellCenteredVectorField CCVSrcColorField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidColorVectorFieldDataX.data(), FluidColorVectorFieldDataY.data(), FluidColorVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstColorField(Res);
	CFaceCenteredVectorField FCVDstVelField(Res);
	CCellCenteredScalarField CCSDstDivergenceField(Res);
	CCellCenteredScalarField CCSDstMarkersField(Res);
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	//printData2TXT("D:\\My_Project\\PythonCode\\SolidSDF.txt", Res, CCSSolidSDFField);
	//printData2TXT("D:\\My_Project\\PythonCode\\FluidSDF.txt", Res, CCSFluidSDFField);

	CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.2, Res);

	GridFluidSolver.setColorField(CCVSrcColorField);
	GridFluidSolver.setFluidDomainField(CCSFluidSDFField);
	GridFluidSolver.addSolidBoundary(CCSSolidSDFField);
	GridFluidSolver.setSolidVelField(FCVSolidVelField);

	int Frames = 0;

	for (int i = 0; i < Frames; i++)
	{
		string OutputFileName = "D:\\My_Project\\PythonCode\\DamBreak" + to_string(i + 1) + ".txt";

		GridFluidSolver.update();

		CCVDstColorField = GridFluidSolver.getColorField();
		FCVDstVelField = GridFluidSolver.getVelocityField();
		CCSDstMarkersField = GridFluidSolver.getPressureSolver()->getMarkers();

		//thrust::host_vector<Real> TempMarkers = CCSDstMarkersField.getConstGridData();

		//FCVDstVelField.divergence(CCSDstDivergenceField);
		//thrust::host_vector<Real> TempDivergence = CCSDstDivergenceField.getConstGridData();
		//vector<Real> Divergence(TempDivergence.begin(), TempDivergence.end());

		//Real DivergenceMax = -REAL_MAX;

		//for (int z = 0; z < Res.z; z++)
		//{
		//	for (int y = 0; y < Res.y; y++)
		//	{
		//		for (int x = 0; x < Res.x; x++)
		//		{
		//			if (TempMarkers[z * Res.x * Res.y + y * Res.x + x] == 1.0)
		//			{
		//				DivergenceMax = max(DivergenceMax, abs(TempDivergence[z * Res.x * Res.y + y * Res.x + x]));
		//			}
		//		}
		//	}
		//}

		FCVDstVelField *= (100.0);
		printData2TXT(OutputFileName.data(), ResY, FCVDstVelField);
		//CCSDstMarkersField *= (100);
		//printData2TXT(OutputFileName.data(), Res, CCSDstMarkersField);
		//printData2TXT(OutputFileName.data(), Res, CCVDstColorField);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//321×321×1网格，溃坝(ParticleInCell)
TEST(GridFluidSolver, GridFluidSolver5)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3 Color = Vector3(0, 191, 254);
	//Vector3i Res = Vector3i(641, 641, 1);
	Vector3i Res = Vector3i(321, 321, 1);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidColorVectorFieldDataX(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataY(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataZ(Res.x * Res.y * Res.z, 0);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (k > 0 && k < 91 && j > 0 && j < 151)
				{
					FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = Color.x;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = Color.y;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = Color.z;
				}
				else
				{
					FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = 255.0;
				}

				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = std::min(std::min((k - 0.5), (j - 0.5)), (std::min(Res.x - 1.5 - k, Res.x - 1.5 - j)));
			}
		}
	}
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (j == 0 || j == Res.y - 1 || k == 0 || k == Res.x - 1)
				{
					EXPECT_LT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
				else
				{
					EXPECT_GT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
			}
		}
	}

	CCellCenteredVectorField CCVSrcColorField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidColorVectorFieldDataX.data(), FluidColorVectorFieldDataY.data(), FluidColorVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstColorField(Res);
	CFaceCenteredVectorField FCVDstVelField(Res);
	CCellCenteredScalarField CCSDstDivergenceField(Res);
	CCellCenteredScalarField CCSDstMarkersField(Res);
	CCellCenteredScalarField CCSDstFluidDomainField(Res);
	CCellCenteredScalarField CCSDstFluidSDFField(Res);
	CCellCenteredScalarField CCSDstFluidInsideSDFField(Res);
	CCellCenteredScalarField CCSDstFluidOutsideSDFField(Res);
	CCellCenteredScalarField CCSDstFluidDensityField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1));
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	//printData2TXT("D:\\My_Project\\PythonCode\\SolidSDF.txt", Res, CCSSolidSDFField);
	//printData2TXT("D:\\My_Project\\PythonCode\\FluidSDF.txt", Res, CCSFluidSDFField);

	ParticleInCellSolverPtr PICSolverPtr = make_shared<CParticleInCell>();
	PICSolverPtr->resizeParticleInCell(CCSFluidSDFField, CCSSolidSDFField, 8, Res, Vector3(0, 0, 0), Vector3(1, 1, 1));

	CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.2, Res);
	GridFluidSolver.setAdvectionSolver(PICSolverPtr);

	GridFluidSolver.setColorField(CCVSrcColorField);
	GridFluidSolver.setFluidDomainField(CCSFluidSDFField);
	GridFluidSolver.addSolidBoundary(CCSSolidSDFField);
	GridFluidSolver.setSolidVelField(FCVSolidVelField);

	int Frames = 0;

	for (int i = 0; i < Frames; i++)
	{
		string OutputFileName = "D:\\My_Project\\PythonCode\\DamBreak" + to_string(i + 1) + ".txt";

		GridFluidSolver.update();
		//GridFluidSolver.updateWithoutPressure();
		//GridFluidSolver.solvePressure();
		GridFluidSolver.generateCurFluidSDF();
		GridFluidSolver.generateCurFluidDensity();

		//CCVDstColorField = GridFluidSolver.getColorField();
		//FCVDstVelField = GridFluidSolver.getVelocityField();
		//CCSDstMarkersField = GridFluidSolver.getPressureSolver()->getMarkers();
		//auto TempEulerParticles = PICSolverPtr->getEulerParticles();
		//TempEulerParticles.statisticalFluidDensity(CCSDstFluidDensityField);
		CCSDstFluidDomainField = GridFluidSolver.getFluidDomainField();
		//CCSDstFluidSDFField = GridFluidSolver.getFluidSDFField();
		CCSDstFluidDensityField = GridFluidSolver.getFluidDensityField();

		buildFluidInsideSDFInvoker
		(
			CCSDstFluidDensityField,
			CCSSolidSDFField,
			CCSDstFluidInsideSDFField,
			1000
		);

		//thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		//vector<Real> DebugResult(Result.begin(), Result.end());

		//thrust::host_vector<Real> TempMarkers = CCSDstMarkersField.getConstGridData();

		//FCVDstVelField.divergence(CCSDstDivergenceField);
		//thrust::host_vector<Real> TempDivergence = CCSDstDivergenceField.getConstGridData();
		//vector<Real> Divergence(TempDivergence.begin(), TempDivergence.end());

		//Real DivergenceMax = -REAL_MAX;

		//for (int z = 0; z < Res.z; z++)
		//{
		//	for (int y = 0; y < Res.y; y++)
		//	{
		//		for (int x = 0; x < Res.x; x++)
		//		{
		//			if (TempMarkers[z * Res.x * Res.y + y * Res.x + x] == 1.0)
		//			{
		//				DivergenceMax = max(DivergenceMax, abs(TempDivergence[z * Res.x * Res.y + y * Res.x + x]));
		//			}
		//		}
		//	}
		//}

		//FCVDstVelField *= 10.0;
		//printData2TXT(OutputFileName.data(), ResY, FCVDstVelField);
		//CCSDstMarkersField *= (100);
		//printData2TXT(OutputFileName.data(), Res, CCSDstMarkersField);
		//printData2TXT(OutputFileName.data(), Res, CCVDstColorField);
		//CCSDstFluidDensityField *= (100);
		//printData2TXT(OutputFileName.data(), Res, CCSDstFluidDensityField);
		//CCSDstFluidSDFField *= 10.0;
		//printSDFData2TXT(OutputFileName.data(), Res, CCSDstFluidSDFField);
		//CCSDstFluidDomainField *= -1.0;
		//printData2TXT(OutputFileName.data(), Res, CCSDstFluidDomainField);
		CCSDstFluidInsideSDFField *= 10.0;
		printData2TXT(OutputFileName.data(), Res, CCSDstFluidInsideSDFField);
		//printDensityData2TXT(OutputFileName.data(), Res, CCSDstFluidDensityField);
		//printSurfaceData2TXT(OutputFileName.data(), Res, CCSDstFluidSDFField);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//321×321×1网格，溃坝(FluidImplicitParticle)
/*TEST(GridFluidSolver, GridFluidSolver6)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3 Color = Vector3(0, 191, 254);
	//Vector3i Res = Vector3i(641, 641, 1);
	Vector3i Res = Vector3i(321, 321, 1);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidColorVectorFieldDataX(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataY(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataZ(Res.x * Res.y * Res.z, 0);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (k > 0 && k < 91 && j > 0 && j < 151)
				{
					if (k == 90 || j == 150)
						FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -50;
					else
						FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -100;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = Color.x;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = Color.y;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = Color.z;
				}
				else
				{
					FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = 5;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = 255.0;
				}

				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = std::min(std::min((k - 0.5), (j - 0.5)), (std::min(Res.x - 1.5 - k, Res.x - 1.5 - j)));
			}
		}
	}
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (j == 0 || j == Res.y - 1 || k == 0 || k == Res.x - 1)
				{
					EXPECT_LT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
				else
				{
					EXPECT_GT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
			}
		}
	}

	CCellCenteredVectorField CCVSrcColorField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidColorVectorFieldDataX.data(), FluidColorVectorFieldDataY.data(), FluidColorVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstColorField(Res);
	CFaceCenteredVectorField FCVDstVelField(Res);
	CCellCenteredScalarField CCSDstDivergenceField(Res);
	CCellCenteredScalarField CCSDstMarkersField(Res);
	CCellCenteredScalarField CCSDstFluidDomainField(Res);
	CCellCenteredScalarField CCSDstFluidSDFField(Res);
	CCellCenteredScalarField CCSDstFluidDensityField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1));
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	//printData2TXT("D:\\My_Project\\PythonCode\\SolidSDF.txt", Res, CCSSolidSDFField);
	//printData2TXT("D:\\My_Project\\PythonCode\\FluidSDF.txt", Res, CCSFluidSDFField);

	FluidImplicitParticleSolverPtr FLIPSolverPtr = make_shared<CFluidImplicitParticle>();
	FLIPSolverPtr->resizeFluidImplicitParticle(CCSFluidSDFField, CCSSolidSDFField, 8, Res, Vector3(0, 0, 0), Vector3(1, 1, 1));

	CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.2, Res);
	GridFluidSolver.setAdvectionSolver(FLIPSolverPtr);

	//ParticleInCellSolverPtr PICSolverPtr = make_shared<CParticleInCell>();
	//PICSolverPtr->resizeParticleInCell(CCSFluidSDFField, CCSSolidSDFField, 8, Res, Vector3(0, 0, 0), Vector3(1, 1, 1));

	//CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.2, Res);
	//GridFluidSolver.setAdvectionSolver(PICSolverPtr);

	GridFluidSolver.setColorField(CCVSrcColorField);
	GridFluidSolver.setFluidDomainField(CCSFluidSDFField);
	GridFluidSolver.setSolidSDFField(CCSSolidSDFField);
	GridFluidSolver.setSolidVelField(FCVSolidVelField);

	int Frames = 100;

	for (int i = 0; i < Frames; i++)
	{
		string OutputFileName = "D:\\My_Project\\PythonCode\\DamBreak" + to_string(i + 1) + ".txt";

		GridFluidSolver.update();
		//GridFluidSolver.updateWithoutPressure();
		//GridFluidSolver.solvePressure();
		GridFluidSolver.generateCurFluidSDF();
		GridFluidSolver.generateCurFluidDensity();

		CCVDstColorField = GridFluidSolver.getColorField();
		FCVDstVelField = GridFluidSolver.getVelocityField();
		CCSDstMarkersField = GridFluidSolver.getPressureSolver()->getMarkers();
		auto TempEulerParticles = FLIPSolverPtr->getEulerParticles();
		TempEulerParticles.statisticalFluidDensity(CCSDstFluidDensityField);
		CCSDstFluidDomainField = GridFluidSolver.getFluidDomainField();
		CCSDstFluidSDFField = GridFluidSolver.getFluidSDFField();
		CCSDstFluidDensityField = GridFluidSolver.getFluidDensityField();

		//thrust::host_vector<Real> TempMarkers = CCSDstMarkersField.getConstGridData();

		//FCVDstVelField.divergence(CCSDstDivergenceField);
		//thrust::host_vector<Real> TempDivergence = CCSDstDivergenceField.getConstGridData();
		//vector<Real> Divergence(TempDivergence.begin(), TempDivergence.end());

		//Real DivergenceMax = -REAL_MAX;

		//for (int z = 0; z < Res.z; z++)
		//{
		//	for (int y = 0; y < Res.y; y++)
		//	{
		//		for (int x = 0; x < Res.x; x++)
		//		{
		//			if (TempMarkers[z * Res.x * Res.y + y * Res.x + x] == 1.0)
		//			{
		//				DivergenceMax = max(DivergenceMax, abs(TempDivergence[z * Res.x * Res.y + y * Res.x + x]));
		//			}
		//		}
		//	}
		//}

		//FCVDstVelField *= 10.0;
		//printData2TXT(OutputFileName.data(), ResY, FCVDstVelField);
		//CCSDstMarkersField *= (100);
		//printData2TXT(OutputFileName.data(), Res, CCSDstMarkersField);
		//printData2TXT(OutputFileName.data(), Res, CCVDstColorField);
		//CCSDstFluidDensityField *= (10);
		//printData2TXT(OutputFileName.data(), Res, CCSDstFluidDensityField);
		CCSDstFluidDomainField *= -1.0;
		printData2TXT(OutputFileName.data(), Res, CCSDstFluidDomainField);

		//printDensityData2TXT(OutputFileName.data(), Res, CCSDstFluidDensityField);
		//printSurfaceData2TXT(OutputFileName.data(), Res, CCSDstFluidSDFField);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}*/

//321×321×1网格，溃坝(ParticleInCell)
TEST(GridFluidSolver, GridFluidSolver7)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3 Color = Vector3(0, 191, 254);
	Vector3i Res = Vector3i(641, 641, 1);
	//Vector3i Res = Vector3i(321, 321, 1);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> FluidColorVectorFieldDataX(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataY(Res.x * Res.y * Res.z, 0);
	vector<Real> FluidColorVectorFieldDataZ(Res.x * Res.y * Res.z, 0);
	vector<Real> SolidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> FluidSDFScalarFieldData(Res.x * Res.y * Res.z);
	vector<Real> SolidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> SolidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> SolidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);
	vector<Real> FluidVelVectorFieldDataX(ResX.x * ResX.y * ResX.z, 0);
	vector<Real> FluidVelVectorFieldDataY(ResY.x * ResY.y * ResY.z, 0);
	vector<Real> FluidVelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z, 0);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (k > 0 && k < 91 && j > 0 && j < 151)
				{
					FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = -FluidDomainValue;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = Color.x;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = Color.y;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = Color.z;
				}
				else
				{
					FluidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = FluidDomainValue;
					FluidColorVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = 255.0;
					FluidColorVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = 255.0;
				}

				SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = std::min(std::min((k - 0.5), (j - 0.5)), (std::min(Res.x - 1.5 - k, Res.x - 1.5 - j)));
			}
		}
	}
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (j == 0 || j == Res.y - 1 || k == 0 || k == Res.x - 1)
				{
					EXPECT_LT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
				else
				{
					EXPECT_GT(SolidSDFScalarFieldData[i * Res.x * Res.y + j * Res.x + k], 0);
				}
			}
		}
	}

	//for (int i = 0; i < ResX.z; i++)
	//{
	//	for (int j = 0; j < ResX.y; j++)
	//	{
	//		for (int k = 0; k < ResX.z; k++)
	//		{
	//			int CurLinearIndex = i * ResX.x * ResX.y + j * ResX.x + k;
	//			Real U = 1.0f;
	//			Real a = 0.3f;
	//			Real r = 0;
	//			Vector2 TempVel = (U / a) * (2.0f - (r * r - a * a)) * exp(0.5f * (1.0f - (r * r - a * a)));
	//		}
	//	}
	//}

	CCellCenteredVectorField CCVSrcColorField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidColorVectorFieldDataX.data(), FluidColorVectorFieldDataY.data(), FluidColorVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstColorField(Res);
	CFaceCenteredVectorField FCVDstVelField(Res);
	CCellCenteredScalarField CCSDstDivergenceField(Res);
	CCellCenteredScalarField CCSDstMarkersField(Res);
	CCellCenteredScalarField CCSDstFluidDomainField(Res);
	CCellCenteredScalarField CCSDstFluidSDFField(Res);
	CCellCenteredScalarField CCSDstFluidInsideSDFField(Res);
	CCellCenteredScalarField CCSDstFluidOutsideSDFField(Res);
	CCellCenteredScalarField CCSDstFluidDensityField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1));
	CFaceCenteredVectorField FCVSolidVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidVelVectorFieldDataX.data(), SolidVelVectorFieldDataY.data(), SolidVelVectorFieldDataZ.data());
	CCellCenteredScalarField CCSSolidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), SolidSDFScalarFieldData.data());
	CCellCenteredScalarField CCSFluidSDFField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), FluidSDFScalarFieldData.data());

	ParticleInCellSolverPtr PICSolverPtr = make_shared<CParticleInCell>();
	PICSolverPtr->resizeParticleInCell(CCSFluidSDFField, CCSSolidSDFField, 8, Res, Vector3(0, 0, 0), Vector3(1, 1, 1));

	CGridFluidSolver GridFluidSolver = CGridFluidSolver(0.2, Res);
	GridFluidSolver.setAdvectionSolver(PICSolverPtr);

	GridFluidSolver.setColorField(CCVSrcColorField);
	GridFluidSolver.setFluidDomainField(CCSFluidSDFField);
	GridFluidSolver.addSolidBoundary(CCSSolidSDFField);
	GridFluidSolver.setSolidVelField(FCVSolidVelField);
	GridFluidSolver.addExternalForce(Vector3(0.0f, 9.8f, 0.0f));

	int Frames = 20;

	for (int i = 0; i < Frames; i++)
	{
		string OutputFileName = "D:\\My_Project\\PythonCode\\DamBreak" + to_string(i + 1) + ".txt";

		GridFluidSolver.update();
		GridFluidSolver.generateCurFluidSDF();
		GridFluidSolver.generateCurFluidDensity();

		//CCVDstColorField = GridFluidSolver.getColorField();
		//FCVDstVelField = GridFluidSolver.getVelocityField();
		CCSDstMarkersField = GridFluidSolver.getPressureSolver()->getMarkers();
		//auto TempEulerParticles = PICSolverPtr->getEulerParticles();
		//TempEulerParticles.statisticalFluidDensity(CCSDstFluidDensityField);
		//CCSDstFluidDomainField = GridFluidSolver.getFluidDomainField();
		//CCSDstFluidSDFField = GridFluidSolver.getFluidSDFField();
		//CCSDstFluidDensityField = GridFluidSolver.getFluidDensityField();

		//buildFluidInsideSDFInvoker
		//(
		//	CCSDstFluidDensityField,
		//	CCSSolidSDFField,
		//	CCSDstFluidInsideSDFField,
		//	1000
		//);

		//thrust::host_vector<Real> Result = CCSDstFluidDensityField.getConstGridData();
		//vector<Real> DebugResult(Result.begin(), Result.end());

		//thrust::host_vector<Real> TempMarkers = CCSDstMarkersField.getConstGridData();

		//FCVDstVelField.divergence(CCSDstDivergenceField);
		//thrust::host_vector<Real> TempDivergence = CCSDstDivergenceField.getConstGridData();
		//vector<Real> Divergence(TempDivergence.begin(), TempDivergence.end());

		//Real DivergenceMax = -REAL_MAX;

		//for (int z = 0; z < Res.z; z++)
		//{
		//	for (int y = 0; y < Res.y; y++)
		//	{
		//		for (int x = 0; x < Res.x; x++)
		//		{
		//			if (TempMarkers[z * Res.x * Res.y + y * Res.x + x] == 1.0)
		//			{
		//				DivergenceMax = max(DivergenceMax, abs(TempDivergence[z * Res.x * Res.y + y * Res.x + x]));
		//			}
		//		}
		//	}
		//}

		//FCVDstVelField *= 10.0;
		//printData2TXT(OutputFileName.data(), ResY, FCVDstVelField);
		CCSDstMarkersField *= (100);
		printData2TXT(OutputFileName.data(), Res, CCSDstMarkersField);
		//printData2TXT(OutputFileName.data(), Res, CCVDstColorField);
		//CCSDstFluidDensityField *= (100);
		//printData2TXT(OutputFileName.data(), Res, CCSDstFluidDensityField);
		//CCSDstFluidSDFField *= 10.0;
		//printSDFData2TXT(OutputFileName.data(), Res, CCSDstFluidSDFField);
		//CCSDstFluidDomainField *= -1.0;
		//printData2TXT(OutputFileName.data(), Res, CCSDstFluidDomainField);
		//CCSDstFluidInsideSDFField *= 10.0;
		//printData2TXT(OutputFileName.data(), Res, CCSDstFluidInsideSDFField);
		//printDensityData2TXT(OutputFileName.data(), Res, CCSDstFluidDensityField);
		//printSurfaceData2TXT(OutputFileName.data(), Res, CCSDstFluidSDFField);
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
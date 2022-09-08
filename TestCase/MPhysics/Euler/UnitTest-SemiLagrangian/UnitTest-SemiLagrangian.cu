#include "pch.h"
#include "AdvectionSolver.h"
#include "SemiLagrangian.h"
#include "CudaContextManager.cpp"
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

//TEST(SemiLagrangian, SemiLagrangian_Visualization1)
//{
//	CCudaContextManager::getInstance().initCudaContext();
//
//	Vector3i Res = Vector3i(641, 641, 1);
//	Vector3i ResX = Res + Vector3i(1, 0, 0);
//	Vector3i ResY = Res + Vector3i(0, 1, 0);
//	Vector3i ResZ = Res + Vector3i(0, 0, 1);
//
//	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
//	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
//	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
//
//	for (int i = 0; i < ResX.z; i++)
//	{
//		for (int j = 0; j < ResX.y; j++)
//		{
//			for (int k = 0; k < ResX.x; k++)
//			{
//				VelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 0;
//			}
//		}
//	}
//
//	for (int i = 0; i < ResY.z; i++)
//	{
//		for (int j = 0; j < ResY.y; j++)
//		{
//			for (int k = 0; k < ResY.x; k++)
//			{
//				VelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
//			}
//		}
//	}
//
//	for (int i = 0; i < ResZ.z; i++)
//	{
//		for (int j = 0; j < ResZ.y; j++)
//		{
//			for (int k = 0; k < ResZ.x; k++)
//			{
//				VelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
//			}
//		}
//	}
//
//	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());
//
//	int Frames = 0;
//
//	for (int i = 0; i < Frames; i++)
//	{
//		string InputFileName = "D:\\My_Project\\PythonCode\\Taichi" + to_string(i) + ".txt";
//		string OutputFileName = "D:\\My_Project\\PythonCode\\Taichi" + to_string(i + 1) + ".txt";
//
//		CCellCenteredVectorField CCVSrcField = readDataFromTXT(InputFileName.data(), Res);
//		CCellCenteredVectorField CCVDstField = CCVSrcField;
//
//		CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res);
//
//		SemiLagrangianSolver.advect(CCVSrcField, FCVVelField, 10, CCVDstField);
//
//		printData2TXT(OutputFileName.data(), Res, CCVDstField);
//	}
//
//	CCudaContextManager::getInstance().freeCudaContext();
//}

//TEST(SemiLagrangian, SemiLagrangian_Visualization2)
//{
//	CCudaContextManager::getInstance().initCudaContext();
//
//	Vector3i Res = Vector3i(641, 641, 1);
//	Vector3i ResX = Res + Vector3i(1, 0, 0);
//	Vector3i ResY = Res + Vector3i(0, 1, 0);
//	Vector3i ResZ = Res + Vector3i(0, 0, 1);
//
//	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
//	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
//	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);
//
//	Vector3 VectorZ = Vector3(0, 0, -1);
//	Real R = 320;
//
//	for (int i = 0; i < ResX.z; i++)
//	{
//		for (int j = 0; j < ResX.y; j++)
//		{
//			for (int k = 0; k < ResX.x; k++)
//			{
//				Vector3 RelPos = Vector3(320, 320, 0.5) - Vector3(k, j + 0.5, i + 0.5);
//				Real Alpha = length(RelPos) / R;
//				Vector3 VelDir = cross(VectorZ, RelPos);
//				VelDir = normalize(VelDir);
//				VelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = -VelDir.x * Alpha;
//			}
//		}
//	}
//
//	for (int i = 0; i < ResY.z; i++)
//	{
//		for (int j = 0; j < ResY.y; j++)
//		{
//			for (int k = 0; k < ResY.x; k++)
//			{
//				Vector3 RelPos = Vector3(320, 320, 0.5) - Vector3(k + 0.5, j, i + 0.5);
//				Real Alpha = length(RelPos) / R;
//				Vector3 VelDir = cross(VectorZ, RelPos);
//				VelDir = normalize(VelDir);
//				VelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = -VelDir.y * Alpha;
//			}
//		}
//	}
//
//	for (int i = 0; i < ResZ.z; i++)
//	{
//		for (int j = 0; j < ResZ.y; j++)
//		{
//			for (int k = 0; k < ResZ.x; k++)
//			{
//				VelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
//			}
//		}
//	}
//
//	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(1, 1, 1), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());
//
//	int Frames = 0;
//
//	for (int i = 0; i < Frames; i++)
//	{
//		string InputFileName = "D:\\My_Project\\PythonCode\\Taichi" + to_string(i) + ".txt";
//		string OutputFileName = "D:\\My_Project\\PythonCode\\Taichi" + to_string(i + 1) + ".txt";
//
//		CCellCenteredVectorField CCVSrcField = readDataFromTXT(InputFileName.data(), Res);
//		CCellCenteredVectorField CCVDstField = CCVSrcField;
//
//		CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res);
//
//		SemiLagrangianSolver.advect(CCVSrcField, FCVVelField, 20, CCVDstField, EBackTraceAlgorithm::RK1);
//
//		printData2TXT(OutputFileName.data(), Res, CCVDstField);
//	}
//
//	CCudaContextManager::getInstance().freeCudaContext();
//}

//BackTrace测试(速度场恒定)
TEST(SemiLagrangian, SemiLagrangian_BackTrace1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcPosVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataZ(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcPosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 + 5;
				SrcPosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 + 10;
				SrcPosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 + 15;
			}
		}
	}

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				VelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 1;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				VelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 2;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				VelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 3;
			}
		}
	}

	CCellCenteredVectorField CCVSrcPosField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcPosVectorFieldDataX.data(), SrcPosVectorFieldDataY.data(), SrcPosVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstPosField = CCVSrcPosField;
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res);

	SemiLagrangianSolver.backTrace(CCVSrcPosField, FCVVelField, 5, CCVDstPosField);

	thrust::host_vector<Real> FieldResultDataX = CCVDstPosField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVDstPosField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVDstPosField.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - k * 10), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - j * 20), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - i * 30), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//BackTrace测试(速度场为0)
TEST(SemiLagrangian, SemiLagrangian_BackTrace2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcPosVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataZ(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcPosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 + 5;
				SrcPosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 + 10;
				SrcPosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 + 15;
			}
		}
	}

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				VelVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				VelVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = 0;
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				VelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = 0;
			}
		}
	}

	CCellCenteredVectorField CCVSrcPosField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcPosVectorFieldDataX.data(), SrcPosVectorFieldDataY.data(), SrcPosVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstPosField = CCVSrcPosField;
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res);

	SemiLagrangianSolver.backTrace(CCVSrcPosField, FCVVelField, 5, CCVDstPosField);

	thrust::host_vector<Real> FieldResultDataX = CCVDstPosField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVDstPosField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVDstPosField.getConstGridDataZ();
	
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - (k * 10 + 5)), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - (j * 20 + 10)), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - (i * 30 + 15)), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//BackTrace测试(RK1)
TEST(SemiLagrangian, SemiLagrangian_BackTrace3)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcPosVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataZ(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcPosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 + 5;
				SrcPosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 + 10;
				SrcPosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 + 15;
			}
		}
	}

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

	CCellCenteredVectorField CCVSrcPosField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcPosVectorFieldDataX.data(), SrcPosVectorFieldDataY.data(), SrcPosVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstPosField = CCVSrcPosField;
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res);

	SemiLagrangianSolver.backTrace(CCVSrcPosField, FCVVelField, 5, CCVDstPosField, EAdvectionAccuracy::RK1);

	thrust::host_vector<Real> FieldResultDataX = CCVDstPosField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVDstPosField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVDstPosField.getConstGridDataZ();
	
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - (k * 10 + 5 - (k + 0.5) * 5)), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - (j * 20 + 10 - (j + 0.5) * 5)), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - (i * 30 + 15 - (i + 0.5) * 5)), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//BackTrace测试(RK2)
TEST(SemiLagrangian, SemiLagrangian_BackTrace4)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcPosVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataZ(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcPosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 + 5;
				SrcPosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 + 10;
				SrcPosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 + 15;
			}
		}
	}

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


	CCellCenteredVectorField CCVSrcPosField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcPosVectorFieldDataX.data(), SrcPosVectorFieldDataY.data(), SrcPosVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstPosField = CCVSrcPosField;
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res);

	SemiLagrangianSolver.backTrace(CCVSrcPosField, FCVVelField, 5, CCVDstPosField);

	thrust::host_vector<Real> FieldResultDataX = CCVDstPosField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVDstPosField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVDstPosField.getConstGridDataZ();	

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - (k * 10 + 5 - 5 * (k * 10 + 5 - (k + 0.5) * 5 * 0.5) / 10)), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - (j * 20 + 10 - 5 * (j * 20 + 10 - (j + 0.5) * 5 * 0.5) / 20)), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - (i * 30 + 15 - 5 * (i * 30 + 15 - (i + 0.5) * 5 * 0.5) / 30)), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//BackTrace测试(RK3)
TEST(SemiLagrangian, SemiLagrangian_BackTrace5)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcPosVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcPosVectorFieldDataZ(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcPosVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 + 5;
				SrcPosVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 + 10;
				SrcPosVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 + 15;
			}
		}
	}

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


	CCellCenteredVectorField CCVSrcPosField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcPosVectorFieldDataX.data(), SrcPosVectorFieldDataY.data(), SrcPosVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstPosField = CCVSrcPosField;
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res);

	SemiLagrangianSolver.backTrace(CCVSrcPosField, FCVVelField, 5, CCVDstPosField, EAdvectionAccuracy::RK3);

	thrust::host_vector<Real> FieldResultDataX = CCVDstPosField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVDstPosField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVDstPosField.getConstGridDataZ();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				Real Vel1X = (k * 10.0 + 5.0) / 10.0;
				Real Vel2X = ((k * 10.0 + 5.0) - 0.5 * 5 * Vel1X) / 10.0;
				Real Vel3X = ((k * 10.0 + 5.0) - 0.75 * 5 * Vel2X) / 10.0;
				Real DstPosX = k * 10 + 5 - 5 * (2.0 / 9.0 * Vel1X + 3.0 / 9.0 * Vel2X + 4.0 / 9.0 * Vel3X);
				Real Vel1Y = (j * 20.0 + 10.0) / 20.0;
				Real Vel2Y = ((j * 20.0 + 10.0) - 0.5 * 5 * Vel1Y) / 20.0;
				Real Vel3Y = ((j * 20.0 + 10.0) - 0.75 * 5 * Vel2Y) / 20.0;
				Real DstPosY = j * 20.0 + 10.0 - 5 * (2.0 / 9.0 * Vel1Y + 3.0 / 9.0 * Vel2Y + 4.0 / 9.0 * Vel3Y);
				Real Vel1Z = (i * 30 + 15) / 30.0;
				Real Vel2Z = ((i * 30 + 15) - 0.5 * 5 * Vel1Z) / 30.0;
				Real Vel3Z = ((i * 30 + 15) - 0.75 * 5 * Vel2Z) / 30.0;
				Real DstPosZ = i * 30 + 15 - 5 * (2.0 / 9.0 * Vel1Z + 3.0 / 9.0 * Vel2Z + 4.0 / 9.0 * Vel3Z);
				EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - DstPosX), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - DstPosY), GRID_SOLVER_EPSILON);
				EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - DstPosZ), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//AdvectionCCS场
TEST(SemiLagrangian, SemiLagrangian_AdvectionSolver1)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 1, 1);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = k * 100 + 50;
			}
		}
	}

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

	CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcScalarFieldData.data());
	CCellCenteredScalarField CCSDstField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	SemiLagrangianSolver.advect(CCSSrcField, FCVVelField, 5, CCSDstField);

	thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();
	
	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				if (k == 0)
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res.x * Res.y + j * Res.x + k] - SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k]), GRID_SOLVER_EPSILON);
				}
				else
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res.x * Res.y + j * Res.x + k] - 10 * (k * 10 + 5 - 5 * (k * 10 + 5 - (k + 0.5) * 5 * 0.5) / 10)), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//AdvectionCCS场(稍微复杂的情况)
TEST(SemiLagrangian, SemiLagrangian_AdvectionSolver2)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcScalarFieldData(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcScalarFieldData[i * Res.x * Res.y + j * Res.x + k] = k * 10 + 5 + j * 20 + 10 + i * 30 + 15;
			}
		}
	}

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

	CCellCenteredScalarField CCSSrcField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcScalarFieldData.data());
	CCellCenteredScalarField CCSDstField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	SemiLagrangianSolver.advect(CCSSrcField, FCVVelField, 5, CCSDstField);

	thrust::host_vector<Real> DstFieldResultData = CCSDstField.getConstGridData();

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				Real TempX = (k * 10 + 5 - 5 * (k * 10 + 5 - (k + 0.5) * 5 * 0.5) / 10);
				Real TempY = (j * 20 + 10 - 5 * (j * 20 + 10 - (j + 0.5) * 5 * 0.5) / 20);
				Real TempZ = (i * 30 + 15 - 5 * (i * 30 + 15 - (i + 0.5) * 5 * 0.5) / 30);
				if (k == 0)
				{
					TempX = 5;
				}
				else if (j == 0)
				{
					TempY = 10;
				}
				else if (i == 0)
				{
					TempZ = 15;
				}
				else
				{
					EXPECT_LT(abs(DstFieldResultData[i * Res.x * Res.y + j * Res.x + k] - TempX - TempY - TempZ), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//AdvectionCCV场
TEST(SemiLagrangian, SemiLagrangian_AdvectionSolver3)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcVectorFieldDataX(Res.x * Res.y * Res.z);
	vector<Real> SrcVectorFieldDataY(Res.x * Res.y * Res.z);
	vector<Real> SrcVectorFieldDataZ(Res.x * Res.y * Res.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				SrcVectorFieldDataX[i * Res.x * Res.y + j * Res.x + k] = k * 10 + 5;
				SrcVectorFieldDataY[i * Res.x * Res.y + j * Res.x + k] = j * 20 + 10;
				SrcVectorFieldDataZ[i * Res.x * Res.y + j * Res.x + k] = i * 30 + 15;
			}
		}
	}

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

	CCellCenteredVectorField CCVSrcField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
	CCellCenteredVectorField CCVDstField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	SemiLagrangianSolver.advect(CCVSrcField, FCVVelField, 5, CCVDstField);

	thrust::host_vector<Real> FieldResultDataX = CCVDstField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = CCVDstField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = CCVDstField.getConstGridDataZ();
	

	for (int i = 0; i < Res.z; i++)
	{
		for (int j = 0; j < Res.y; j++)
		{
			for (int k = 0; k < Res.x; k++)
			{
				Real TempX = (k * 10 + 5 - 5 * (k * 10 + 5 - (k + 0.5) * 5 * 0.5) / 10);
				Real TempY = (j * 20 + 10 - 5 * (j * 20 + 10 - (j + 0.5) * 5 * 0.5) / 20);
				Real TempZ = (i * 30 + 15 - 5 * (i * 30 + 15 - (i + 0.5) * 5 * 0.5) / 30);
				if (k == 0)
				{
					TempX = 5;
				}
				else if (j == 0)
				{
					TempY = 10;
				}
				else if (i == 0)
				{
					TempZ = 15;
				}
				else
				{
					EXPECT_LT(abs(FieldResultDataX[i * Res.x * Res.y + j * Res.x + k] - TempX), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataY[i * Res.x * Res.y + j * Res.x + k] - TempY), GRID_SOLVER_EPSILON);
					EXPECT_LT(abs(FieldResultDataZ[i * Res.x * Res.y + j * Res.x + k] - TempZ), GRID_SOLVER_EPSILON);
				}
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}

//AdvectionFCV场
TEST(SemiLagrangian, SemiLagrangian_AdvectionSolver4)
{
	CCudaContextManager::getInstance().initCudaContext();

	Vector3i Res = Vector3i(14, 15, 16);
	Vector3i ResX = Res + Vector3i(1, 0, 0);
	Vector3i ResY = Res + Vector3i(0, 1, 0);
	Vector3i ResZ = Res + Vector3i(0, 0, 1);

	vector<Real> SrcVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> SrcVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> SrcVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	vector<Real> VelVectorFieldDataX(ResX.x * ResX.y * ResX.z);
	vector<Real> VelVectorFieldDataY(ResY.x * ResY.y * ResY.z);
	vector<Real> VelVectorFieldDataZ(ResZ.x * ResZ.y * ResZ.z);

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				SrcVectorFieldDataX[i * ResX.x * ResX.y + j * ResX.x + k] = k * 10;
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
				SrcVectorFieldDataY[i * ResY.x * ResY.y + j * ResY.x + k] = j * 20;
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
				SrcVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i * 30;
				VelVectorFieldDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] = i;
			}
		}
	}

	CFaceCenteredVectorField FCVSrcField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), SrcVectorFieldDataX.data(), SrcVectorFieldDataY.data(), SrcVectorFieldDataZ.data());
	CFaceCenteredVectorField FCVDstField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));
	CFaceCenteredVectorField FCVVelField(Res, Vector3(0, 0, 0), Vector3(10, 20, 30), VelVectorFieldDataX.data(), VelVectorFieldDataY.data(), VelVectorFieldDataZ.data());

	CSemiLagrangian SemiLagrangianSolver = CSemiLagrangian(Res, Vector3(0, 0, 0), Vector3(10, 20, 30));

	SemiLagrangianSolver.advect(FCVSrcField, FCVVelField, 5, FCVDstField);

	thrust::host_vector<Real> FieldResultDataX = FCVDstField.getConstGridDataX();
	thrust::host_vector<Real> FieldResultDataY = FCVDstField.getConstGridDataY();
	thrust::host_vector<Real> FieldResultDataZ = FCVDstField.getConstGridDataZ();
	

	for (int i = 0; i < ResX.z; i++)
	{
		for (int j = 0; j < ResX.y; j++)
		{
			for (int k = 0; k < ResX.x; k++)
			{
				Real TempX = (k * 10 - 5 * (k * 10 - k * 5 * 0.5) / 10);
				EXPECT_LT(abs(FieldResultDataX[i * ResX.x * ResX.y + j * ResX.x + k] - TempX), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResY.z; i++)
	{
		for (int j = 0; j < ResY.y; j++)
		{
			for (int k = 0; k < ResY.x; k++)
			{
				Real TempY = (j * 20 - 5 * (j * 20 - j * 5 * 0.5) / 20);
				EXPECT_LT(abs(FieldResultDataY[i * ResY.x * ResY.y + j * ResY.x + k] - TempY), GRID_SOLVER_EPSILON);
			}
		}
	}

	for (int i = 0; i < ResZ.z; i++)
	{
		for (int j = 0; j < ResZ.y; j++)
		{
			for (int k = 0; k < ResZ.x; k++)
			{
				Real TempZ = (i * 30 - 5 * (i * 30 - i * 5 * 0.5) / 30);
				EXPECT_LT(abs(FieldResultDataZ[i * ResZ.x * ResZ.y + j * ResZ.x + k] - TempZ), GRID_SOLVER_EPSILON);
			}
		}
	}

	CCudaContextManager::getInstance().freeCudaContext();
}
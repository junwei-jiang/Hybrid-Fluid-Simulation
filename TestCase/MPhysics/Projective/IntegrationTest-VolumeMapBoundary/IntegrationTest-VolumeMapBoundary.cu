#pragma once
#include "pch.h"

#include "SPHKernelFunc.cuh"
#include "GaussParamCPU.h"
#include "BoundaryVolumeMap.h"
#include "CudaContextManager.h"

TEST(Boundary_VolumeMap, queryVolumeAndClosestPoint_Cube)
{
	CCudaContextManager::getInstance().initCudaContext();
	initAgzUtilsDX11Device();

	CRigidBodyBoundaryVolumeMap Box;

	vector<string> Path;
	vector<SBoundaryTransform> Transforms;
	Path.push_back("./Cube.obj");
	SBoundaryTransform BoundaryTransform;
	BoundaryTransform.Pos = Vector3(0, 0, 0);
	BoundaryTransform.Scale = Vector3(1, 1, 1);
	BoundaryTransform.Rotaion.row0 = Vector3(1, 0, 0);
	BoundaryTransform.Rotaion.row1 = Vector3(0, 1, 0);
	BoundaryTransform.Rotaion.row2 = Vector3(0, 0, 1);
	Transforms.push_back(BoundaryTransform);

	Box.bindBoundaryMesh(
		Path,
		Transforms,
		Vector3ui(32, 32, 32),
		0.1,
		true,
		false,
		"."
	);

	CCubicKernel CubicSmoothKernel;
	CubicSmoothKernel.setRadius(0.1);

	SCLDGridInfo GridInfo = Box.getCLDGridInfo();
	thrust::device_vector<Real> SDFField = Box.getField(0);
	thrust::device_vector<Real> VolumeField = Box.getField(1);
	thrust::device_vector<UInt> Cell = Box.getCell();
	Vector3ui Res = GridInfo.Resolution;

	UInt NodeIndex = Cell[GridInfo.multiToSingleIndex(Vector3ui(3, 3, 3)) * NodePerCell + 30];
	Real VolumeData = VolumeField[NodeIndex];
	EXPECT_LE(abs(VolumeData - 0.00343926), FLT_EPSILON);

	NodeIndex = Cell[GridInfo.multiToSingleIndex(Vector3ui(16, 16, 16)) * NodePerCell + 30];
	VolumeData = VolumeField[NodeIndex];
	EXPECT_LE(abs(VolumeData - 0), FLT_EPSILON);

	freeAgzUtilsDX11Device();
	CCudaContextManager::getInstance().freeCudaContext();
}
#pragma once
#include "pch.h"

#include "BoundaryVolumeMap.h"
#include "CudaContextManager.h"
#include "SPHKernelFunc.cuh"
__global__ void sample
(
	const Vector3* vPos, 
	UInt vPosCount, 
	cudaTextureObject_t vResourceObject,
	Vector3ui vSDFRes,
	bool vLogDebugInfo,

	float* voSampleResult
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vPosCount) return;
	voSampleResult[Index] = tex3D<float>(
		vResourceObject, 
		vPos[Index].x, 
		vPos[Index].y, 
		vPos[Index].z
	);
}

TEST(Boundary_VolumeMap, SDFData_Cube)
{
	CCudaContextManager::getInstance().initCudaContext();
	initAgzUtilsDX11Device();

	CRigidBodyBoundaryVolumeMap VolumeMap;
	Vector3ui SDFRes = Vector3ui(8, 8, 8);

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

	VolumeMap.bindBoundaryMesh(Path, Transforms, SDFRes, 0.1, true, false, ".");

	SAABB SDFDomain = VolumeMap.getSDFDomain();
	SDFRes = VolumeMap.getSDFRes();
	Vector3 SDFCellSize = SDFDomain.getDia() / castToVector3(SDFRes);

	cudaTextureObject_t SDF3dTexture = VolumeMap.getExtSDFCuda3dTexture();

	thrust::device_vector<Vector3> NormalizePos(SDFRes.x * SDFRes.y * SDFRes.z);
	vector<Real> GrandTruth(SDFRes.x * SDFRes.y * SDFRes.z);
	SAABB Cube;
	Cube.Max = Vector3(1, 1, 1);
	Cube.Min = Vector3(-1, -1, -1);
	for (UInt i = 0; i < SDFRes.z; i++)
		for (UInt k = 0; k < SDFRes.y; k++)
			for (UInt j = 0; j < SDFRes.x; j++)
			{
				Vector3 Pos = SDFDomain.Min + SDFCellSize * (Vector3(j, k, i) + 0.5);
				NormalizePos[SDFRes.x * SDFRes.y * i + SDFRes.x * k + j] = Vector3(j, k, i);
				GrandTruth[SDFRes.x * SDFRes.y * i + SDFRes.x * k + j] = Cube.getDistance(Pos);
			}

	thrust::device_vector<float> SampleResult(NormalizePos.size());

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(NormalizePos.size(), BlockSize, GridSize);
	sample<<<GridSize, BlockSize>>>
	(
		raw_pointer_cast(NormalizePos.data()),
		NormalizePos.size(),
		SDF3dTexture,
		SDFRes,
		true,
		raw_pointer_cast(SampleResult.data())
	);

	thrust::host_vector<float> ResultCPU = SampleResult;
	UInt Index = 0;
	for (auto GrandTruthValue : GrandTruth)
	{
		EXPECT_LE(abs(abs(GrandTruthValue) - abs(ResultCPU[Index++])), 1e-5) << Index - 1;
	}

	freeAgzUtilsDX11Device();
	CCudaContextManager::getInstance().freeCudaContext();
}
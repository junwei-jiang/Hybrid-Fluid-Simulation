#include "BoundaryVolumeMapKernel.cuh"
#include "CudaContextManager.h"

__global__ void queryVolumeAndClosestPoint
(
	const Real* vDistData,
	Real* vioQueryPos,
	Real* vioParticleVel,
	UInt vQueryDataSize,
	Real vSupportRadius,
	Real vParticleRadius,

	Real* vioVolumeData,
	Real* vioClosestPos,

	Vector3 vGridTransform,
	SMatrix3x3 vGridRotation,
	SMatrix3x3 vInvGridRotation
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vQueryDataSize) return;

	Real Dist = vDistData[Index];
	Vector3 X = ParticlePos(vioQueryPos, Index);
	Vector3 GridObjectSapceX = vGridRotation * (X - vGridTransform);

	Real VolumeData = vioVolumeData[Index];
	Vector3 Normal = ParticlePos(vioClosestPos, Index);
	if (Dist > 0.0 && Dist < vSupportRadius)
	{
		if (VolumeData > 0.0 && VolumeData != REAL_MAX)
		{
			if (length(Normal) > 1.0e-9)
			{
				Vector3 norm = normalize(Normal);
				Real d = thrust::max(Dist + 0.5 * vParticleRadius, 2.0 * vParticleRadius);
				Vector3 ClosestPoint = GridObjectSapceX - d * norm;

				ClosestPoint = vInvGridRotation * ClosestPoint + vGridTransform;

				ParticlePosPortion(vioClosestPos, Index, 0) = ClosestPoint.x;
				ParticlePosPortion(vioClosestPos, Index, 1) = ClosestPoint.y;
				ParticlePosPortion(vioClosestPos, Index, 2) = ClosestPoint.z;
			}
			else
			{
				VolumeData = 0.0;
			}
		}
		else
		{
			VolumeData = 0.0;
		}
	}
	else if (Dist <= 0.0) //粒子已出界
	{
		if (length(Normal) > 1.0e-5)
		{
			Vector3 norm = normalize(Normal);

			Real Delta = 2.0 * vParticleRadius - Dist;
			Delta = thrust::min(Delta, vParticleRadius * static_cast<Real>(0.1));		// get up in small steps
			Vector3 NewPos = GridObjectSapceX + Delta * norm;

			NewPos = vInvGridRotation * NewPos + vGridTransform;

			ParticlePosPortion(vioQueryPos, Index, 0) = NewPos.x;
			ParticlePosPortion(vioQueryPos, Index, 1) = NewPos.y;
			ParticlePosPortion(vioQueryPos, Index, 2) = NewPos.z;

			ParticlePosPortion(vioParticleVel, Index, 0) = 0;
			ParticlePosPortion(vioParticleVel, Index, 1) = 0;
			ParticlePosPortion(vioParticleVel, Index, 2) = 0;
		}
		VolumeData = 0.0;
	}
	else
	{
		VolumeData = 0.0;
	}
	vioVolumeData[Index] = VolumeData;
}

void queryVolumeAndClosestPointInvoker
(
	const Real* vDistData,
	Real* vioQueryPos,
	Real* vioParticleVel,
	UInt vQueryDataSize,
	Real vSupportRadius,
	Real vParticleRadius,

	Real* vioVolumeData,
	Real* vioClosestPos,

	Vector3 vGridTransform,
	SMatrix3x3 vGridRotation,
	SMatrix3x3 vInvGridRotation
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vQueryDataSize, BlockSize, GridSize);

	queryVolumeAndClosestPoint LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vDistData,
		vioQueryPos,
		vioParticleVel,
		vQueryDataSize,
		vSupportRadius,
		vParticleRadius,
		vioVolumeData,
		vioClosestPos,
		vGridTransform,
		vGridRotation,
		vInvGridRotation
	);
	CUDA_SYN
}
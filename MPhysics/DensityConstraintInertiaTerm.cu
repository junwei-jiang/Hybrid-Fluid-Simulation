#include "DensityConstraintKernel.cuh"
#include "CudaContextManager.h"
#include "Particle.h"
__global__ void computeInertiaTerm
(
	Real* voXn,
	const Real* vVn,
	const Real* vAn,
	UInt vParticleSize,
	Real vDeltaT
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vParticleSize) return;
	Vector3 X = ParticlePos(voXn, Index);
	Vector3 Vel = ParticlePos(vVn, Index);
	Vector3 Acc = ParticlePos(vAn, Index);
	ParticlePosPortion(voXn, Index, 0) = X.x + vDeltaT * Vel.x;
	ParticlePosPortion(voXn, Index, 1) = X.y + vDeltaT * Vel.y;
	ParticlePosPortion(voXn, Index, 2) = X.z + vDeltaT * Vel.z;
}

void computeInertiaTermInvoker
(
	Real* voXn,
	const Real* vVn,
	const Real* vAn,
	UInt vParticleSize,
	Real vDeltaT
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);
	computeInertiaTerm LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		voXn,
		vVn, 
		vAn,
		vParticleSize,
		vDeltaT
	); 
	CUDA_SYN
}
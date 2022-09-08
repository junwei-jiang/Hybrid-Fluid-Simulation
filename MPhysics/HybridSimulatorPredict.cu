#include "HybridSimulatorKernel.cuh"
#include "Particle.h"
#include "CudaContextManager.h"

__global__ void predictParticleVel
(
	Real* voVel,
	UInt vParticleSize,
	Vector3 vExtAcc,
	Real vDeltaTime
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vParticleSize) return;
	vExtAcc *= vDeltaTime;
	ParticlePosPortion(voVel, Index, 0) += vExtAcc.x;
	ParticlePosPortion(voVel, Index, 1) += vExtAcc.y;
	ParticlePosPortion(voVel, Index, 2) += vExtAcc.z;
}

void predictParticleVelInvoker
(
	thrust::device_vector<Real>& voVel,
	UInt vParticleSize,
	Vector3 vExtAcc,
	Real vDeltaTime
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);

	predictParticleVel LANCH_PARAM_1D_GB(GridSize, BlockSize)
	(
		thrust::raw_pointer_cast(voVel.data()),
		vParticleSize,
		vExtAcc,
		vDeltaTime
	);
}
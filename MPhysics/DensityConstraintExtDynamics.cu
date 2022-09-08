#include "DensityConstraintKernel.cuh"
#include "CudaContextManager.h"
#include "SimulationConfigManager.h"
#include "Particle.h"

__global__ void computeAcceleration
(
	Real* voAcceleration,
	UInt vParticleSize,
	Real vGravity
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vParticleSize) return;

	ParticlePosPortion(voAcceleration, Index, 0) = 0;
	ParticlePosPortion(voAcceleration, Index, 1) = -vGravity;
	ParticlePosPortion(voAcceleration, Index, 2) = 0;

}
void computeAccelerationInvoker
(
	CCuDenseVector& voAcceleration,
	Real vGravity
)
{
	UInt ParticleSize = voAcceleration.getSize() / 3;
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(ParticleSize, BlockSize, GridSize);
	computeAcceleration LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		voAcceleration.getVectorValueGPUPtr(),
		voAcceleration.getSize() / 3,
		vGravity
	);
	CUDA_SYN
}

__global__ void computeVelLength
(
	const Real* vVel,
	UInt vParticleSize,

	Real* voVelLength
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vParticleSize) return;

	voVelLength[Index] = length(ParticlePos(vVel, Index));
}

Real updateTimeStepInvoker
(
	const CCuDenseVector& vParticleVel,
	const CCuDenseVector& vParticleAcc,
	UInt vParticleSize,
	Real vCFLFactor,
	Real vParticleRadius,
	Real vMaxTimeStepSize,
	Real vMinTimeStepSize
)
{
	CCuDenseVector PredictVel = vParticleVel;
	PredictVel.plusAlphaX(vParticleAcc, CSimulationConfigManager::getInstance().getTimeStep());

	CCuDenseVector VelLength(vParticleSize);
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);
	computeVelLength LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		PredictVel.getConstVectorValueGPUPtr(),
		vParticleSize,
		VelLength.getVectorValueGPUPtr()
	);
	CUDA_SYN

	Real MaxVel = *thrust::max_element(VelLength.getVectorValue().begin(), VelLength.getVectorValue().end());

	Real Diameter = 2.0 * vParticleRadius;
	Real NewTimeStep = vCFLFactor * 0.4 * (Diameter / sqrt(MaxVel));
	
	NewTimeStep = thrust::min(NewTimeStep, vMaxTimeStepSize);
	NewTimeStep = thrust::max(NewTimeStep, vMinTimeStepSize);

	return NewTimeStep;
}
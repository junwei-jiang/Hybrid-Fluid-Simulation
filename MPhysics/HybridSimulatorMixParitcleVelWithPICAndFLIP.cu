#include "HybridSimulatorKernel.cuh"
#include "Particle.h"
#include "CudaContextManager.h"

__global__ void mixParitcleVelWithPICAndFLIP
(
	const Real* vPICVel,
	const Real* vFLIPVel,
	const Real* vDistP,
	UInt vParticleSize,
	Real vCellSpace,
	Real vGRate,
	Real vGFRate,
	Real vFRate,
	Real vPRate,

	Real* voPredictVel
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vParticleSize) return;
	Vector3 FLIPVel = ParticlePos(vFLIPVel, Index);
	Vector3 PICVel = ParticlePos(vPICVel, Index);
	Vector3 PredictVel = ParticlePos(voPredictVel, Index);
	Real Dist = vDistP[Index];

	Real Distg = vGRate * vCellSpace;
	Real Distgf = vGFRate * vCellSpace;
	Real Distf = vFRate * vCellSpace;
	Real Distp = vPRate * vCellSpace;

	Vector3 MixedVel;
	if (Dist <= Distg)
	{
		MixedVel = PICVel;
	}
	else if (Dist > Distg && Dist <= Distgf)
	{
		MixedVel = lerp(PICVel, PredictVel, (Dist - Distg) / (Distgf - Distg));
	}
	else if (Dist > Distgf && Dist <= Distf)
	{
		MixedVel = FLIPVel;
	}
	else if (Dist > Distf && Dist <= Distp)
	{
		MixedVel = lerp(FLIPVel, PredictVel, (Dist - Distf) / (Distp - Distf));
	}
	else
	{
		MixedVel = PredictVel;
	}

	ParticlePosPortion(voPredictVel, Index, 0) = MixedVel.x;
	ParticlePosPortion(voPredictVel, Index, 1) = MixedVel.y;
	ParticlePosPortion(voPredictVel, Index, 2) = MixedVel.z;
}

void mixParitcleVelWithPICAndFLIPInvoker
(
	const Real* vPICVel,
	const Real* vFLIPVel,
	const Real* vDistP,
	UInt vParticleSize,
	Real vCellSpace,
	Real vGRate,
	Real vGFRate,
	Real vFRate,
	Real vPRate,

	Real* voPredictVel
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);
	mixParitcleVelWithPICAndFLIP LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vPICVel,
		vFLIPVel,
		vDistP,
		vParticleSize,
		vCellSpace,
		vGRate,
		vGFRate,
		vFRate,
		vPRate,
		voPredictVel
	);
}
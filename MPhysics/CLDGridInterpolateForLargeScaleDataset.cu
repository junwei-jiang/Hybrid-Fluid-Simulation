#include "CLDGridKernel.cuh"
#include "CudaContextManager.h"
#include "CLDGridinterpolate.cuh"
#include "Particle.h"

__global__ void interpolateLargeDataSet
(
	const Real* vX,
	UInt vXCount,

	const Real* vNode,
	const UInt* vCell,
	SCLDGridInfo vCLDGridInfo,

	Real* voInterpolateResult,
	Real* voInterpolateGrad,
	Vector3 vGridTransform,
	SMatrix3x3 vGridRotation
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vXCount) return;

	Vector3 InputX = ParticlePos(vX, Index);

	Vector3 GridObjectSapceX = vGridRotation * (InputX - vGridTransform);

	Real Value;
	Vector3 Grad;
	interpolateSinglePos(GridObjectSapceX, vNode, vCell, vCLDGridInfo, Value, voInterpolateGrad == nullptr ? nullptr : &Grad);
	Real Temp = Value;
	Vector3 TempGrad = Grad;
	if (voInterpolateGrad)
	{
		ParticlePosPortion(voInterpolateGrad, Index, 0) = Grad.x;
		ParticlePosPortion(voInterpolateGrad, Index, 1) = Grad.y;
		ParticlePosPortion(voInterpolateGrad, Index, 2) = Grad.z;
	}

	voInterpolateResult[Index] = Value;
}

void interpolateInvoker
(
	const Real* vX,
	UInt vXCount,

	const Real* vNode,
	const UInt* vCell,
	SCLDGridInfo vCLDGridInfo,

	Real* voInterpolateResult,
	Real* voInterpolateGrad,
	Vector3 vGridTransform,
	SMatrix3x3 vGridRotation
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vXCount, BlockSize, GridSize, 0.25);

	interpolateLargeDataSet LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vX, 
		vXCount, 
		vNode, 
		vCell, 
		vCLDGridInfo, 
		voInterpolateResult, 
		voInterpolateGrad,
		vGridTransform,
		vGridRotation
	);
	CUDA_SYN
}
#include "DensityConstraintKernel.cuh"
#include "ThrustWapper.cuh"

#include "SimulationConfigManager.h"
#include "CudaContextManager.h"
#include "Particle.h"

__global__ void solveLinerSystem
(
	const Real* vVectorb,
	const UInt* vNeighborCount,
	UInt vConstraintSize,
	Real vTimeStep2,
	Real vStiffness,
	Real vMass,

	Real* voX
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vConstraintSize) return;

	UInt NeighborCount = vNeighborCount[Index];
	Real Factor = vTimeStep2 * vStiffness * static_cast<Real>(NeighborCount + 1) + vMass + EPSILON;
	voX[Index * 3 + 0] = vVectorb[Index * 3 + 0] / Factor;
	voX[Index * 3 + 1] = vVectorb[Index * 3 + 1] / Factor;
	voX[Index * 3 + 2] = vVectorb[Index * 3 + 2] / Factor;
}

void solveLinerInvoker
(
	const thrust::device_vector<Real>& vVectorb,
	const thrust::device_vector<UInt>& vNeighborCount,
	Real vTimeStep2,
	Real vStiffness,
	Real vMass,

	thrust::device_vector<Real>& voX
)
{
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vNeighborCount.size(), BlockSize, GridSize);
	solveLinerSystem LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		raw_pointer_cast(vVectorb.data()),
		raw_pointer_cast(vNeighborCount.data()),
		vNeighborCount.size(),
		vTimeStep2,
		vStiffness,
		vMass,
		raw_pointer_cast(voX.data())
	);

}

__global__ void generateA
(
	const Real* vPos,
	const UInt* vNeighborData,
	const UInt* vNeighborCount,
	const UInt* vNeighborOffset,
	Real vStiffness,
	Real vDeltaTime,
	UInt vConstraintSize
)
{

}
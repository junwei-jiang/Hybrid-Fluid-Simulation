#include "HybridSimulatorKernel.cuh"

__global__ void deleteParticleUnderWater
(
	const Real* vDistPC,
	const Real* vParticlePos,
	Real* vioRhoGridG,

	/*上述网格的属性*/
	Vector3i vGridResolution,
	Vector3 vGridMin,
	UInt vGridDataSize,
	Real vCellSpace,

	UInt vParticleSize,
	Real vCellSize,
	Real vDeleteRate,
	Real vTimeStep,
	Real vMaxLiveTime,

	Real* vioLiveTime,
	bool* voShouldDelete
)
{
	Int Index = blockIdx.x * blockDim.x + threadIdx.x;
	if (Index >= vParticleSize) return;

	Real DistPC = vDistPC[Index];
	Real LiveTime = ParticlePosPortion(vioLiveTime, Index, 0);
	if (DistPC <= 0)
		LiveTime += vTimeStep;

	if (LiveTime >= vMaxLiveTime)
	{
		ParticlePosPortion(voShouldDelete, Index, 0) = true;
		ParticlePosPortion(voShouldDelete, Index, 1) = true;
		ParticlePosPortion(voShouldDelete, Index, 2) = true;

		Vector3 ParticlePos = ParticlePos(vParticlePos, Index);
		Real RhoGX[8];
		Real RhoGNeighbor[8];
		getCCSFieldTriLinearWeightAndValue
		(
			vGridResolution,
			vGridMin,
			Vector3(vCellSpace, vCellSpace, vCellSpace),
			ParticlePos,
			vioRhoGridG,
			RhoGX,
			RhoGNeighbor
		);

		Vector3 RelPos = (ParticlePos - vGridMin - vCellSpace * 0.5);
		Vector3 RelPosIndex = RelPos / vCellSpace;
		Vector3ui DownBackLeftIndex = castToVector3ui(floorM(RelPosIndex));
		for (UInt z = 0; z <= 1; z++)
		{
			for (UInt y = 0; y <= 1; y++)
			{
				for (UInt x = 0; x <= 1; x++)
				{
					UInt LinerIndex = convert3DIndexToLiner((DownBackLeftIndex + Vector3ui(x, y, z)), vGridResolution);
					vioRhoGridG[LinerIndex] += RhoGX[z * 2 * 2 + y * 2 + x];
				}
			}
		}
	}
	else
	{
		ParticlePosPortion(vioLiveTime, Index, 0) = LiveTime;
		ParticlePosPortion(vioLiveTime, Index, 1) = LiveTime;
		ParticlePosPortion(vioLiveTime, Index, 2) = LiveTime;
	}
}

void deleteParticleUnderWater
(
	const Real* vDistPC,
	const Real* vParticlePos,
	Real* vioRhoGridG,

	/*上述网格的属性*/
	Vector3i vGridResolution,
	Vector3 vGridMin,
	UInt vGridDataSize,
	Real vCellSpace,

	UInt vParticleSize,
	Real vCellSize,
	Real vDeleteRate,
	Real vTimeStep,
	Real vMaxLiveTime,

	thrust::device_vector<bool>& voShouldDelete,
	std::shared_ptr<CParticleGroup> voTarget
)
{
	voShouldDelete.resize(voTarget->getSize() * 3);
	thrust::fill(voShouldDelete.begin(), voShouldDelete.end(), false);

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);

	deleteParticleUnderWater LANCH_PARAM_1D_GB(GridSize, BlockSize)
	(
		vDistPC,
		vParticlePos,
		vioRhoGridG,

		vGridResolution,
		vGridMin,
		vGridDataSize,
		vCellSpace,

		vParticleSize,
		vCellSize,
		vDeleteRate,
		vTimeStep,
		vMaxLiveTime,
		voTarget->getParticleLiveTimeGPUPtr(),
		thrust::raw_pointer_cast(voShouldDelete.data())
	);

	voTarget->removeParticles(voShouldDelete);
}
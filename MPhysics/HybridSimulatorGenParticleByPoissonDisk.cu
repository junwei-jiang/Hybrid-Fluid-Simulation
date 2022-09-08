#include "HybridSimulatorKernel.cuh"
#include "Particle.h"
#include "KNNKernel.cuh"
#include "FieldMathTool.cuh"
#include "EulerMathTool.cuh"
#include <curand_kernel.h>

__global__ void genParticleByPoissonDisk
(
	const UInt* vNeedCellIndex, //已经与流体粒子密度混合的网格的Dist
	UInt vNeedCellIndexCount, //已经与流体粒子密度混合的网格的Dist

	const Real* vRhoGridC, //已经与流体粒子密度混合的网格的Rho
	Real* vioRhoGridG, //未与流体粒子密度混合的网格的Rho

	/*上述网格的属性*/
	Vector3i vGridResolution,
	Vector3 vGridMin,
	UInt vGridDataSize,
	Real vCellSpace,

	/*SPH使用的空间Hash数据*/
	const Real* vSortParticlePos,
	const UInt* vCellOffset,
	const UInt* vCellParticleCounts,
	SGridInfo vHashGridInfo,
	UInt vParticleSize,

	/*混合模拟器属性*/
	UInt vSampleK,
	Real vCreateDistRate,
	Real vParticleVolume,
	Real vRhoMin,

	Real* voNewParticlePos,
	Real* voNewPrevParticlePos,
	Real* voNewParticleVel,
	Real* voNewParticleLiveTime
)
{
	UInt Index = blockIdx.x * blockDim.x + threadIdx.x;
	UInt InsertIndex = Index / vSampleK;
	if (InsertIndex >= vNeedCellIndexCount) return;

	UInt TargetCellIndex = vNeedCellIndex[InsertIndex];
	if (TargetCellIndex >= vGridDataSize) return;

	Vector3ui CurrCell3DIndex;
	CurrCell3DIndex.z = TargetCellIndex / (vGridResolution.x * vGridResolution.y);
	UInt ZBlock = CurrCell3DIndex.z * (vGridResolution.x * vGridResolution.y);
	CurrCell3DIndex.y = (TargetCellIndex - ZBlock) / vGridResolution.x;
	UInt YBlock = CurrCell3DIndex.y * vGridResolution.x;
	CurrCell3DIndex.x = TargetCellIndex - ZBlock - YBlock;

	Vector3 CellMin = vGridMin + vCellSpace * castToVector3(CurrCell3DIndex);

	curandState S;
	curand_init(Index + clock(), 0, 0, &S);
	Vector3 Pos = CellMin + Vector3(vCellSpace * curand_uniform(&S), vCellSpace * curand_uniform(&S), vCellSpace * curand_uniform(&S));

	Real RhoCX = sampleOneCellInCCSFieldTrilerp
	(
		vRhoGridC,
		vGridResolution,
		Vector3(vCellSpace, vCellSpace, vCellSpace),
		vGridMin,
		Pos
	);

	/*条件1：RhoCX大于RhoMin*/
	if (RhoCX < vRhoMin) return;

	/*条件2：vSupportRadius范围内没有其他粒子*/
	//Vector3 HashGridCellPos = (Pos - vHashGridInfo.AABB.Min) * vHashGridInfo.GridDelta;
	//Vector3ui HashGridCellIndex = Vector3ui(HashGridCellPos.x, HashGridCellPos.y, HashGridCellPos.z);

	/*if (HashGridCellIndex.x >= 0 && HashGridCellIndex.y >= 0 && HashGridCellIndex.z >= 0 &&
		HashGridCellIndex.x < vHashGridInfo.GridDimension.x &&
		HashGridCellIndex.y < vHashGridInfo.GridDimension.y &&
		HashGridCellIndex.z < vHashGridInfo.GridDimension.z)
	{
		for (Int i = -1; i <= 1; i++)
		{
			for (Int k = -1; k <= 1; k++)
			{
				for (Int j = -1; j <= 1; j++)
				{
					Vector3i NeighborHashCell3DIndex = castToVector3i(HashGridCellIndex) + Vector3i(i, k, j);

					if (NeighborHashCell3DIndex.x < 0 || NeighborHashCell3DIndex.y < 0 || NeighborHashCell3DIndex.z < 0 ||
						NeighborHashCell3DIndex.x >= vHashGridInfo.GridDimension.x ||
						NeighborHashCell3DIndex.y >= vHashGridInfo.GridDimension.y ||
						NeighborHashCell3DIndex.z >= vHashGridInfo.GridDimension.z)
						continue;

					Vector3ui FinalCoord = castToVector3ui(NeighborHashCell3DIndex);
					UInt NeighborCellIndex = toCellIndexOfMortonMetaGrid(vHashGridInfo, FinalCoord);
					if (NeighborCellIndex >= vHashGridInfo.CellCount) continue;

					UInt NeighborCellCount = vCellParticleCounts[NeighborCellIndex];
					UInt NeighborCellStart = vCellOffset[NeighborCellIndex];

					if (NeighborCellStart + NeighborCellCount >= vParticleSize) continue;

					for (UInt w = NeighborCellStart; w < NeighborCellStart + NeighborCellCount; w++)
					{
						Vector3 Dis = ParticlePos(vSortParticlePos, w) - Pos;
						Real Distance = length(Dis);
						if (Distance < vHashGridInfo.SearchRadius && abs(Distance - 0.0) > EPSILON)
						{
							return;
						}
					}
				}
			}
		}
	}*/

	/*条件3：RhoGX大于所有相邻的网格G的格点密度*/
	Real RhoGX[8];
	Real RhoGNeighbor[8];
	getCCSFieldTriLinearWeightAndValue
	(
		vGridResolution,
		vGridMin,
		Vector3(vCellSpace, vCellSpace, vCellSpace),
		Pos,
		vioRhoGridG,
		RhoGX,
		RhoGNeighbor
	);
#pragma unroll
	for (UInt i = 0; i < 8; i++)
	{
		RhoGX[i] *= (vParticleVolume / (vCellSpace * vCellSpace * vCellSpace));

		if (RhoGX[i] > RhoGNeighbor[i])
		{
			return;
		}
	}

	/*若满足上述条件，则生成一个新粒子，并将RhoGX从所有相邻的网格G的格点密度减去*/
	UInt Offset = vParticleSize + Index;
	ParticlePosPortion(voNewParticlePos, Offset, 0) = Pos.x;
	ParticlePosPortion(voNewParticlePos, Offset, 1) = Pos.y;
	ParticlePosPortion(voNewParticlePos, Offset, 2) = Pos.z;

	ParticlePosPortion(voNewPrevParticlePos, Offset, 0) = Pos.x;
	ParticlePosPortion(voNewPrevParticlePos, Offset, 1) = Pos.y;
	ParticlePosPortion(voNewPrevParticlePos, Offset, 2) = Pos.z;

	ParticlePosPortion(voNewParticleVel, Offset, 0) = 0;
	ParticlePosPortion(voNewParticleVel, Offset, 1) = 0;
	ParticlePosPortion(voNewParticleVel, Offset, 2) = 0;

	ParticlePosPortion(voNewParticleLiveTime, Offset, 0) = 0;
	ParticlePosPortion(voNewParticleLiveTime, Offset, 1) = 0;
	ParticlePosPortion(voNewParticleLiveTime, Offset, 2) = 0;

	Vector3 RelPos = (Pos - vGridMin - vCellSpace * 0.5);
	Vector3 RelPosIndex = RelPos / vCellSpace;
	Vector3ui DownBackLeftIndex = castToVector3ui(floorM(RelPosIndex));
	for (UInt z = 0; z <= 1; z++)
	{
		for (UInt y = 0; y <= 1; y++)
		{
			for (UInt x = 0; x <= 1; x++)
			{
				UInt LinerIndex = convert3DIndexToLiner((DownBackLeftIndex + Vector3ui(x, y, z)), vGridResolution);
				vioRhoGridG[LinerIndex] -= RhoGX[z * 2 * 2 + y * 2 + x];
			}
		}
	}
}

__global__ void collectNeedGenNewParticleCell
(
	const Real* vDistGridC,
	const Real* vRhoGridG,

	Vector3i vGridResolution,
	Vector3 vGridMin,
	UInt vGridDataSize,
	Real vCellSpace,
	Real vCreateDistRate,

	UInt* voNeedGenNewParticleCellIndexCache
)
{
	Vector3i CurrCell3DIndex;
	CurrCell3DIndex.x = threadIdx.x;
	CurrCell3DIndex.y = blockIdx.x;
	CurrCell3DIndex.z = blockIdx.y;
	UInt Index = transIndex2LinearWithOffset(CurrCell3DIndex, vGridResolution);
	if (Index >= vGridDataSize) return;

	Real DistGridC = vDistGridC[Index];
	Real RhoGridG = vRhoGridG[Index];
	UInt InsertPos = 0;
	if ((RhoGridG <= 0) || DistGridC >= (vCellSpace * vCreateDistRate) || DistGridC <= 0)
	{
		voNeedGenNewParticleCellIndexCache[Index] = UINT_MAX;
	}
}

void genParticleByPoissonDiskInvoker
(
	const Real* vDistGridC,
	const Real* vRhoGridC,
	Real* vioRhoGridG,

	Vector3i vGridResolution,
	Vector3 vGridMin,
	UInt vGridDataSize,
	Real vCellSpace,

	const UInt* vCellOffset,
	const UInt* vCellParticleCounts,
	SGridInfo vHashGridInfo,

	UInt vSampleK,
	Real vCreateDistRate,
	Real vRhoMin,
	Real vFluidRestDensity,

	std::shared_ptr<CParticleGroup> voParticleGroup,
	thrust::device_vector<UInt>& voNeedGenNewParticleCellIndexCache
)
{
	thrust::sequence(voNeedGenNewParticleCellIndexCache.begin(), voNeedGenNewParticleCellIndexCache.end(), 0);

	//调用核函数添加新粒子
	dim3 ThreadPerBlock(vGridResolution.x);
	dim3 NumBlock(vGridResolution.y, vGridResolution.z);
	collectNeedGenNewParticleCell LANCH_PARAM_1D_GB(NumBlock, ThreadPerBlock)
	(
		vDistGridC,
		vioRhoGridG,
		vGridResolution,
		vGridMin,
		vGridDataSize,
		vCellSpace,
		vCreateDistRate,
		thrust::raw_pointer_cast(voNeedGenNewParticleCellIndexCache.data())
	);

	UInt NeedGenNewParticleCellCount = thrust::remove(voNeedGenNewParticleCellIndexCache.begin(), voNeedGenNewParticleCellIndexCache.end(), UINT_MAX) - voNeedGenNewParticleCellIndexCache.begin();
	//if (NeedGenNewParticleCellCount < vGridDataSize * 0.01) return;

	voParticleGroup->appendEmptyParticle(NeedGenNewParticleCellCount * vSampleK);

	UInt BlockSize;
	UInt GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(NeedGenNewParticleCellCount * vSampleK, BlockSize, GridSize);

	genParticleByPoissonDisk LANCH_PARAM_1D_GB(GridSize, BlockSize)
	(
		thrust::raw_pointer_cast(voNeedGenNewParticleCellIndexCache.data()),
		NeedGenNewParticleCellCount,

		vRhoGridC,
		vioRhoGridG,

		vGridResolution,
		vGridMin,
		vGridDataSize,
		vCellSpace,

		voParticleGroup->getConstParticlePosGPUPtr(),
		vCellOffset,
		vCellParticleCounts,
		vHashGridInfo,
		voParticleGroup->getSize(),

		vSampleK,
		vCreateDistRate,
		voParticleGroup->getParticleVolume(),
		vRhoMin,

		voParticleGroup->getParticlePosGPUPtr(),
		voParticleGroup->getPrevParticlePosGPUPtr(),
		voParticleGroup->getParticleVelGPUPtr(),
		voParticleGroup->getParticleLiveTimeGPUPtr()
	);

	voParticleGroup->reduceEmptyParticle();
}
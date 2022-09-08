#include "KNNKernel.cuh"
#include "ThrustWapper.cuh"

#include "CudaContextManager.h"
#include "GPUTimer.h"

__device__ int MinMaxGPU[6];

__host__ __device__ uint32_t expandBits(uint32_t v)
{
	v &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	v = (v ^ (v << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	v = (v ^ (v << 8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	v = (v ^ (v << 4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	v = (v ^ (v << 2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return v;
}

__host__ __device__ uint32_t toMorton3D(Vector3ui vGridCellIndex)
{
	return (expandBits(vGridCellIndex.z) << 2) + (expandBits(vGridCellIndex.y) << 1) + expandBits(vGridCellIndex.x);;
}

__host__ __device__ UInt convertCellIndicesToLinearIndex(Vector3ui cellDimensions, Vector3ui xyz)
{
	return xyz.z * cellDimensions.y * cellDimensions.x + xyz.y * cellDimensions.x + xyz.x;
}

__host__ __device__ UInt toCellIndexOfMortonMetaGrid(SGridInfo vGridInfo, Vector3ui vGridCellIndex)
{
	Vector3ui MetaGridCellIndex = Vector3ui(
		vGridCellIndex.x / vGridInfo.MetaGridGroupSize,
		vGridCellIndex.y / vGridInfo.MetaGridGroupSize,
		vGridCellIndex.z / vGridInfo.MetaGridGroupSize
	);
	Vector3ui InnerGridCellIndex = vGridCellIndex - MetaGridCellIndex * vGridInfo.MetaGridGroupSize;
	return convertCellIndicesToLinearIndex(vGridInfo.MetaGridDimension, MetaGridCellIndex)
		* vGridInfo.MetaGridBlockSize + toMorton3D(InnerGridCellIndex);
}

__global__ void ComputeMinMax
(
	const Real * vParticlePos,
	UInt vParticleSize,
	Real vSearchRadius
)
{
	unsigned int ParticleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ParticleIndex >= vParticleSize) return;
	Vector3 Particle = ParticlePos(vParticlePos, ParticleIndex);

	Vector3i NormalCell;
	NormalCell.x = (Int)floor(Particle.x / vSearchRadius);
	NormalCell.y = (Int)floor(Particle.y / vSearchRadius);
	NormalCell.z = (Int)floor(Particle.z / vSearchRadius);

	atomicMin(&(MinMaxGPU[0]), NormalCell.x);
	atomicMin(&(MinMaxGPU[1]), NormalCell.y);
	atomicMin(&(MinMaxGPU[2]), NormalCell.z);

	atomicMax(&(MinMaxGPU[3]), NormalCell.x);
	atomicMax(&(MinMaxGPU[4]), NormalCell.y);
	atomicMax(&(MinMaxGPU[5]), NormalCell.z);
}

__global__ void insertParticlesToMortonCell
(
	const Real *vParticlePosGPUPtr,
	UInt vParticleCount,
	SGridInfo vGridInfo,
	UInt *voParticleCellIndices,
	UInt *voCellParticleCounts,
	UInt *voSortIndices
)
{
	UInt ParticleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ParticleIndex >= vParticleCount) return;
	Vector3 ParticlePos = ParticlePos(vParticlePosGPUPtr, ParticleIndex);

	Vector3 GridCellPos = (ParticlePos - vGridInfo.AABB.Min) * vGridInfo.GridDelta;
	Vector3ui GridCellIndex = Vector3ui(GridCellPos.x, GridCellPos.y, GridCellPos.z);

	UInt CellLineIndex = toCellIndexOfMortonMetaGrid(vGridInfo, GridCellIndex);
	if (CellLineIndex >= vGridInfo.CellCount) return;

	voParticleCellIndices[ParticleIndex] = CellLineIndex;
	voSortIndices[ParticleIndex] = atomicAdd(&voCellParticleCounts[CellLineIndex], 1);
}

__global__ void countingSortIndices(
	const SGridInfo vGridInfo,

	const UInt *vParticleCellIndices,
	const UInt *vCellOffsets,
	const UInt *vSortIndicesSrc,
	UInt vParticleCount,

	UInt * voSortIndicesDest
)
{
	UInt ParticleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ParticleIndex >= vParticleCount) return;

	UInt GridCellIndex = vParticleCellIndices[ParticleIndex];

	UInt SortIndex = vSortIndicesSrc[ParticleIndex] + vCellOffsets[GridCellIndex];
	if (SortIndex >= vParticleCount) return;

	voSortIndicesDest[SortIndex * 3 + 0] = ParticleIndex * 3 + 0;
	voSortIndicesDest[SortIndex * 3 + 1] = ParticleIndex * 3 + 1;
	voSortIndicesDest[SortIndex * 3 + 2] = ParticleIndex * 3 + 2;
}

__global__ void queryNeighborCount
(
	const Real *vSortParticlePos,
	UInt vParticleCount,

	SGridInfo vGridInfo,
	const UInt *vCellOffsets,
	const UInt *vCellParticleCounts,

	UInt *voNeighborCounts
)
{
	UInt ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ThreadIndex >= vParticleCount) return;

	Vector3 QueryPaticlePos = ParticlePos(vSortParticlePos, ThreadIndex);
	Vector3 CoordInt = (QueryPaticlePos - vGridInfo.AABB.Min) * (vGridInfo.GridDelta);
	Vector3ui Coord = Vector3ui(CoordInt.x, CoordInt.y, CoordInt.z);
	if (Coord.x >= vGridInfo.GridDimension.x) Coord.x = vGridInfo.GridDimension.x - 1;
	if (Coord.y >= vGridInfo.GridDimension.y) Coord.y = vGridInfo.GridDimension.y - 1;
	if (Coord.z >= vGridInfo.GridDimension.z) Coord.z = vGridInfo.GridDimension.z - 1;

	UInt NeighborCount = 0;

	for (Int z = -1; z < 2; z++)
	{
		for (Int y = -1; y < 2; y++)
		{
			for (Int x = -1; x < 2; x++)
			{
				Vector3i TempCoord = Vector3i(
					(Int)Coord.x + x,
					(Int)Coord.y + y,
					(Int)Coord.z + z
				);

				if (TempCoord.x < 0 || TempCoord.y < 0 || TempCoord.z < 0 ||
					TempCoord.x >= vGridInfo.GridDimension.x ||
					TempCoord.y >= vGridInfo.GridDimension.y ||
					TempCoord.z >= vGridInfo.GridDimension.z)
					continue;

				Vector3ui FinalCoord = castToVector3ui(TempCoord);
				UInt NeighborCellIndex = toCellIndexOfMortonMetaGrid(vGridInfo, FinalCoord);
				UInt NeighborCellCount = vCellParticleCounts[NeighborCellIndex];
				UInt NeighborCellStart = vCellOffsets[NeighborCellIndex];

				for (UInt i = NeighborCellStart; i < NeighborCellStart + NeighborCellCount; i++)
				{
					Vector3 Dis = ParticlePos(vSortParticlePos, i) - QueryPaticlePos;
					Real Distance = dot(Dis, Dis);
					if (Distance < pow(vGridInfo.SearchRadius, 2) && abs(Distance - 0.0) > EPSILON)
					{
						NeighborCount++;
						if (NeighborCount == MAXNeighborCount)
						{
							voNeighborCounts[ThreadIndex] = NeighborCount;
							return;
						}
					}
				}
			}
		}
	}
	voNeighborCounts[ThreadIndex] = NeighborCount;
}

__global__ void queryNeighbors
(
	const Real *vSortParticlePos,
	UInt vParticleCount,

	SGridInfo vGridInfo,
	const UInt *vCellOffsets,
	const UInt *vCellParticleCounts,

	const UInt *vNeighborOffset,
	UInt *voNeighbors
)
{
	UInt ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ThreadIndex >= vParticleCount) return;

	Vector3 QueryPaticlePos = ParticlePos(vSortParticlePos, ThreadIndex);
	Vector3 CoordInt = (QueryPaticlePos - vGridInfo.AABB.Min) * (vGridInfo.GridDelta);
	Vector3ui Coord = Vector3ui(CoordInt.x, CoordInt.y, CoordInt.z);
	if (Coord.x >= vGridInfo.GridDimension.x) Coord.x = vGridInfo.GridDimension.x - 1;
	if (Coord.y >= vGridInfo.GridDimension.y) Coord.y = vGridInfo.GridDimension.y - 1;
	if (Coord.z >= vGridInfo.GridDimension.z) Coord.z = vGridInfo.GridDimension.z - 1;

	UInt NeighborOffset = vNeighborOffset[ThreadIndex];
	UInt Count = 0;
	for (Int z = -1; z < 2; z++)
	{
		for (Int y = -1; y < 2; y++)
		{
			for (Int x = -1; x < 2; x++)
			{
				Vector3i TempCoord = Vector3i(
					(Int)Coord.x + x,
					(Int)Coord.y + y,
					(Int)Coord.z + z
				);

				if (TempCoord.x < 0 || TempCoord.y < 0 || TempCoord.z < 0 ||
					TempCoord.x >= vGridInfo.GridDimension.x ||
					TempCoord.y >= vGridInfo.GridDimension.y ||
					TempCoord.z >= vGridInfo.GridDimension.z)
					continue;

				Vector3ui FinalCoord = castToVector3ui(TempCoord);
				UInt NeighborCellIndex = toCellIndexOfMortonMetaGrid(vGridInfo, FinalCoord);
				UInt NeighborCellCount = vCellParticleCounts[NeighborCellIndex];
				UInt NeighborCellStart = vCellOffsets[NeighborCellIndex];

				for (UInt i = NeighborCellStart; i < NeighborCellStart + NeighborCellCount; i++)
				{
					Vector3 Dis = ParticlePos(vSortParticlePos, i) - QueryPaticlePos;
					Real Distance = dot(Dis, Dis);
					if (Distance < pow(vGridInfo.SearchRadius, 2) && abs(Distance - 0.0) > EPSILON)
					{
						voNeighbors[NeighborOffset + Count] = i;
						Count++;
						if (Count == MAXNeighborCount) return;
					}
				}
			}
		}
	}
}

void computeMinMax(const Real* vParticlePosGPUPtr, UInt vParticleSize, Real vRadius, SAABB& voAABB)
{
	_ASSERT(vParticleSize != 0);

	int MinMaxCPU[6] = { INT_MAX,INT_MAX,INT_MAX,INT_MIN,INT_MIN,INT_MIN };
	CHECK_CUDA(cudaMemcpyToSymbol(MinMaxGPU, MinMaxCPU, sizeof(MinMaxCPU)));

	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);

	ComputeMinMax LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vParticlePosGPUPtr,
		vParticleSize,
		vRadius
	);
	CHECK_CUDA(cudaMemcpyFromSymbol(MinMaxCPU, MinMaxGPU, sizeof(MinMaxCPU)));

	voAABB.Min = Vector3(MinMaxCPU[0], MinMaxCPU[1], MinMaxCPU[2]) * vRadius;
	voAABB.Max = Vector3(MinMaxCPU[3], MinMaxCPU[4], MinMaxCPU[5]) * vRadius;
}

void computeCellInformation
(
	const Real* vParticlePos,
	UInt* voParticleCellIndices,
	UInt* voTempSortIndices,
	UInt* voSortIndices,
	UInt vParticleSize,

	UInt* voCellParticleCounts,
	UInt* voCellOffsets,
	SGridInfo vGridInfo
)
{
	//为每一个粒子计算其对应的Cell的索引，并记录每一个Cell中的粒子的个数。
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);

	insertParticlesToMortonCell LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vParticlePos,
		vParticleSize, 
		vGridInfo, 
		voParticleCellIndices,
		voCellParticleCounts,
		voTempSortIndices
	);
	CUDA_SYN

	//前缀和求出各个Cell中对应粒子的offset。
	thrust::exclusive_scan(thrust::device, voCellParticleCounts, voCellParticleCounts + vGridInfo.CellCount, voCellOffsets);
	CUDA_SYN

	//Counting Sort将粒子按Cell索引排序。
	countingSortIndices LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vGridInfo,
		voParticleCellIndices,
		voCellOffsets,
		voTempSortIndices,
		vParticleSize,
		voSortIndices
	);
	CUDA_SYN
}

void searchNeighbors
(
	const Real* vSortParticlePos,
	UInt vParticleSize,

	const UInt* vCellParticleCounts,
	const UInt* vCellOffsets,
	SGridInfo vGridInfo,

	thrust::device_vector<UInt>& voNeighborOffset,
	thrust::device_vector<UInt>& voNeighborCounts,
	thrust::device_vector<UInt>& voNeighbors
)
{
	//KNN搜索
	UInt BlockSize, GridSize;
	CCudaContextManager::getInstance().fetchPropBlockGridSize1D(vParticleSize, BlockSize, GridSize);

	queryNeighborCount LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vSortParticlePos,
		vParticleSize,

		vGridInfo,
		vCellOffsets,
		vCellParticleCounts,

		raw_pointer_cast(voNeighborCounts.data())
	);
	CUDA_SYN
	thrust::exclusive_scan(thrust::device, voNeighborCounts.begin(), voNeighborCounts.end(), voNeighborOffset.begin());
	UInt LastNeighborCount = voNeighborCounts.back();
	UInt LastOffset = voNeighborOffset.back();
	voNeighbors.resize(LastNeighborCount + LastOffset);

	queryNeighbors LANCH_PARAM_1D_GBS(GridSize, BlockSize, 0)
	(
		vSortParticlePos,
		vParticleSize,

		vGridInfo,
		vCellOffsets,
		vCellParticleCounts,

		raw_pointer_cast(voNeighborOffset.data()),
		raw_pointer_cast(voNeighbors.data())
	);
	CUDA_SYN
}

void zsortParticleGroup
(
	thrust::device_vector<Real> & voPos,
	thrust::device_vector<Real> & voVel,
	thrust::device_vector<Real> & voPrevPos,
	thrust::device_vector<Real> & voLiveTime,
	thrust::device_vector<Real> & voSortCache,
	const thrust::device_vector<UInt>& vSortIndeics
)
{
	thrust::gather(vSortIndeics.begin(), vSortIndeics.end(), voPos.begin(), voSortCache.begin());
	voPos = voSortCache;

	thrust::gather(vSortIndeics.begin(), vSortIndeics.end(), voVel.begin(), voSortCache.begin());
	voVel = voSortCache;

	thrust::gather(vSortIndeics.begin(), vSortIndeics.end(), voPrevPos.begin(), voSortCache.begin());
	voPrevPos = voSortCache;

	thrust::gather(vSortIndeics.begin(), vSortIndeics.end(), voLiveTime.begin(), voSortCache.begin());
	voLiveTime = voSortCache;
}

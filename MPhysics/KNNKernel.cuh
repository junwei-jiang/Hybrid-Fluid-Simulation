#pragma once
#include "KNNSearch.h"

__host__ __device__ uint32_t expandBits(uint32_t v);

__host__ __device__ uint32_t toMorton3D(Vector3ui vGridCellIndex);

__host__ __device__ UInt convertCellIndicesToLinearIndex(Vector3ui cellDimensions, Vector3ui xyz);

__host__ __device__ UInt toCellIndexOfMortonMetaGrid(SGridInfo vGridInfo, Vector3ui vGridCellIndex);

void computeMinMax(const Real* vParticlePosGPUPtr, UInt vParticleSize, Real vRadius, SAABB& voAABB);

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
);

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
);

void zsortParticleGroup
(
	thrust::device_vector<Real> & voPos,
	thrust::device_vector<Real> & voVel,
	thrust::device_vector<Real> & voPrevPos,
	thrust::device_vector<Real> & voLiveTime,
	thrust::device_vector<Real> & voSortCache,
	const thrust::device_vector<UInt>& vSortIndeics
);
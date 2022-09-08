#include "KNNSearch.h"
#include "SimulationConfigManager.h"

#include "KNNKernel.cuh"
#include "ThrustWapper.cuh"
#include "GPUTimer.h"

CKNNSearch::CKNNSearch()
{}

CKNNSearch::~CKNNSearch()
{}

void CKNNSearch::search()
{
	__updateGrid();

	computeCellInformation
	(
		m_TargetParticleGroup->getConstParticlePosGPUPtr(),
		getRawDevicePointerUInt(m_ParticleCellIndices),
		getRawDevicePointerUInt(m_TempSortIndices),
		getRawDevicePointerUInt(m_SortIndices),
		m_TargetParticleGroup->getSize(),
		getRawDevicePointerUInt(m_CellParticleCounts),
		getRawDevicePointerUInt(m_CellParticleOffsets),
		m_GridInfo
	);

	zsortParticleGroup
	(
		m_TargetParticleGroup->getParticlePos().getVectorValue(),
		m_TargetParticleGroup->getParticleVel().getVectorValue(),
		m_TargetParticleGroup->getPrevParticlePos().getVectorValue(),
		m_TargetParticleGroup->getLiveTime(),
		m_SortCache,
		m_SortIndices
	);

	searchNeighbors
	(
		m_TargetParticleGroup->getConstParticlePosGPUPtr(),
		m_TargetParticleGroup->getSize(),
		getReadOnlyRawDevicePointer(m_CellParticleCounts),
		getReadOnlyRawDevicePointer(m_CellParticleOffsets),
		m_GridInfo,
		m_NeighborOffsets,
		m_NeighborCounts,
		m_Neighbors
	);
}

void CKNNSearch::bindParticleGroup(shared_ptr<CParticleGroup> vioTarget)
{
	_ASSERT(vioTarget != nullptr);

	m_TargetParticleGroup = vioTarget;
	UInt ParticleCount = m_TargetParticleGroup->getSize();
	m_SearchRadius = m_TargetParticleGroup->getParticleSupportRadius();
	m_CellSize = m_TargetParticleGroup->getParticleSupportRadius();

	resizeDeviceVector(m_ParticleCellIndices, ParticleCount);
	resizeDeviceVector(m_SortIndices, ParticleCount * 3);
	resizeDeviceVector(m_SortCache, ParticleCount * 3);
	resizeDeviceVector(m_TempSortIndices, ParticleCount);
	resizeDeviceVector(m_NeighborCounts, ParticleCount);
	resizeDeviceVector(m_NeighborOffsets, ParticleCount);
}

void CKNNSearch::getNeighborsDebug(map<UInt, unordered_set<UInt>>& voNeighbors)
{
	for (int i = 0; i < m_TargetParticleGroup->getSize(); i++)
	{
		unordered_set<UInt> Neighbors;
		UInt NeighborCount = getElementUInt(m_NeighborCounts, i);
		UInt NeighborOffset = getElementUInt(m_NeighborOffsets, i);
		for (int k = NeighborOffset; k < NeighborOffset + NeighborCount; k++)
		{
			Neighbors.insert(getElementUInt(m_Neighbors, k));
		}
		voNeighbors[i] = Neighbors;
	}
}

const thrust::device_vector<UInt>& CKNNSearch::getNeighorData() const
{
	return m_Neighbors;
}

const thrust::device_vector<UInt>& CKNNSearch::getNeighborCounts() const
{
	return m_NeighborCounts;
}

const thrust::device_vector<UInt>& CKNNSearch::getNeighorOffsets() const
{
	return m_NeighborOffsets;
}

const thrust::device_vector<UInt>& CKNNSearch::getCellParticleCounts() const
{
	return m_CellParticleCounts;
}

const thrust::device_vector<UInt>& CKNNSearch::getCellParticleOffsets() const
{
	return m_CellParticleOffsets;
}

UInt CKNNSearch::getCellMortonLinerIndex(Vector3 vPos) const
{
	Vector3 GridCellPos = (vPos - m_GridInfo.AABB.Min) * m_GridInfo.GridDelta;
	Vector3ui GridCellIndex = Vector3ui(GridCellPos.x, GridCellPos.y, GridCellPos.z);
	return toCellIndexOfMortonMetaGrid(m_GridInfo, GridCellIndex);
}

UInt CKNNSearch::getCellMortonLinerIndex(Vector3ui vIndex3D) const
{
	return toCellIndexOfMortonMetaGrid(m_GridInfo, vIndex3D);
}

void CKNNSearch::setMetaGridGroupSize(UInt vInput)
{
	m_MetaGridGroupSize = vInput;
}

void CKNNSearch::setSearchRadius(Real vInput)
{
	m_SearchRadius = vInput;
}

void CKNNSearch::setCellSize(Real vInput)
{
	_ASSERT(vInput >= m_SearchRadius);
	m_CellSize = vInput;
}

UInt CKNNSearch::getNeighorDataSize() const
{
	return getDeviceVectorSize(m_Neighbors);
}

UInt CKNNSearch::getMetaGridGroupSize() const
{
	return m_MetaGridGroupSize;
}

UInt CKNNSearch::getMetaGridBlockSize() const
{
	return m_MetaGridGroupSize * m_MetaGridGroupSize * m_MetaGridGroupSize;
}

Real CKNNSearch::getSearchRadius() const
{
	return m_SearchRadius;
}

Real CKNNSearch::getCellSize() const
{
	return m_CellSize;
}

UInt CKNNSearch::getNeighborCount(UInt vIndex) const
{
	return getElementUInt(m_NeighborCounts, vIndex);
}

Vector3 CKNNSearch::getGridMin() const
{
	return m_GridInfo.AABB.Min;
}

Vector3 CKNNSearch::getGridMax() const
{
	return m_GridInfo.AABB.Max;
}

SGridInfo CKNNSearch::getGridInfo() const
{
	return m_GridInfo;
}

void CKNNSearch::__updateGrid()
{
	computeMinMax
	(
		m_TargetParticleGroup->getConstParticlePosGPUPtr(),
		m_TargetParticleGroup->getSize(),
		m_SearchRadius,
		m_GridInfo.AABB
	);
	CSimulationConfigManager::getInstance().getSimualtionRange().lerp(m_GridInfo.AABB);

	Real SearchRadius = m_SearchRadius;
	Real GridSize = m_CellSize;

	m_GridInfo.SearchRadius = SearchRadius;
	m_GridInfo.AABB.Min -= Vector3(GridSize, GridSize, GridSize) * 2.0;
	m_GridInfo.AABB.Max += Vector3(GridSize, GridSize, GridSize) * 2.0;

	m_GridInfo.GridDimension = castToVector3ui((m_GridInfo.AABB.Max - m_GridInfo.AABB.Min) / GridSize);

	m_GridInfo.MetaGridBlockSize = m_MetaGridGroupSize * m_MetaGridGroupSize * m_MetaGridGroupSize;
	m_GridInfo.MetaGridGroupSize = m_MetaGridGroupSize;
	m_GridInfo.MetaGridDimension = ceilToVector3ui(castToVector3(m_GridInfo.GridDimension) / (Real)m_GridInfo.MetaGridGroupSize);

	Vector3 GridLength = castToVector3(m_GridInfo.GridDimension) * GridSize;
	m_GridInfo.GridDelta = castToVector3(m_GridInfo.GridDimension) / GridLength;

	m_GridInfo.CellCount =
		m_GridInfo.MetaGridDimension.x *
		m_GridInfo.MetaGridDimension.y *
		m_GridInfo.MetaGridDimension.z *
		m_GridInfo.MetaGridBlockSize;

	m_GridInfo.GridDimension = m_GridInfo.MetaGridDimension * m_GridInfo.MetaGridGroupSize;
	m_GridInfo.AABB.Max = castToVector3(m_GridInfo.GridDimension) * m_CellSize;

	resizeDeviceVector(m_CellParticleOffsets, m_GridInfo.CellCount);
	resizeDeviceVector(m_CellParticleCounts, m_GridInfo.CellCount);

	CHECK_CUDA(cudaMemset(getRawDevicePointerUInt(m_CellParticleOffsets), 0, m_GridInfo.CellCount * sizeof(UInt)));
	CHECK_CUDA(cudaMemset(getRawDevicePointerUInt(m_CellParticleCounts), 0, m_GridInfo.CellCount * sizeof(UInt)));

	CHECK_CUDA(cudaDeviceSynchronize());
}

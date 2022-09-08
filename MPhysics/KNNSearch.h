#pragma once
#include "Common.h"
#include "Particle.h"

struct SGridInfo
{
	SAABB AABB;
	Vector3 GridDelta;
	Vector3ui GridDimension;
	Vector3ui MetaGridDimension;
	Real SearchRadius;
	UInt MetaGridGroupSize;
	UInt MetaGridBlockSize;
	UInt CellCount;
};

class CKNNSearch
{
public:
	CKNNSearch();
	~CKNNSearch();

	void search();
	void bindParticleGroup(shared_ptr<CParticleGroup> vioTarget);
	void getNeighborsDebug(map<UInt, unordered_set<UInt>>& voNeighbors);

	const thrust::device_vector<UInt>& getNeighorData() const;
	const thrust::device_vector<UInt>& getNeighborCounts() const;
	const thrust::device_vector<UInt>& getNeighorOffsets() const;

	const thrust::device_vector<UInt>& getCellParticleCounts() const;
	const thrust::device_vector<UInt>& getCellParticleOffsets() const;
	UInt getCellMortonLinerIndex(Vector3 vPos) const;
	UInt getCellMortonLinerIndex(Vector3ui vIndex3D) const;

	void setMetaGridGroupSize(UInt vInput);
	void setSearchRadius(Real vInput);
	void setCellSize(Real vInput);

	UInt getNeighorDataSize() const;
	UInt getMetaGridGroupSize() const;
	UInt getMetaGridBlockSize() const;
	Real getSearchRadius() const;
	Real getCellSize() const;
	UInt getNeighborCount(UInt vIndex) const;
	Vector3 getGridMin() const;
	Vector3 getGridMax() const;
	SGridInfo getGridInfo() const;

private:
	SGridInfo m_GridInfo;

	UInt m_MetaGridGroupSize = 8;
	Real m_SearchRadius = static_cast<Real>(0.1);
	Real m_CellSize = static_cast<Real>(0.1);

	shared_ptr<CParticleGroup> m_TargetParticleGroup = nullptr;
	thrust::device_vector<Real> m_SortCache;

	thrust::device_vector<UInt> m_CellParticleOffsets;
	thrust::device_vector<UInt> m_CellParticleCounts;

	thrust::device_vector<UInt> m_ParticleCellIndices;
	thrust::device_vector<UInt> m_TempSortIndices;
	thrust::device_vector<UInt> m_SortIndices;

	thrust::device_vector<UInt> m_Neighbors;
	thrust::device_vector<UInt> m_NeighborCounts;
	thrust::device_vector<UInt> m_NeighborOffsets;

	void __updateGrid();
};


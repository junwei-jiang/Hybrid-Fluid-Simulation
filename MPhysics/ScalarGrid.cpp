#include "ScalarGrid.h"

CScalarGrid::CScalarGrid()
{

}

CScalarGrid::~CScalarGrid()
{

}

const thrust::device_vector<Real>& CScalarGrid::getConstGridData() const
{
	return m_Data;
}

thrust::device_vector<Real>& CScalarGrid::getGridData()
{
	return m_Data;
}

const Real* CScalarGrid::getConstGridDataGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_Data);
}

Real* CScalarGrid::getGridDataGPUPtr()
{
	return getRawDevicePointerReal(m_Data);
}

void CScalarGrid::resizeData(Vector3i vRes)
{
	resizeDeviceVector(m_Data, vRes.x * vRes.y * vRes.z);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_Data), 0, vRes.x * vRes.y * vRes.z * sizeof(Real)));
}

void CScalarGrid::resizeData(const Real* vStartPtr, const Real* vEndPtr)
{
	assignDeviceVectorReal(m_Data, vStartPtr, vEndPtr);
}

void CScalarGrid::resizeData(const thrust::device_vector<Real>& vDeviceVector)
{
	assignDeviceVectorReal(vDeviceVector, m_Data);
}
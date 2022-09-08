#include "VectorGrid.h"

CVectorGrid::CVectorGrid()
{

}

CVectorGrid::~CVectorGrid()
{

}

const thrust::device_vector<Real>& CVectorGrid::getConstGridDataX() const
{
	return m_DataX;
}

const thrust::device_vector<Real>& CVectorGrid::getConstGridDataY() const
{
	return m_DataY;
}

const thrust::device_vector<Real>& CVectorGrid::getConstGridDataZ() const
{
	return m_DataZ;
}

thrust::device_vector<Real>& CVectorGrid::getGridDataX()
{
	return m_DataX;
}

thrust::device_vector<Real>& CVectorGrid::getGridDataY()
{
	return m_DataY;
}

thrust::device_vector<Real>& CVectorGrid::getGridDataZ()
{
	return m_DataZ;
}

const Real* CVectorGrid::getConstGridDataXGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_DataX);
}

const Real* CVectorGrid::getConstGridDataYGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_DataY);
}

const Real* CVectorGrid::getConstGridDataZGPUPtr() const
{
	return getReadOnlyRawDevicePointer(m_DataZ);
}

Real* CVectorGrid::getGridDataXGPUPtr()
{
	return getRawDevicePointerReal(m_DataX);
}

Real* CVectorGrid::getGridDataYGPUPtr()
{
	return getRawDevicePointerReal(m_DataY);
}

Real* CVectorGrid::getGridDataZGPUPtr()
{
	return getRawDevicePointerReal(m_DataZ);
}

void CVectorGrid::resizeDataX(Vector3i vResX)
{
	resizeDeviceVector(m_DataX, vResX.x * vResX.y * vResX.z);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_DataX), 0, vResX.x * vResX.y * vResX.z * sizeof(Real)));
}

void CVectorGrid::resizeDataY(Vector3i vResY)
{
	resizeDeviceVector(m_DataY, vResY.x * vResY.y * vResY.z);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_DataY), 0, vResY.x * vResY.y * vResY.z * sizeof(Real)));
}

void CVectorGrid::resizeDataZ(Vector3i vResZ)
{
	resizeDeviceVector(m_DataZ, vResZ.x * vResZ.y * vResZ.z);
	CHECK_CUDA(cudaMemset(getRawDevicePointerReal(m_DataZ), 0, vResZ.x * vResZ.y * vResZ.z * sizeof(Real)));
}

void CVectorGrid::resizeDataX(const Real* vStartPtrX, const Real* vEndPtrX)
{
	assignDeviceVectorReal(m_DataX, vStartPtrX, vEndPtrX);
}

void CVectorGrid::resizeDataY(const Real* vStartPtrY, const Real* vEndPtrY)
{
	assignDeviceVectorReal(m_DataY, vStartPtrY, vEndPtrY);
}

void CVectorGrid::resizeDataZ(const Real* vStartPtrZ, const Real* vEndPtrZ)
{
	assignDeviceVectorReal(m_DataZ, vStartPtrZ, vEndPtrZ);
}

void CVectorGrid::resizeDataX(const thrust::device_vector<Real>& vDeviceVectorX)
{
	assignDeviceVectorReal(vDeviceVectorX, m_DataX);
}

void CVectorGrid::resizeDataY(const thrust::device_vector<Real>& vDeviceVectorY)
{
	assignDeviceVectorReal(vDeviceVectorY, m_DataY);
}

void CVectorGrid::resizeDataZ(const thrust::device_vector<Real>& vDeviceVectorZ)
{
	assignDeviceVectorReal(vDeviceVectorZ, m_DataZ);
}